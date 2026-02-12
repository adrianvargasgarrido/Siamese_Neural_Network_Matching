"""
candidate_generation.py — Candidate retrieval, filtering, ranking, and episode construction.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from .data_prep import rebuild_combined_text_for_row


# ─── Filtering helpers ───────────────────────────────────────────────

def apply_amount_filter(
    subset: pd.DataFrame,
    query_amount,
    amount_tol_pct: float,
    amount_col: str,
    enforce_same_sign: bool = False,
):
    """Filter candidates by relative amount tolerance and optional same-sign rule."""
    subset = subset.copy()

    if pd.isna(query_amount):
        subset["amount_diff"] = np.nan
        subset["amount_diff_pct"] = np.nan
        return subset

    subset_amounts = pd.to_numeric(subset[amount_col], errors="coerce").values.astype(float)
    q_amt = float(query_amount)

    amount_diff = np.abs(subset_amounts - q_amt)
    denom = max(abs(q_amt), 1e-12)
    amount_pct = amount_diff / denom

    mask = amount_pct <= float(amount_tol_pct)

    if enforce_same_sign:
        q_sign = np.sign(q_amt)
        subset_signs = np.sign(subset_amounts)
        mask = mask & (subset_signs == q_sign)

    out = subset.loc[mask].copy()
    out["amount_diff"] = amount_diff[mask]
    out["amount_diff_pct"] = amount_pct[mask]
    return out


def apply_ranking_and_topk(
    subset: pd.DataFrame,
    query_row: pd.Series,
    top_k: int,
    ref_col=None,
    id_col: str = "Trade Id",
):
    """Stable, deterministic ranking by ref_exact (desc), amount_diff (asc), date_diff (asc)."""
    if subset.empty:
        return subset

    subset = subset.copy()

    if ref_col and (ref_col in subset.columns) and (ref_col in query_row.index):
        qref = query_row.get(ref_col, None)
        subset["ref_exact"] = (subset[ref_col] == qref).astype(int)
    else:
        subset["ref_exact"] = 0

    sort_cols = ["ref_exact", "amount_diff"]
    sort_asc = [False, True]
    if "date_diff" in subset.columns:
        sort_cols.append("date_diff")
        sort_asc.append(True)

    ranked = subset.sort_values(
        by=sort_cols, ascending=sort_asc, kind="mergesort", na_position="last"
    )

    if id_col in ranked.columns:
        ranked = ranked.drop_duplicates(subset=[id_col], keep="first")

    return ranked.head(int(top_k))


# ─── Candidate retrieval ─────────────────────────────────────────────

def get_candidates(
    query_row: pd.Series,
    pool_df: pd.DataFrame,
    *,
    id_col: str,
    currency_col: str,
    amount_col: str,
    date_int_cols: list,
    ref_col=None,
    matched_by_currency=None,
    top_k: int = 50,
    window_days: int = 20,
    amount_tol_pct: float = 0.30,
    date_policy: str = "any",
    enforce_same_sign: bool = False,
) -> pd.DataFrame:
    """
    Generate candidate rows for a single query using blocking:
    currency → date window → amount tolerance → rank & top-K.
    """
    empty_result = pd.DataFrame(
        columns=["a_id", "b_id", "ref_exact", "amount_diff", "amount_diff_pct", "date_diff"]
    )

    currency = query_row.get(currency_col, np.nan)
    if pd.isna(currency):
        return empty_result

    if matched_by_currency is not None:
        if currency not in matched_by_currency.groups:
            return empty_result
        subset = matched_by_currency.get_group(currency).copy()
    else:
        subset = pool_df[pool_df[currency_col] == currency].copy()
        if subset.empty:
            return empty_result

    # Exclude self
    a_id_val = query_row.get(id_col, None)
    if a_id_val is not None and id_col in subset.columns:
        subset = subset[subset[id_col] != a_id_val]

    # Date window
    valid_cols = [c for c in date_int_cols if pd.notna(query_row.get(c, np.nan))]
    if valid_cols:
        for c in valid_cols:
            if c not in subset.columns:
                subset[c] = np.nan

        subset_dates = subset[valid_cols].values.astype(float)
        query_dates = np.array([query_row[c] for c in valid_cols], dtype=float)
        date_diffs = np.abs(subset_dates - query_dates)
        has_any = ~np.all(np.isnan(date_diffs), axis=1)

        agg = np.full(date_diffs.shape[0], np.inf)
        if has_any.any():
            if date_policy == "all":
                agg[has_any] = np.nanmax(date_diffs[has_any], axis=1)
            else:
                agg[has_any] = np.nanmin(date_diffs[has_any], axis=1)

        mask = agg <= float(window_days)
        subset = subset.loc[mask].copy()
        subset["date_diff"] = agg[mask]

        if subset.empty:
            return empty_result

    # Amount tolerance
    query_amount = pd.to_numeric(query_row.get(amount_col, np.nan), errors="coerce")
    subset = apply_amount_filter(
        subset, query_amount, amount_tol_pct,
        amount_col=amount_col, enforce_same_sign=enforce_same_sign,
    )
    if subset.empty:
        return empty_result

    # Ranking + Top-K
    result = apply_ranking_and_topk(
        subset, query_row, top_k=top_k, ref_col=ref_col, id_col=id_col
    )
    if result.empty:
        return result

    result["a_id"] = a_id_val
    result["b_id"] = result[id_col] if id_col in result.columns else None

    if "ref_exact" not in result.columns:
        result["ref_exact"] = 0

    if "b_id" in result.columns:
        result = result.drop_duplicates(subset=["b_id"], keep="first")

    return result


# ─── Episode builder ─────────────────────────────────────────────────

def build_training_episodes_single_df_debug(
    df_pool: pd.DataFrame,
    *,
    id_col: str,
    currency_col: str,
    amount_col: str,
    date_int_cols: list,
    columns_to_normalize: list,
    ref_col=None,
    n_episodes: int = 2_000,
    train_k_neg: int = 20,
    rule_col: str = "Match Rule",
    matched_by_currency=None,
    window_days: Optional[int] = None,
    amount_tol_pct: Optional[float] = None,
    date_policy: Optional[str] = None,
    enforce_same_sign: bool = True,
    random_state: Optional[int] = None,
    id_norm_col: str = "Trade Id",
    debug_limit: Optional[int] = 5,
    # Default fallbacks
    default_window_days: int = 20,
    default_amount_tol_pct: float = 0.30,
    default_date_policy: str = "any",
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Build training episodes with candidate retrieval and debug output.

    Returns:
      - episodes: list of dicts with query_row, candidates_df, positive_index, etc.
      - candidates_long_df: columns ["AID", "BID", "rank", "label", "rule"]
    """

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    if df_pool is None or df_pool.empty:
        print(f"[{_ts()}] df_pool is empty -> returning empty outputs.")
        return [], pd.DataFrame(columns=["AID", "BID", "rank", "label", "rule"])

    if id_col not in df_pool.columns:
        raise ValueError(f"df_pool must contain id_col='{id_col}'")
    if id_norm_col not in df_pool.columns:
        raise ValueError(f"'{id_norm_col}' not found in df_pool.")
    for c in date_int_cols:
        if c not in df_pool.columns:
            raise ValueError(f"Missing required date-int column '{c}' in df_pool.")

    rng = np.random.default_rng(seed=random_state)

    def _seed() -> int:
        return int(rng.integers(0, 2**32 - 1))

    n_sample = min(int(n_episodes), len(df_pool))
    base = df_pool.sample(n_sample, replace=False, random_state=_seed())

    episodes: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    pool_cols = list(df_pool.columns)
    pool_ids_set = set(df_pool[id_norm_col].unique())

    for epi_idx, row_vals in enumerate(base.itertuples(index=False, name=None), start=1):
        row_b_pos = dict(zip(base.columns, row_vals))
        rule_value = row_b_pos.get(rule_col, "UNKNOWN_RULE")
        true_id = row_b_pos.get(id_col)

        print(f"\n[{_ts()}] ==== EPISODE {epi_idx} ====")
        print(f"[{_ts()}] true_id={true_id} | rule={rule_value}")

        # Query A = clone(B) with synthetic ID
        a_row = dict(row_b_pos)
        a_id = f"Q::{true_id if true_id is not None else 'NA'}::{uuid4().hex}"
        a_row[id_col] = a_id

        # Rebuild combined_text after changing Trade Id
        a_row_series = pd.Series(a_row)
        a_row["combined_text"] = rebuild_combined_text_for_row(
            a_row_series, columns_to_normalize
        )

        print(f"[{_ts()}] a_id={a_id}")
        print(f"[{_ts()}] Query combined_text (first 80 chars): {a_row.get('combined_text', '')[:80]}")
        print(f"[{_ts()}] Positive combined_text (first 80 chars): {row_b_pos.get('combined_text', '')[:80]}")

        _window_days = window_days if window_days is not None else default_window_days
        _amount_tol_pct = amount_tol_pct if amount_tol_pct is not None else default_amount_tol_pct
        _date_policy = date_policy if date_policy is not None else default_date_policy

        print(
            f"[{_ts()}] CALL get_candidates("
            f"top_k={int(train_k_neg)*2}, window_days={_window_days}, "
            f"amount_tol_pct={_amount_tol_pct}, date_policy={_date_policy}, "
            f"enforce_same_sign={enforce_same_sign})"
        )

        neg_cands = get_candidates(
            query_row=pd.Series(a_row),
            pool_df=df_pool,
            id_col=id_col,
            currency_col=currency_col,
            amount_col=amount_col,
            date_int_cols=date_int_cols,
            ref_col=ref_col,
            matched_by_currency=matched_by_currency,
            top_k=int(train_k_neg) * 2,
            window_days=_window_days,
            amount_tol_pct=_amount_tol_pct,
            date_policy=_date_policy,
            enforce_same_sign=enforce_same_sign,
        )

        if isinstance(neg_cands, pd.DataFrame):
            print(f"[{_ts()}] neg_cands shape={neg_cands.shape}")
            if not neg_cands.empty and "b_id" in neg_cands.columns:
                uniq_b = pd.unique(neg_cands["b_id"]).tolist()
                print(f"[{_ts()}] retriever b_id unique count={len(uniq_b)}")

        # Exclusion
        ex_set = {a_id}
        if true_id is not None:
            ex_set.add(true_id)

        neg_ids: List[Any] = []
        if isinstance(neg_cands, pd.DataFrame) and not neg_cands.empty and "b_id" in neg_cands.columns:
            seen = set()
            for bid in neg_cands["b_id"]:
                if bid in ex_set or bid in seen:
                    continue
                if bid not in pool_ids_set:
                    continue
                seen.add(bid)
                neg_ids.append(bid)
                if len(neg_ids) >= int(train_k_neg):
                    break
        print(f"[{_ts()}] neg_ids after retriever: {len(neg_ids)}")

        # Top-up
        if len(neg_ids) < int(train_k_neg):
            need = int(train_k_neg) - len(neg_ids)
            ex_all = set(neg_ids) | ex_set
            allow_df = df_pool.loc[~df_pool[id_norm_col].isin(ex_all)]
            allow_unique_ids = allow_df[id_norm_col].drop_duplicates()

            print(f"[{_ts()}] Top-up needed={need}; allowed pool={len(allow_unique_ids)}")
            if not allow_unique_ids.empty:
                take = min(need, len(allow_unique_ids))
                picked = allow_unique_ids.sample(n=take, replace=False, random_state=_seed()).tolist()
                neg_ids.extend(picked)

        print(f"[{_ts()}] final neg_ids: {len(neg_ids)}")

        # Materialize negatives
        if neg_ids:
            neg_rows = df_pool[df_pool[id_norm_col].isin(neg_ids)].copy()
            neg_rows["_ord"] = pd.Categorical(neg_rows[id_norm_col], categories=neg_ids, ordered=True)
            neg_rows = neg_rows.sort_values("_ord").drop(columns="_ord")
            neg_rows = neg_rows.drop_duplicates(subset=[id_norm_col])
        else:
            neg_rows = df_pool.iloc[0:0].copy()

        # Assemble candidates = [B_positive] + negatives
        b_pos_df = pd.DataFrame([row_b_pos], columns=pool_cols)
        candidates_df = pd.concat([b_pos_df, neg_rows], ignore_index=True)
        positive_index = 0

        assert a_id not in set(candidates_df[id_col]), \
            f"Query id {a_id} unexpectedly present in candidates_df"

        candidate_ids = candidates_df[id_col].tolist()
        print(f"[{_ts()}] candidates_df.shape={candidates_df.shape}; positive_index={positive_index}")

        episodes.append({
            "query_row": pd.Series(a_row),
            "candidates_df": candidates_df,
            "positive_index": positive_index,
            "rule": rule_value,
            "query_id": a_id,
            "candidate_ids": candidate_ids,
        })

        for rank, bid in enumerate(candidate_ids):
            long_rows.append({
                "AID": a_id,
                "BID": bid,
                "rank": rank,
                "label": 1 if rank == 0 else 0,
                "rule": rule_value,
            })

        if debug_limit is not None and epi_idx >= debug_limit:
            print(f"[{_ts()}] Debug limit reached ({debug_limit}). Stopping.")
            break

    candidates_long_df = pd.DataFrame(long_rows, columns=["AID", "BID", "rank", "label", "rule"])
    print(f"[{_ts()}] done; episodes={len(episodes)}, long_rows={len(long_rows)}")
    return episodes, candidates_long_df


# ─── Single-episode worker (top-level for pickling) ──────────────────

def _build_one_episode(
    row_b_pos: dict,
    df_pool: pd.DataFrame,
    pool_ids_set: set,
    *,
    id_col: str,
    currency_col: str,
    amount_col: str,
    date_int_cols: list,
    columns_to_normalize: list,
    ref_col,
    train_k_neg: int,
    rule_col: str,
    window_days: int,
    amount_tol_pct: float,
    date_policy: str,
    enforce_same_sign: bool,
    id_norm_col: str,
    random_seed: int,
) -> Optional[Dict[str, Any]]:
    """Process one episode: create query, retrieve candidates, assemble result."""
    rng = np.random.default_rng(seed=random_seed)
    rule_value = row_b_pos.get(rule_col, "UNKNOWN_RULE")
    true_id = row_b_pos.get(id_col)
    pool_cols = list(df_pool.columns)

    # Clone positive row to create query with synthetic ID
    a_row = dict(row_b_pos)
    a_id = f"Q::{true_id if true_id is not None else 'NA'}::{uuid4().hex}"
    a_row[id_col] = a_id
    a_row["combined_text"] = rebuild_combined_text_for_row(
        pd.Series(a_row), columns_to_normalize
    )

    # Retrieve hard-negative candidates via blocking
    neg_cands = get_candidates(
        query_row=pd.Series(a_row),
        pool_df=df_pool,
        id_col=id_col,
        currency_col=currency_col,
        amount_col=amount_col,
        date_int_cols=date_int_cols,
        ref_col=ref_col,
        top_k=train_k_neg * 2,
        window_days=window_days,
        amount_tol_pct=amount_tol_pct,
        date_policy=date_policy,
        enforce_same_sign=enforce_same_sign,
    )

    # Collect negative IDs, excluding query and positive
    ex_set = {a_id}
    if true_id is not None:
        ex_set.add(true_id)

    neg_ids: List[Any] = []
    if isinstance(neg_cands, pd.DataFrame) and not neg_cands.empty and "b_id" in neg_cands.columns:
        seen = set()
        for bid in neg_cands["b_id"]:
            if bid in ex_set or bid in seen or bid not in pool_ids_set:
                continue
            seen.add(bid)
            neg_ids.append(bid)
            if len(neg_ids) >= train_k_neg:
                break

    # Random top-up if blocking returned fewer than needed
    if len(neg_ids) < train_k_neg:
        need = train_k_neg - len(neg_ids)
        ex_all = set(neg_ids) | ex_set
        allowed = [tid for tid in pool_ids_set if tid not in ex_all]
        if allowed:
            take = min(need, len(allowed))
            picked = list(rng.choice(allowed, size=take, replace=False))
            neg_ids.extend(picked)

    # Materialise negative rows
    if neg_ids:
        neg_rows = df_pool[df_pool[id_norm_col].isin(neg_ids)].copy()
        neg_rows["_ord"] = pd.Categorical(
            neg_rows[id_norm_col], categories=neg_ids, ordered=True
        )
        neg_rows = neg_rows.sort_values("_ord").drop(columns="_ord")
        neg_rows = neg_rows.drop_duplicates(subset=[id_norm_col])
    else:
        neg_rows = df_pool.iloc[0:0].copy()

    # Assemble: positive first, then negatives
    b_pos_df = pd.DataFrame([row_b_pos], columns=pool_cols)
    candidates_df = pd.concat([b_pos_df, neg_rows], ignore_index=True)
    candidate_ids = candidates_df[id_col].tolist()

    return {
        "query_row": pd.Series(a_row),
        "candidates_df": candidates_df,
        "positive_index": 0,
        "rule": rule_value,
        "query_id": a_id,
        "candidate_ids": candidate_ids,
    }


# ─── Parallel episode builder ────────────────────────────────────────

def build_training_episodes_parallel(
    df_pool: pd.DataFrame,
    *,
    id_col: str,
    currency_col: str,
    amount_col: str,
    date_int_cols: list,
    columns_to_normalize: list,
    ref_col=None,
    n_episodes: int = 2_000,
    train_k_neg: int = 20,
    rule_col: str = "Match Rule",
    window_days: int = 20,
    amount_tol_pct: float = 0.30,
    date_policy: str = "any",
    enforce_same_sign: bool = True,
    random_state: Optional[int] = None,
    id_norm_col: str = "Trade Id",
    max_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Parallel version of episode construction.

    Uses ProcessPoolExecutor to build multiple episodes concurrently.
    Each worker independently creates a query, retrieves candidates via
    blocking, and assembles the episode dict.

    Parameters are the same as ``build_training_episodes_single_df_debug``
    minus the debug/print options.  ``max_workers`` controls concurrency
    (defaults to cpu_count - 1).

    Returns
    -------
    episodes : list[dict]
    candidates_long_df : pd.DataFrame  with columns [AID, BID, rank, label, rule]
    """
    t0 = datetime.now()

    if df_pool is None or df_pool.empty:
        return [], pd.DataFrame(columns=["AID", "BID", "rank", "label", "rule"])

    rng = np.random.default_rng(seed=random_state)
    n_sample = min(int(n_episodes), len(df_pool))
    base = df_pool.sample(
        n_sample, replace=False,
        random_state=int(rng.integers(0, 2**32 - 1)),
    )

    pool_ids_set = set(df_pool[id_norm_col].unique())

    # Pre-generate one seed per episode for reproducibility
    seeds = rng.integers(0, 2**32 - 1, size=n_sample).tolist()

    # Convert sampled rows to list of dicts (avoids shipping the whole DF index)
    base_rows = [dict(zip(base.columns, vals)) for vals in base.itertuples(index=False, name=None)]

    shared_kwargs = dict(
        id_col=id_col,
        currency_col=currency_col,
        amount_col=amount_col,
        date_int_cols=date_int_cols,
        columns_to_normalize=columns_to_normalize,
        ref_col=ref_col,
        train_k_neg=train_k_neg,
        rule_col=rule_col,
        window_days=window_days,
        amount_tol_pct=amount_tol_pct,
        date_policy=date_policy,
        enforce_same_sign=enforce_same_sign,
        id_norm_col=id_norm_col,
    )

    episodes: List[Dict[str, Any]] = [None] * n_sample  # preserve order
    errors = 0

    if max_workers is None:
        import os as _os
        max_workers = max(1, (_os.cpu_count() or 2) - 1)

    print(
        f"Building {n_sample} episodes with {max_workers} workers …"
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _build_one_episode,
                row,
                df_pool,
                pool_ids_set,
                random_seed=seeds[i],
                **shared_kwargs,
            ): i
            for i, row in enumerate(base_rows)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                episodes[idx] = future.result()
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    print(f"  ⚠ Episode {idx} failed: {exc}")

    # Drop any failed episodes
    episodes = [ep for ep in episodes if ep is not None]

    # Build the long-format diagnostics table
    long_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        for rank, bid in enumerate(ep["candidate_ids"]):
            long_rows.append({
                "AID": ep["query_id"],
                "BID": bid,
                "rank": rank,
                "label": 1 if rank == 0 else 0,
                "rule": ep["rule"],
            })

    candidates_long_df = pd.DataFrame(
        long_rows, columns=["AID", "BID", "rank", "label", "rule"]
    )

    elapsed = (datetime.now() - t0).total_seconds()
    print(
        f"✅ Done: {len(episodes)} episodes, "
        f"{errors} errors, {elapsed:.1f}s"
    )
    return episodes, candidates_long_df
