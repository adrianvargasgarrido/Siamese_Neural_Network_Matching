"""
vectorization.py — TF-IDF text iteration, scalar/pair feature engineering, episode vectorization.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Corpus iterator ─────────────────────────────────────────────────

def iter_episode_text(episodes):
    """Yield all `combined_text` strings from queries and candidates in episodes."""
    for ep in episodes:
        q = ep["query_row"].get("combined_text", "") or ""
        yield q

        df = ep["candidates_df"]
        if df is not None and "combined_text" in df.columns and not df.empty:
            for (txt,) in df[["combined_text"]].itertuples(index=False, name=None):
                yield txt or ""


# ─── Scalarization ───────────────────────────────────────────────────

def _scalarize_frame(df: pd.DataFrame, amount_col: str, date_cols: list):
    """
    Vectorized scalarization for a whole DataFrame.

    Returns:
        scal : (K, 2) -> [log1p(|amount|), min_date_norm]
        amt  : (K,)   -> raw float amounts
    """
    amt = (
        pd.to_numeric(df.get(amount_col, 0.0), errors="coerce")
        .fillna(0.0).astype(float).to_numpy()
    )

    cols_int = [f"{c}_int" for c in date_cols if f"{c}_int" in df.columns]
    if cols_int:
        min_int = (
            df[cols_int].apply(pd.to_numeric, errors="coerce")
            .min(axis=1).fillna(0.0).astype(float).to_numpy()
        )
    else:
        min_int = np.zeros(len(df), dtype=float)

    min_date_norm = (min_int / 365.0).astype(np.float32)
    scal = np.stack(
        [np.log1p(np.abs(amt)).astype(np.float32), min_date_norm],
        axis=1,
    )
    return scal, amt


def _scalarize_row(s: pd.Series, amount_col: str, date_cols: list):
    """Single-row scalarization, delegating to the vectorized implementation."""
    scal, amt = _scalarize_frame(pd.DataFrame([s]), amount_col, date_cols)
    return scal[0], float(amt[0])


def _min_date_diff_wrt_query(query_row: pd.Series, C: pd.DataFrame, date_cols: list):
    """
    Compute per-candidate min date absolute difference (days) w.r.t. query_row.

    Returns: np.ndarray shape (K,), NaN-safe.
    """
    K = len(C)
    if K == 0:
        return np.zeros(0, dtype=np.float32)

    stacks = []
    for c in date_cols:
        ai = query_row.get(f"{c}_int", np.nan)
        if f"{c}_int" in C.columns:
            bi = pd.to_numeric(C[f"{c}_int"], errors="coerce").astype(float).values
        else:
            bi = np.full(K, np.nan, dtype=float)

        if pd.isna(ai):
            stacks.append(np.full(K, np.nan, dtype=float))
        else:
            stacks.append(np.abs(bi - float(ai)))

    if not stacks:
        return np.zeros(K, dtype=np.float32)

    arr = np.vstack(stacks)
    has_data = np.isfinite(arr).any(axis=0)
    out = np.zeros(K, dtype=np.float32)
    if has_data.any():
        mins = np.nanmin(arr[:, has_data], axis=0)
        out[has_data] = np.where(np.isfinite(mins), mins, 0.0).astype(np.float32)
    return out


# ─── Episode vectorization ───────────────────────────────────────────

def vectorize_episode(
    query_row: pd.Series,
    candidates_df: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer,
    amount_col: str,
    date_cols: list,
    ref_col=None,
):
    """
    Vectorize a single episode into tensors for the model.

    Returns:
        vec_q  : (T,)
        scal_q : (2,)
        vec_C  : (K, T)
        scal_C : (K, 2)
        pair_C : (K, 3)
        pos_ix : 0
    """
    if vectorizer is None:
        raise ValueError("vectorize_episode requires a fitted `vectorizer` argument.")

    # Query
    txt_q = query_row.get("combined_text", "") or ""
    vec_q = vectorizer.transform([txt_q]).toarray().astype(np.float32)[0]
    scal_q, q_amt = _scalarize_row(query_row, amount_col, date_cols)

    # Candidates
    if candidates_df is None or candidates_df.empty:
        T = vec_q.shape[0]
        return (
            vec_q, scal_q,
            np.zeros((0, T), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0, 3), np.float32),
            0,
        )

    texts = (
        candidates_df["combined_text"].fillna("")
        if "combined_text" in candidates_df.columns
        else [""] * len(candidates_df)
    )

    vec_C = vectorizer.transform(texts).toarray().astype(np.float32)
    assert vec_C.shape[1] == vec_q.shape[0], \
        "TF-IDF vector length mismatch between query and candidates."

    scal_C, amt = _scalarize_frame(candidates_df, amount_col, date_cols)

    # Pair features
    log_amt_diff = np.log1p(np.abs(q_amt - amt)).astype(np.float32)
    min_date_diff = _min_date_diff_wrt_query(query_row, candidates_df, date_cols).astype(np.float32)
    log_min_date_diff = np.log1p(min_date_diff).astype(np.float32)

    ref_exact = np.zeros(len(candidates_df), dtype=np.float32)
    if ref_col and (ref_col in query_row.index) and (ref_col in candidates_df.columns):
        ra = query_row.get(ref_col, None)
        if pd.notna(ra):
            ref_exact = (candidates_df[ref_col].to_numpy() == ra).astype(np.float32)

    pair_C = np.stack(
        [log_amt_diff, log_min_date_diff, ref_exact], axis=1
    ).astype(np.float32)

    pos_ix = 0
    return vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix
