"""
preprocessing.py — Data cleaning, normalization, date handling, and splitting.
"""
import re
import numpy as np
import pandas as pd
from functools import wraps
import time
from sklearn.model_selection import StratifiedShuffleSplit


# ─── Text normalization ─────────────────────────────────────────────

def normalize(text):
    """Normalize a single text value to lowercase, stripped, with collapsed whitespace."""
    s = "" if text is None else str(text)
    s = s.strip().lower()

    if s in {"", "none", "na", "n/a", "nan", "null"}:
        return ""

    s = re.sub(r"\W+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_and_combine(df, columns_to_normalize):
    """Normalize specified columns and create a `combined_text` column."""
    cols = [c for c in columns_to_normalize if c in df.columns]

    if not cols:
        df["combined_text"] = ""
        return df

    df["combined_text"] = df[cols].apply(
        lambda row: " ".join(v for v in (normalize(x) for x in row) if v),
        axis=1,
    )
    return df


def rebuild_combined_text_for_row(row: pd.Series, columns_to_normalize: list) -> str:
    """
    Rebuild combined_text for a single row (e.g., after changing Trade Id).
    Used in episode builder to ensure query text differs from positive candidate.
    """
    cols = [c for c in columns_to_normalize if c in row.index]
    if not cols:
        return ""

    pieces = []
    for col in cols:
        val = row.get(col, None)
        normalized = normalize(val) if val is not None else ""
        if normalized:
            pieces.append(normalized)

    return " ".join(pieces)


# ─── Date handling ───────────────────────────────────────────────────

def add_date_int_cols(df, date_cols, epoch=pd.Timestamp("1970-01-01")):
    """Create integer date columns (days since epoch) for each date column."""
    for c in date_cols:
        if c not in df.columns:
            df[c] = pd.NaT

        df[c] = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_int"] = (df[c] - epoch).dt.days.astype("float")

    return df


# ─── Data splitting ──────────────────────────────────────────────────

def stratified_group_split_3way(
    df, group_col="Match ID", strat_col="Comments",
    train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Split df into train/val/test without leaking groups across folds.
    Stratification is applied using the group's strat_col.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-9

    g = df[[group_col, strat_col]].drop_duplicates(subset=[group_col]).rename(
        columns={strat_col: "strat_val"}
    )

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_g_idx, hold_g_idx = next(sss1.split(g, g["strat_val"]))
    g_train = g.iloc[train_g_idx]
    g_hold = g.iloc[hold_g_idx]

    val_ratio = val_size / (val_size + test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=random_state + 1)
    val_g_idx, test_g_idx = next(sss2.split(g_hold, g_hold["strat_val"]))
    g_val = g_hold.iloc[val_g_idx]
    g_test = g_hold.iloc[test_g_idx]

    df_train = df[df[group_col].isin(g_train[group_col])]
    df_val = df[df[group_col].isin(g_val[group_col])]
    df_test = df[df[group_col].isin(g_test[group_col])]

    # Verify disjoint groups
    tr_ids = set(g_train[group_col])
    va_ids = set(g_val[group_col])
    te_ids = set(g_test[group_col])
    assert tr_ids.isdisjoint(va_ids), "train and val share groups"
    assert tr_ids.isdisjoint(te_ids), "train and test share groups"
    assert va_ids.isdisjoint(te_ids), "val and test share groups"

    return df_train.copy(), df_val.copy(), df_test.copy()


# ─── Utilities ───────────────────────────────────────────────────────

def timer(func):
    """Simple timing decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.2f}s")
        return result
    return wrapper
