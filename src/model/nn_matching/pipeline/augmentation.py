"""
augmentation.py — Optional data augmentation for Siamese trade matching.

Applies realistic noise to the QUERY SIDE ONLY of training episodes,
teaching the model that the same trade can look different at entry time.

Architecture: Option B — noise is applied inside RankingEpisodeDataset.__getitem__,
so every epoch sees a DIFFERENT perturbation of the same episode.

┌─────────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION PIPELINE                            │
│                                                                     │
│  Episode dict ──► __getitem__ ──► augment_query(row) ──►           │
│                                        │                            │
│                                ┌───────┴────────┐                   │
│                         Text perturbations  Scalar perturbations    │
│                         (on combined_text)  (on amount/dates)       │
│                                │                    │               │
│                                ▼                    ▼               │
│                     vectorize_episode(modified_row, candidates)     │
│                                        │                            │
│                              ┌─────────┴──────────┐                │
│                         TF-IDF vector          pair_features        │
│                    (reflects noisy text)   (derived from noisy      │
│                                            scalars — NOT injected   │
│                                            directly)                │
│                                        │                            │
│                                   Model input                       │
└─────────────────────────────────────────────────────────────────────┘

WHY EACH PERTURBATION EXISTS:
─────────────────────────────
1. token_dropout   — Simulates missing fields / partial data feeds.
2. token_swap      — Simulates reordered fields, copy-paste errors.
3. char_noise      — Simulates typos, OCR errors, keyboard mistakes.
4. synonym_sub     — Simulates different naming conventions across systems.
5. field_omission  — Simulates an entire column missing from a source system.
6. scalar_perturb  — Simulates rounding, FX slippage, T+1/T+2 date conventions.

ANTI-PATTERNS (things this module does NOT do):
─────────────────────────────────────────────────
• Never adds noise to pair features — they are derived, not independent.
• Never corrupts both query AND candidate — only query side A.
• Never adds Gaussian noise to TF-IDF vectors — that breaks sparsity.
• Never uses unrealistic magnitudes (>5% amount, ±30 days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import rebuild_combined_text_for_row


# ═══════════════════════════════════════════════════════════════════
#  1. TEXT PERTURBATIONS — operate on the combined_text string
# ═══════════════════════════════════════════════════════════════════


def token_dropout(
    text: str,
    drop_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Randomly remove tokens (words) from the text.

    What it simulates:
        Missing fields in source systems, partial data feeds, incomplete
        manual entry.  E.g. one system has "BUY 1000 APPLE INC SHARES"
        while another only has "BUY APPLE INC".

    How it works:
        1. Split text on whitespace → list of tokens.
        2. For each token, flip a coin with probability `drop_prob`.
           If heads → remove the token.
        3. Safety: always keep at least 1 token (never return "").
        4. Rejoin remaining tokens with spaces.

    Effect on TF-IDF:
        Dropped tokens remove their character n-grams from the sparse
        vector. The model learns that a PARTIAL text can still match
        the full version — it cannot rely on every token being present.

    Parameters
    ----------
    text : str
        The combined_text string (already normalised).
    drop_prob : float
        Probability of dropping each individual token. Typical: 0.10–0.20.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Example
    -------
    >>> token_dropout("buy 1000 apple inc shares cusip 037833100", drop_prob=0.3)
    "buy apple inc cusip 037833100"
    # "1000" and "shares" were dropped
    """
    if rng is None:
        rng = np.random.default_rng()

    tokens = text.split()
    if len(tokens) <= 1:
        return text

    # Decide which tokens to keep
    keep_mask = rng.random(len(tokens)) >= drop_prob

    # Safety: never drop ALL tokens
    if not keep_mask.any():
        keep_mask[rng.integers(0, len(tokens))] = True

    return " ".join(t for t, keep in zip(tokens, keep_mask) if keep)


def token_swap(
    text: str,
    swap_prob: float = 0.10,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Swap two adjacent tokens in the text.

    What it simulates:
        Reordered fields across systems ("LAST FIRST" vs "FIRST LAST"),
        copy-paste errors, different system field ordering conventions.

    How it works:
        1. Split text on whitespace → list of tokens.
        2. With probability `swap_prob`, pick a random index i and swap
           tokens[i] with tokens[i+1].
        3. At most one swap per call (keeps corruption minimal).

    Effect on TF-IDF:
        With char_wb n-grams, the n-grams at word boundaries change.
        E.g. "apple inc" → n-grams like ["le i", "e in", " inc"]
        vs. "inc apple" → ["nc a", "c ap", " app"]. Different!
        The model learns that token ORDER is not rigidly important.

    Parameters
    ----------
    text : str
        The combined_text string.
    swap_prob : float
        Probability of performing a swap. Typical: 0.10–0.15.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Example
    -------
    >>> token_swap("buy 1000 apple inc shares", swap_prob=1.0)
    "buy 1000 inc apple shares"
    # "apple" and "inc" were swapped
    """
    if rng is None:
        rng = np.random.default_rng()

    tokens = text.split()
    if len(tokens) < 2:
        return text

    if rng.random() < swap_prob:
        i = rng.integers(0, len(tokens) - 1)
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

    return " ".join(tokens)


def _apply_char_noise(token: str, rng: np.random.Generator) -> str:
    """
    Apply ONE random character-level operation to a token.

    Operations (chosen uniformly):
      - insert:     add a random char at a random position
      - delete:     remove one char (if token length > 1)
      - substitute: replace one char with a random char

    Character pool: lowercase letters + digits (matches the normalised
    vocabulary of combined_text).

    This is an internal helper — called by char_noise() below.
    """
    if len(token) < 2:
        return token

    ops = ["insert", "delete", "substitute"]
    op = rng.choice(ops)
    chars = list(token)
    pos = int(rng.integers(0, len(chars)))
    char_pool = "abcdefghijklmnopqrstuvwxyz0123456789"

    if op == "insert":
        chars.insert(pos, rng.choice(list(char_pool)))
    elif op == "delete":
        if len(chars) > 1:
            chars.pop(pos)
    elif op == "substitute":
        chars[pos] = rng.choice(list(char_pool))

    return "".join(chars)


def char_noise(
    text: str,
    noise_prob: float = 0.05,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Apply character-level noise (insert/delete/substitute) to tokens.

    What it simulates:
        Typos, OCR errors, keyboard mistakes.  E.g. "APPLE" becoming
        "APLE" (deletion) or "APPXLE" (insertion) or "APPLE" → "ARPLE"
        (substitution).

    How it works:
        1. Split text into tokens.
        2. For each token, with probability `noise_prob`, apply ONE
           random character operation (insert, delete, or substitute).
        3. Only one operation per token — keeps corruption minimal.

    Effect on TF-IDF:
        Changes a few character n-grams per corrupted token.
        Example: "apple" generates n-grams [" ap", "app", "ppl", "ple", "le "].
        Deleting 'p' → "aple" gives [" ap", "apl", "ple", "le "] — MOST
        n-grams survive, but not all.  The model learns approximate
        string matching: "aple" ≈ "apple".

    Parameters
    ----------
    text : str
        The combined_text string.
    noise_prob : float
        Probability of applying char noise to each token. Typical: 0.03–0.08.
    rng : np.random.Generator, optional
        Random number generator.

    Example
    -------
    >>> char_noise("apple inc shares", noise_prob=1.0)
    "aple imc shares"
    # 'p' deleted from "apple", 'n'→'m' in "inc", "shares" survived
    """
    if rng is None:
        rng = np.random.default_rng()

    tokens = text.split()
    if not tokens:
        return text

    noisy_tokens = []
    for token in tokens:
        if rng.random() < noise_prob:
            token = _apply_char_noise(token, rng)
        noisy_tokens.append(token)

    return " ".join(noisy_tokens)


# Default synonym dictionary for financial trade matching.
# Extend this with domain-specific abbreviations as needed.
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "international": ["intl", "int"],
    "limited":       ["ltd"],
    "incorporated":  ["inc", "incorp"],
    "corporation":   ["corp"],
    "company":       ["co"],
    "securities":    ["secs", "sec"],
    "investment":    ["inv", "invest"],
    "management":    ["mgmt", "mgt"],
    "financial":     ["fin"],
    "partners":      ["ptnrs"],
    "associates":    ["assoc"],
    "exchange":      ["exch"],
    "capital":       ["cap"],
    "global":        ["glbl", "gbl"],
    "european":      ["eur"],
    "american":      ["amer", "am"],
    "technology":    ["tech"],
    "services":      ["svcs", "svc"],
    "trust":         ["tr"],
}


def synonym_substitution(
    text: str,
    synonym_dict: dict[str, list[str]] | None = None,
    sub_prob: float = 0.20,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Replace tokens with known synonyms/abbreviations.

    What it simulates:
        Different naming conventions across trading systems.
        E.g. "JP Morgan International Limited" in one system vs
        "JP Morgan Intl Ltd" in another.  Both refer to the same entity.

    How it works:
        1. Split text into tokens.
        2. For each token, check if it exists as a KEY in synonym_dict.
        3. If yes, with probability `sub_prob`, replace it with a
           randomly chosen synonym from the list.
        4. This is the ONLY perturbation that uses domain knowledge.

    Effect on TF-IDF:
        COMPLETELY changes the n-grams for the substituted token.
        "international" and "intl" share ZERO character n-grams.
        This forces the model to learn matching through other signals
        (same amount, same date, same ISIN) rather than pure text.

    Parameters
    ----------
    text : str
        The combined_text string.
    synonym_dict : dict[str, list[str]], optional
        Mapping from canonical token → list of synonyms.
        Defaults to DEFAULT_SYNONYMS (financial abbreviations).
    sub_prob : float
        Probability of substituting each eligible token. Typical: 0.15–0.30.
    rng : np.random.Generator, optional
        Random number generator.

    Example
    -------
    >>> synonym_substitution("apple international limited", sub_prob=1.0)
    "apple intl ltd"
    """
    if rng is None:
        rng = np.random.default_rng()
    if synonym_dict is None:
        synonym_dict = DEFAULT_SYNONYMS

    tokens = text.split()
    if not tokens:
        return text

    result = []
    for token in tokens:
        if token in synonym_dict and rng.random() < sub_prob:
            replacement = rng.choice(synonym_dict[token])
            result.append(replacement)
        else:
            result.append(token)

    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════
#  2. FIELD-LEVEL PERTURBATION — operates on the pd.Series
# ═══════════════════════════════════════════════════════════════════


def field_omission(
    row: pd.Series,
    columns_to_normalize: list[str],
    drop_prob: float = 0.10,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """
    Blank out an entire text column and rebuild combined_text.

    What it simulates:
        A field that exists in one system but is completely absent in
        another.  E.g. System A has ISIN + CUSIP + SEDOL, System B
        has only ISIN.  The model must match with fewer identifiers.

    How it works:
        1. With probability `drop_prob`, pick ONE column from
           `columns_to_normalize` that has a non-empty value.
        2. Set that column's value to "" (blank).
        3. Call `rebuild_combined_text_for_row()` to regenerate
           `combined_text` from the remaining columns.
        4. This uses YOUR EXISTING normalisation pipeline — no new
           text processing logic is introduced.

    Effect on TF-IDF:
        ALL n-grams from the dropped field disappear from the vector.
        If ISIN "US0378331005" was the only source of n-grams like
        "us0", "037", "378", those dimensions become zero.
        The model must rely on remaining text + scalar features.

    Integration note:
        This is the only perturbation that calls rebuild_combined_text_for_row()
        from data_prep.py, maintaining consistency with how combined_text
        is built during episode construction.

    Parameters
    ----------
    row : pd.Series
        The query row (will be copied, not modified in place).
    columns_to_normalize : list[str]
        The text columns used to build combined_text (same list used
        during data preparation).
    drop_prob : float
        Probability of blanking a field. Typical: 0.05–0.15.
    rng : np.random.Generator, optional
        Random number generator.

    Example
    -------
    Before: row["ISIN"] = "US0378331005"
            combined_text = "apple inc us0378331005 cusip037833100 ..."
    After:  row["ISIN"] = ""
            combined_text = "apple inc cusip037833100 ..."
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() >= drop_prob:
        return row  # No change — probability not met

    row = row.copy()

    # Find eligible columns (non-empty, present in row)
    eligible = [
        c for c in columns_to_normalize
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != ""
    ]
    if not eligible:
        return row

    # Blank one random column
    col_to_drop = rng.choice(eligible)
    row[col_to_drop] = ""

    # Rebuild combined_text from remaining columns
    row["combined_text"] = rebuild_combined_text_for_row(row, columns_to_normalize)

    return row


# ═══════════════════════════════════════════════════════════════════
#  3. SCALAR PERTURBATION — operates on amount and date columns
# ═══════════════════════════════════════════════════════════════════


def scalar_perturbation(
    row: pd.Series,
    amount_col: str,
    date_int_cols: list[str],
    *,
    amount_jitter_std: float = 0.005,
    date_shift_max: int = 2,
    date_shift_prob: float = 0.30,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """
    Add small noise to amount and/or shift dates.

    What it simulates:
        - Amount jitter: rounding differences, FX conversion slippage,
          fees applied differently across systems.
        - Date shift: T+1 vs T+2 settlement, timezone mismatches,
          booking date vs trade date differences.

    How it works:
        Amount:
          1. Multiply the amount by (1 + N(0, σ)) where σ = amount_jitter_std.
          2. Default σ = 0.005 → 0.5% noise.  An amount of 1,000,000 might
             become 1,000,470 or 999,530.
          3. This is a multiplicative noise (percentage-based), so it scales
             naturally with trade size.

        Dates:
          1. For each date_int column, with probability `date_shift_prob`,
             add a random integer in [-date_shift_max, +date_shift_max].
          2. Default: ±2 days, 30% chance per column.
          3. Each date column is shifted independently.

    Effect on model features:
        - scal_q changes: log1p(|amount|) and min_date/365 shift slightly.
        - pair_C changes AUTOMATICALLY: log_amt_diff and log_date_diff
          become non-zero for the positive candidate.  This teaches the
          model that "small diff ≠ non-match" — critical for robustness.
        - We do NOT touch pair features directly — they are DERIVED from
          the modified scalars, maintaining mathematical consistency.

    Parameters
    ----------
    row : pd.Series
        The query row (will be copied, not modified in place).
    amount_col : str
        Column name for the monetary amount.
    date_int_cols : list[str]
        The _int date column names (e.g. ["Trade Date_int", ...]).
    amount_jitter_std : float
        Relative standard deviation for amount noise. Typical: 0.002–0.01.
    date_shift_max : int
        Maximum ± days to shift. Typical: 1–3.
    date_shift_prob : float
        Probability of shifting each date column. Typical: 0.2–0.4.
    rng : np.random.Generator, optional
        Random number generator.

    Example
    -------
    Before: amount=1000.00, Trade Date_int=19750
    After:  amount=1000.47, Trade Date_int=19751
    """
    if rng is None:
        rng = np.random.default_rng()

    row = row.copy()

    # ── Amount jitter ──
    if amount_col in row.index and pd.notna(row[amount_col]):
        amount = float(row[amount_col])
        noise = rng.normal(0, amount_jitter_std)
        row[amount_col] = amount * (1 + noise)

    # ── Date shift ──
    for col in date_int_cols:
        if col in row.index and pd.notna(row[col]):
            if rng.random() < date_shift_prob:
                shift = int(rng.integers(-date_shift_max, date_shift_max + 1))
                row[col] = float(row[col]) + shift

    return row


# ═══════════════════════════════════════════════════════════════════
#  4. MASTER FUNCTION — composes all perturbations
# ═══════════════════════════════════════════════════════════════════


def augment_query(
    query_row: pd.Series,
    columns_to_normalize: list[str],
    amount_col: str,
    date_int_cols: list[str],
    *,
    # ── Switches: each perturbation is independently toggleable ──
    enable_token_dropout: bool = False,
    enable_token_swap: bool = False,
    enable_char_noise: bool = False,
    enable_synonym_sub: bool = False,
    enable_field_omission: bool = False,
    enable_scalar_noise: bool = False,
    # ── Parameters per perturbation ──
    token_drop_prob: float = 0.15,
    token_swap_prob: float = 0.10,
    char_noise_prob: float = 0.05,
    synonym_dict: dict[str, list[str]] | None = None,
    synonym_sub_prob: float = 0.20,
    field_drop_prob: float = 0.10,
    amount_jitter_std: float = 0.005,
    date_shift_max: int = 2,
    date_shift_prob: float = 0.30,
    # ── Control ──
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """
    Master augmentation function — applies selected perturbations to a query row.

    ┌──────────────────────────────────────────────────────────┐
    │              AUGMENTATION ORDER                          │
    │                                                          │
    │  1. Text perturbations (on combined_text string):        │
    │     a. token_dropout  → remove random words              │
    │     b. token_swap     → swap adjacent words              │
    │     c. char_noise     → typos in individual words        │
    │     d. synonym_sub    → abbreviation substitution        │
    │                                                          │
    │  2. Field omission (on pd.Series):                       │
    │     → blank entire column + rebuild combined_text        │
    │     NOTE: this OVERWRITES text perturbations from (1)    │
    │     if field_omission triggers, because it calls         │
    │     rebuild_combined_text_for_row(). This is by design:  │
    │     field omission represents a fundamentally different   │
    │     data scenario (missing field) vs text noise.          │
    │                                                          │
    │  3. Scalar perturbations (on amount/date values):        │
    │     → amount jitter + date shift                         │
    │     These are independent of text perturbations.         │
    └──────────────────────────────────────────────────────────┘

    Every parameter has a safe default. If ALL enable_* flags are False,
    this function returns the row unchanged (zero overhead).

    Parameters
    ----------
    query_row : pd.Series
        The episode query row.
    columns_to_normalize : list[str]
        Text columns used for combined_text (for field_omission).
    amount_col : str
        Name of the monetary amount column.
    date_int_cols : list[str]
        Names of the _int date columns.
    enable_* : bool
        Independent on/off switches for each perturbation type.
    *_prob, *_std, *_max : float/int
        Tuning knobs — see individual function docstrings.
    synonym_dict : dict, optional
        Custom synonym mapping. Defaults to DEFAULT_SYNONYMS.
    rng : np.random.Generator, optional
        Shared RNG for reproducibility.

    Returns
    -------
    pd.Series
        Modified copy of query_row (original is never mutated).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Quick exit if nothing is enabled
    if not any([
        enable_token_dropout, enable_token_swap, enable_char_noise,
        enable_synonym_sub, enable_field_omission, enable_scalar_noise,
    ]):
        return query_row

    row = query_row.copy()

    # ── 1. Text perturbations (applied to combined_text string) ─────
    text = str(row.get("combined_text", ""))
    if text:
        if enable_token_dropout:
            text = token_dropout(text, drop_prob=token_drop_prob, rng=rng)

        if enable_token_swap:
            text = token_swap(text, swap_prob=token_swap_prob, rng=rng)

        if enable_char_noise:
            text = char_noise(text, noise_prob=char_noise_prob, rng=rng)

        if enable_synonym_sub:
            text = synonym_substitution(
                text, synonym_dict=synonym_dict,
                sub_prob=synonym_sub_prob, rng=rng,
            )

        row["combined_text"] = text

    # ── 2. Field omission (may overwrite combined_text) ─────────────
    if enable_field_omission:
        row = field_omission(
            row, columns_to_normalize,
            drop_prob=field_drop_prob, rng=rng,
        )

    # ── 3. Scalar perturbations ─────────────────────────────────────
    if enable_scalar_noise:
        row = scalar_perturbation(
            row, amount_col, date_int_cols,
            amount_jitter_std=amount_jitter_std,
            date_shift_max=date_shift_max,
            date_shift_prob=date_shift_prob,
            rng=rng,
        )

    return row
