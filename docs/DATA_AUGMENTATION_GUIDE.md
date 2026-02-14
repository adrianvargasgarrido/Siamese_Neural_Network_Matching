# Data Augmentation Guide for Siamese Trade Matching

## Overview

This guide describes how to add realistic noise to training episodes to make the Siamese neural network more robust. The goal is to simulate real-world data difficulties the model will face at inference time, improving generalisation without collecting more labeled data.

---

## What the Model Sees

Each training episode consists ofs:

| Input | Shape | Source | Description |
|---|---|---|---|
| `t_a`, `t_b` | `(T,)` | TF-IDF vectors | Text representation from `combined_text` |
| `s_a`, `s_b` | `(2,)` | Scalar features | `[log1p(abs(amount)), min_date/365]` |
| `pair_feats` | `(3,)` | Cross features | `[log_amt_diff, log_date_diff, ref_exact]` |

Any augmentation must target **text**, **scalars**, or both. Pair features are derived and should not be perturbed directly.

---

## Realistic Noise Types

### 1. Text Perturbations

Real-world problems: typos, abbreviations, missing fields, reordering, different naming conventions.

| Technique | Simulates | Implementation |
|---|---|---|
| **Token dropout** | Missing fields / partial data | Randomly drop 1–3 tokens from `combined_text` |
| **Token swap** | Typos / reordering | Swap two adjacent tokens |
| **Char-level noise** | Typos / OCR errors | Insert, delete, or substitute 1–2 chars per token |
| **Synonym substitution** | "JP Morgan" ↔ "JPM", "International" ↔ "Intl" | Replace tokens from a lookup dictionary |
| **Field omission** | Missing system field | Zero out one column from `columns_to_normalize` |

**Critical rule:** Apply noise to the **query side (A) only**. Don't corrupt both sides — this teaches "the same trade can look different at entry" without making true pairs indistinguishable.

---

### 2. Scalar Perturbations

Real-world problems: rounding differences, settlement vs. trade date, partial fills, FX conversion.

| Technique | Simulates | Implementation |
|---|---|---|
| **Amount jitter** | Rounding, fees, FX errors | Multiply amount by `1 + N(0, σ)` where `σ ∈ [0.001, 0.02]` |
| **Date shift** | T+1 vs T+2, timezone differences | Add `±{1,2,3}` days to date_int columns |
| **Amount sign flip** | Buy vs. sell side (rare) | Negate amount with very low probability (< 1%) |

**Magnitude guidelines:**
- **Amount noise:** 0.1–1% is typical for exact matches; 1–5% for approximate matching
- **Date shifts:** ±1–3 days are realistic; ±30 days is not
- Keep perturbations small — the goal is robustness, not destruction

---

### 3. Pair Features

**Do NOT add noise to pair features directly.**

`log_amt_diff`, `log_date_diff`, and `ref_exact` are derived from the query and candidate. If you perturb scalars/text, pair features will change automatically. Adding independent noise here breaks the mathematical relationship and teaches incorrect signals.

---

## Where to Inject Noise

### Option A: Episode Construction Time
Augment in `_build_one_episode` before calling `get_candidates`.

**Pros:**
- Simple implementation
- No changes to vectorization

**Cons:**
- Fixed noise per episode (same corruption every epoch)
- Limited augmentation diversity

---

### Option B: Dataset Sampling Time ✅ **Recommended**
Augment in `RankingEpisodeDataset.__getitem__` during training.

**Pros:**
- Different noise every epoch = effectively infinite augmentation
- Maximum regularisation effect
- No need to rebuild episodes

**Cons:**
- Slightly more complex (but still straightforward)

**Why this is best:** Every epoch sees a different perturbation of the same episode, maximising diversity without needing more data or storage.

---

## Recommended Function Design

### Core Augmentation Function

```python
def augment_query(
    query_row: pd.Series,
    columns_to_normalize: list,
    amount_col: str,
    date_int_cols: list,
    *,
    # Text augmentation
    text_token_drop_prob: float = 0.15,       # Prob of dropping each token
    text_char_noise_prob: float = 0.05,       # Prob of char noise per token
    field_drop_prob: float = 0.1,             # Prob of blanking entire column
    
    # Scalar augmentation
    amount_jitter_std: float = 0.005,         # Relative std for amount noise (0.5%)
    date_shift_max: int = 2,                  # Max ±days shift
    date_shift_prob: float = 0.3,             # Prob of shifting each date column
    
    # Control
    rng: np.random.Generator = None,
) -> pd.Series:
    """
    Apply realistic noise to a query row.
    
    Text perturbations:
      - Token dropout: randomly drop tokens from combined_text
      - Char-level noise: insert/delete/substitute characters
      - Field omission: zero out entire text columns
    
    Scalar perturbations:
      - Amount jitter: multiply by (1 + N(0, σ))
      - Date shift: add ±days to date_int columns
    
    Returns a modified copy of the query row.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    row = query_row.copy()
    
    # ─── TEXT AUGMENTATION ───
    
    # 1. Field omission: randomly blank one text column
    if rng.random() < field_drop_prob:
        eligible = [c for c in columns_to_normalize if c in row.index and pd.notna(row[c])]
        if eligible:
            col_to_drop = rng.choice(eligible)
            row[col_to_drop] = ""
    
    # 2. Token-level noise
    text = str(row.get("combined_text", ""))
    if text:
        tokens = text.split()
        
        # Token dropout
        if tokens and rng.random() < text_token_drop_prob:
            n_drop = max(1, int(len(tokens) * 0.2))  # Drop up to 20% of tokens
            keep_mask = rng.random(len(tokens)) > (n_drop / len(tokens))
            tokens = [t for t, keep in zip(tokens, keep_mask) if keep]
        
        # Char-level noise
        if tokens:
            noisy_tokens = []
            for token in tokens:
                if rng.random() < text_char_noise_prob:
                    token = _apply_char_noise(token, rng)
                noisy_tokens.append(token)
            tokens = noisy_tokens
        
        row["combined_text"] = " ".join(tokens)
    
    # ─── SCALAR AUGMENTATION ───
    
    # 1. Amount jitter
    if amount_col in row.index and pd.notna(row[amount_col]):
        amount = float(row[amount_col])
        noise = rng.normal(0, amount_jitter_std)
        row[amount_col] = amount * (1 + noise)
    
    # 2. Date shift
    for date_col in date_int_cols:
        int_col = f"{date_col}_int"
        if int_col in row.index and pd.notna(row[int_col]):
            if rng.random() < date_shift_prob:
                shift = rng.integers(-date_shift_max, date_shift_max + 1)
                row[int_col] = float(row[int_col]) + shift
    
    return row


def _apply_char_noise(token: str, rng: np.random.Generator) -> str:
    """Apply one random char-level operation: insert, delete, or substitute."""
    if len(token) < 2:
        return token
    
    ops = ["insert", "delete", "substitute"]
    op = rng.choice(ops)
    chars = list(token)
    pos = rng.integers(0, len(chars))
    
    if op == "insert":
        # Insert a random adjacent char
        char_pool = "abcdefghijklmnopqrstuvwxyz0123456789"
        chars.insert(pos, rng.choice(list(char_pool)))
    elif op == "delete":
        if len(chars) > 1:
            chars.pop(pos)
    elif op == "substitute":
        char_pool = "abcdefghijklmnopqrstuvwxyz0123456789"
        chars[pos] = rng.choice(list(char_pool))
    
    return "".join(chars)
```

---

### Integration with Dataset

Update `RankingEpisodeDataset` to support augmentation:

```python
class RankingEpisodeDataset(Dataset):
    """Episode-based dataset for listwise training."""

    def __init__(
        self, 
        episodes, 
        vectorizer, 
        amount_col, 
        date_cols, 
        ref_col=None,
        augment: bool = False,            # NEW: enable augmentation
        augment_params: dict = None,      # NEW: custom augmentation params
        columns_to_normalize: list = None,
    ):
        self.episodes = episodes
        self.vectorizer = vectorizer
        self.amount_col = amount_col
        self.date_cols = date_cols
        self.ref_col = ref_col
        self.augment = augment
        self.augment_params = augment_params or {}
        self.columns_to_normalize = columns_to_normalize or []
        
        # Create RNG with fixed seed for reproducibility per worker
        self.rng = np.random.default_rng(seed=None)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        query_row = ep["query_row"]
        candidates_df = ep["candidates_df"]
        
        # Apply augmentation to query if enabled
        if self.augment:
            query_row = augment_query(
                query_row,
                columns_to_normalize=self.columns_to_normalize,
                amount_col=self.amount_col,
                date_int_cols=[f"{c}_int" for c in self.date_cols],
                rng=self.rng,
                **self.augment_params,
            )
        
        # Vectorize (with potentially augmented query)
        vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode(
            query_row, candidates_df,
            vectorizer=self.vectorizer,
            amount_col=self.amount_col,
            date_cols=self.date_cols,
            ref_col=self.ref_col,
        )
        
        # ... rest of __getitem__ unchanged
```

---

### Usage Example

```python
# Training dataset with augmentation
train_dataset = RankingEpisodeDataset(
    episodes=train_episodes,
    vectorizer=vectorizer,
    amount_col="Amount",
    date_cols=["Trade Date", "Settlement Date"],
    ref_col="Reference",
    augment=True,  # Enable augmentation for training
    augment_params={
        "text_token_drop_prob": 0.15,
        "text_char_noise_prob": 0.05,
        "amount_jitter_std": 0.005,  # 0.5% noise
        "date_shift_max": 2,
        "date_shift_prob": 0.3,
    },
    columns_to_normalize=["Counterparty", "Description", "Reference"],
)

# Validation dataset without augmentation
val_dataset = RankingEpisodeDataset(
    episodes=val_episodes,
    vectorizer=vectorizer,
    amount_col="Amount",
    date_cols=["Trade Date", "Settlement Date"],
    ref_col="Reference",
    augment=False,  # No augmentation for validation
    columns_to_normalize=["Counterparty", "Description", "Reference"],
)
```

---

## Anti-Patterns: What NOT to Do

| Don't Do This | Why It's Bad |
|---|---|
| Add Gaussian noise to TF-IDF vectors directly | Breaks sparsity structure, creates impossible token combinations |
| Corrupt both query AND positive candidate | Makes true pairs indistinguishable; model can't learn |
| Use large amount noise (>5%) | Teaches model wildly different amounts can match |
| Add noise to pair features directly | Breaks derived-feature contract, confuses the model |
| Apply same noise every epoch (Option A only) | Model memorises noise patterns instead of learning robustness |
| Use unrealistic magnitudes | e.g., ±30 day shifts, 20% amount changes |

---

## Tuning Guidelines

### Start Conservative

```python
# Conservative baseline
augment_params = {
    "text_token_drop_prob": 0.10,
    "text_char_noise_prob": 0.03,
    "amount_jitter_std": 0.002,  # 0.2%
    "date_shift_max": 1,
    "date_shift_prob": 0.2,
}
```

### Increase Gradually

Monitor validation MRR and Recall@K:
- If validation performance is **similar to training** → increase noise
- If validation is **much worse than training** → decrease noise
- If both plateau → you've found the sweet spot

### Domain-Specific Tuning

- **High-frequency trading:** Lower amount jitter (0.001 = 0.1%)
- **Cross-border trades:** Higher date shift (±3 days for timezone differences)
- **Manual data entry:** Higher char noise (0.08–0.10)
- **Automated feeds:** Lower text noise, focus on scalar perturbations

---

## Expected Benefits

1. **Improved robustness:** Model generalises better to unseen formatting variations
2. **Better calibration:** Confidence scores more reliable for borderline matches
3. **Reduced overfitting:** Acts as implicit regularisation
4. **Higher recall:** Model learns to match despite noise, catching more true positives
5. **Infinite data diversity:** Same episodes provide different training signals each epoch

---

## Implementation Checklist

- [x] Add `augment_query()` function to `pipeline/augmentation.py`
- [x] Add `_apply_char_noise()` helper function
- [x] Add `token_dropout()`, `token_swap()`, `char_noise()`, `synonym_substitution()` text perturbations
- [x] Add `field_omission()` field-level perturbation
- [x] Add `scalar_perturbation()` for amount/date noise
- [x] Update `RankingEpisodeDataset.__init__` with `augment`, `augment_params`, `columns_to_normalize`
- [x] Modify `RankingEpisodeDataset.__getitem__` to call `augment_query`
- [x] Add `columns_to_normalize` to Dataset constructor
- [x] Export all augmentation functions from `pipeline/__init__.py`
- [x] Add `AUGMENT_TRAIN` and `AUGMENT_PARAMS` config to notebook Step 2
- [x] Create training dataset with `augment=True` in training loop
- [x] Create validation/test datasets with `augment=False`
- [x] Add augmentation demo cell (Step 5b) with before/after visual
- [ ] Tune hyperparameters by monitoring val vs train performance
- [ ] Document final augmentation settings in training config

---

## References

- **Best practices:** Augment query only, keep noise realistic, vary per epoch
- **Architecture:** Option B (augment in `__getitem__`) for maximum diversity
- **Magnitude:** Start conservative (0.2% amount, ±1 day), increase if model plateaus
- **Validation:** Always disable augmentation for val/test sets

