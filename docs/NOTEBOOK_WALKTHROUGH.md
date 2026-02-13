# Siamese Trade Matching — Notebook Walkthrough

A step-by-step technical explanation of the `siamese_txn_matching.ipynb` notebook, designed for presenting in a technical review. Covers every step with theory, examples, why decisions were made, and what to look for in the outputs.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SIAMESE TRADE MATCHING PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Step 1  │────▶│    Step 2     │────▶│     Step 3       │
  │ Env &    │     │ Configuration│     │ Data Loading &   │
  │ Imports  │     │ (columns,    │     │ Cleaning         │
  │          │     │  thresholds) │     │ (filter, 1-to-1) │
  └──────────┘     └──────────────┘     └────────┬─────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │     Step 4       │
                                        │ Train/Val/Test   │
                                        │ Split (by group) │
                                        └────────┬─────────┘
                                                  │
          ┌───────────────────────────────────────┘
          ▼
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │     Step 5       │────▶│     Step 6       │────▶│     Step 7       │
  │ Episode          │     │ TF-IDF           │     │ Model Init       │
  │ Construction     │     │ Vectorisation    │     │ (Siamese net,    │
  │ (query, pos,     │     │ (char n-grams,   │     │  datasets,       │
  │  hard negatives) │     │  sparse vectors) │     │  data loaders)   │
  └──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │     Step 8       │
                                                    │ Training Loop    │
                                                    │ (listwise CE,    │
                                                    │  early stopping) │
                                                    └────────┬─────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │     Step 9       │
                                                    │ Evaluation &     │
                                                    │ Analysis         │
                                                    │ (metrics, plots) │
                                                    └──────────────────┘
```

### Data Flow Through the Model (Single Episode)

```
  Trade B (positive)                       Pool trades
       │                                    │ │ │
       │  clone + synthetic ID              │ │ │  blocking
       ▼                                    ▼ ▼ ▼  (currency, date, amount)
  ┌─────────┐                        ┌───────────────────┐
  │ Query A │                        │ Candidates        │
  │ (synth  │                        │ [B_pos, neg₁, neg₂│
  │  ID)    │                        │  neg₃, …, negₖ]  │
  └────┬────┘                        └───────┬───────────┘
       │                                     │
       │         TF-IDF vectorize            │
       ▼                                     ▼
  ┌─────────┐                        ┌───────────────────┐
  │ t_a     │                        │ t_b (for each)    │
  │ s_a     │                        │ s_b (for each)    │
  └────┬────┘                        └───────┬───────────┘
       │                                     │
       │      Shared Siamese Encoder         │
       ▼                                     ▼
  ┌─────────┐                        ┌───────────────────┐
  │ u (32d) │                        │ v₁, v₂, …, vₖ    │
  └────┬────┘                        └───────┬───────────┘
       │                                     │
       └──────────┬──────────────────────────┘
                  │
                  ▼  Comparison head (for each pair)
          ┌───────────────┐
          │  |u - vⱼ|     │
          │  u ⊙ vⱼ       │  ──▶  logit per candidate
          │  pair_feats   │
          └───────────────┘
                  │
                  ▼  Softmax over all K logits
          ┌───────────────┐
          │ Listwise CE   │  ──▶  Push positive to rank 1
          │ Loss          │
          └───────────────┘
```

---

## Step 1 — Environment & Imports

**What:** Detects runtime (Local vs Databricks), configures `sys.path`, imports all pipeline modules.

**Why:** The same notebook needs to run in VS Code locally, on a work laptop, and in Databricks. The path detection makes this portable.

**What to look for in output:**
- `Project Root:` — should point to the repo root where `src/` lives
- `Environment: Local` or `Databricks`

---

## Step 2 — Configuration

**What:** Defines column mappings (`ID_COL`, `AMOUNT_COL`, `DATE_COLS`) and blocking thresholds.

**Why:** Separating config from logic means you can re-run on a different dataset by changing only this cell.

**Key parameters explained:**

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `WINDOW_DAYS = 20` | Max ±20 days between query and candidate dates | Too tight → misses valid matches; too loose → too many candidates |
| `AMOUNT_TOL_PCT = 0.30` | Max 30% relative amount difference | Controls how "similar" amounts must be to be considered |
| `TRAIN_K_NEG = 10` | 10 negative candidates per episode | More negatives = harder task, better signal, slower training |
| `TOP_K = 50` | Max 50 candidates from blocking | Upper bound on retrieval before ranking |

---

## Step 3 — Data Loading & Cleaning

**What:** Loads raw data, filters to matched trades, keeps only valid 1-to-1 pairs.

**The filtering funnel (visual):**
```
Raw matched rows        ──▶  e.g., 5,000
After rule filter       ──▶  e.g., 4,200  (removed rules we don't trust)
After valid Match ID    ──▶  e.g., 4,100  (removed blank / #REF! IDs)
1-to-1 pairs only       ──▶  e.g., 3,600  (kept groups of exactly 2)
```

**Why 1-to-1 only?** The current model is designed as a ranking problem where each query has exactly one correct answer. Multi-match groups (3+ trades sharing a Match ID) would need a different label strategy.

**What to look for:**
- How much data survives each filter (funnel chart)
- Which match rules dominate (rule distribution chart)

---

## Step 4 — Train / Val / Test Split

**What:** Splits the data into three sets using **group-based** splitting by `Match ID`.

**Why group split (not random row split)?**

Imagine trades A and B are matched (Match ID = "M123"). If A goes into train and B goes into test, the model has effectively "seen" the answer during training → inflated metrics. Group split ensures both A and B go into the **same** split.

**Stratification:** Within the group split, we try to keep the same proportion of each match rule in every split. This prevents, e.g., all "FX Revaluation" matches ending up in train with none in test.

**What to look for:**
- Split sizes should be roughly 70/15/15
- Rule bars should be proportionally similar across splits (not all blue, no yellow)

---

## Step 5 — Episode Construction (Enhanced)

### What is an "episode"?

Think of an episode as a **multiple-choice exam question** for the model:

> *"Here is a trade (query). Which of these 21 candidates is its correct match?"*

The model must rank the correct answer (positive) above all wrong answers (negatives).

### Concrete example

Suppose we have this trade in the pool:

| Trade Id | ISIN | Amount | Currency | Date |
|----------|------|--------|----------|------|
| T-1001 | US912828XY | 1,000,000 | USD | 2025-01-15 |

**Step 1: Create the query.** Clone this trade but replace Trade Id with a synthetic one:

| Trade Id | ISIN | Amount | Currency | Date |
|----------|------|--------|----------|------|
| Q::T-1001::a8f3… | US912828XY | 1,000,000 | USD | 2025-01-15 |

The query has the same ISIN, amount, etc., but a **different Trade Id**. This prevents the model from learning "just match the Trade Id" — it must learn to match based on economic features.

**Step 2: Retrieve hard negatives via blocking.** The system finds other trades that look plausible but are NOT the correct match:

| Candidate | Trade Id | ISIN | Amount | Currency | Date | Label |
|-----------|----------|------|--------|----------|------|-------|
| 0 (positive) | T-1001 | US912828XY | 1,000,000 | USD | 2025-01-15 | ✅ |
| 1 (negative) | T-2045 | US912828ZZ | 995,000 | USD | 2025-01-16 | ❌ |
| 2 (negative) | T-3187 | GB00B24CGK7 | 1,010,000 | USD | 2025-01-14 | ❌ |
| … | … | … | … | … | … | ❌ |
| 20 (negative) | T-9821 | US912828AB | 980,000 | USD | 2025-01-17 | ❌ |

Notice the negatives are **hard** — same currency, similar amount, similar date. They're not random garbage; they're realistic confusing alternatives. Candidate 1, for example, has a similar ISIN prefix and amount — the model must learn that the exact ISIN suffix matters.

### Why hard negatives matter

| Negative type | Example | Model learns |
|---------------|---------|-------------|
| **Random** | Trade in JPY from 2023 | Nothing useful — trivially rejected |
| **Easy** | Same currency but wildly different amount | Amount matters (already known) |
| **Hard** | Same currency, similar amount, similar date, different ISIN | Fine-grained identifier distinctions |

Hard negatives are what make the model useful in production. Without them, the model would pass simple tests but fail on the real ambiguous cases.

### The blocking pipeline

Blocking is a fast filter that **narrows the pool** before the model does expensive scoring:

```
Full pool (e.g., 3,000 trades)
    │
    ├── Currency filter ──▶  Only same currency (e.g., 800 USD trades)
    │
    ├── Date window ──▶  Within ±20 days (e.g., 200 trades)
    │
    ├── Amount tolerance ──▶  Within ±30% (e.g., 80 trades)
    │
    └── Rank & Top-K ──▶  Best 20 by ref_exact, amount_diff, date_diff
```

### Episode dict structure

Each episode is a Python dict:

```python
{
    "query_row":      pd.Series,     # The synthetic query (side A)
    "candidates_df":  pd.DataFrame,  # Row 0 = positive, rows 1..K = negatives
    "positive_index": 0,             # Always 0 (positive is first)
    "rule":           "FX Revaluation",  # Which match rule created this pair
    "query_id":       "Q::T-1001::a8f3…",
    "candidate_ids":  ["T-1001", "T-2045", "T-3187", …],
}
```

### What to look for in the outputs

- **Candidates per episode histogram:** Most episodes should have `TRAIN_K_NEG + 1` candidates. If many have fewer, blocking is too strict (widen thresholds).
- **Episodes per rule:** Should roughly reflect the rule distribution in the data. If one rule dominates, the model may specialise.

---

## Step 6 — TF-IDF Vectorisation (Enhanced)

### What is TF-IDF? (Plain English)

TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It converts text into numbers by asking two questions:

1. **TF (Term Frequency):** "How often does this pattern appear in THIS text?"
2. **IDF (Inverse Document Frequency):** "How rare is this pattern across ALL texts?"

A pattern that appears often in one text but rarely in others gets a **high** score — it's distinctive. A pattern that appears everywhere (like common letters) gets a **low** score — it's noise.

### Why character n-grams? (With examples)

We use `analyzer="char_wb"` with `ngram_range=(2, 4)`, meaning we extract **character-level substrings of length 2, 3, and 4** from within word boundaries.

**Example:** The ISIN `"US912828XY"` generates these n-grams:

| Length | N-grams extracted |
|--------|-------------------|
| 2-gram | `us`, `s9`, `91`, `12`, `28`, `82`, `28`, `8x`, `xy` |
| 3-gram | `us9`, `s91`, `912`, `128`, `282`, `828`, `28x`, `8xy` |
| 4-gram | `us91`, `s912`, `9128`, `1282`, `2828`, `828x`, `28xy` |

**Why this matters for trade matching:**

Consider two ISINs: `US912828XY` and `US912828ZZ`. Word-level tokenisation treats them as two completely different words (similarity = 0). But character n-grams capture that they **share the prefix `US912828`** — most of their n-grams overlap. The model can learn that sharing 80% of character n-grams is strong evidence of a match.

**Concrete comparison:**

```
ISIN 1: "US912828XY"
ISIN 2: "US912828ZZ"

Shared 4-grams: us91, s912, 9128, 1282, 2828  (5 shared)
Different 4-grams: 828x, 28xy vs 828z, 28zz    (2 different each)

Cosine similarity ≈ 0.72  (high — they look similar)
```

```
ISIN 1: "US912828XY"
ISIN 3: "GB00B24CGK7"

Shared 4-grams: (almost none)

Cosine similarity ≈ 0.05  (low — very different)
```

This is exactly the signal the model needs: "these two ISINs are mostly the same" vs "these are completely different instruments."

### Why `char_wb` and not `word`?

| Approach | "JP Morgan Chase" ↔ "JPMorgan" | Verdict |
|----------|--------------------------------|---------|
| `word` | 0 shared tokens | ❌ Misses the match |
| `char_wb` (3-grams) | Shares `jpm`, `pmo`, `mor`, `org`, `rga`, `gan` | ✅ Captures partial overlap |

Financial data is full of abbreviations, concatenations, and formatting differences. Character n-grams are robust to all of these.

### What the vectorizer outputs

For each trade's `combined_text`, the vectorizer produces a **sparse vector** of size `VOCAB_SIZE` (typically 3,000–10,000 features). Most values are 0 (sparsity ~95%+), with only the n-grams present in that text having non-zero TF-IDF weights.

### What to look for in the outputs

- **Vocabulary size:** Larger = more expressive but slower. Typical range 3K–15K.
- **Sparsity:** Should be 90%+. If much lower, the text is very repetitive.
- **Similarity distribution plot:**
  - Positive pairs (green) should have **higher** similarity than negatives (red)
  - **Overlap region** = cases where TF-IDF alone cannot distinguish — this is why we need the neural network
  - **Separation gap** = difference between positive and negative means. Larger is better for the model's starting point.

---

## Step 7 — Model Initialisation (Enhanced)

### What are RankingEpisodeDataset and DataLoaders?

In plain English:

- **`RankingEpisodeDataset`** is a wrapper that takes our list of episode dicts and knows how to convert one episode into tensors the model can process. When PyTorch asks "give me episode #42", it:
  1. Takes the episode dict
  2. Runs `vectorize_episode()` to convert text → TF-IDF vectors, amounts → scalars, dates → scalars
  3. Computes pair features (amount diff, date diff, ref exact match)
  4. Returns tensors ready for the GPU

- **`DataLoader`** handles batching, shuffling, and parallelism. It collects 32 episodes at a time (BATCH_SIZE), shuffles them each epoch (training only), and uses `collate_episodes_flat` to merge them into one big tensor.

### How does `collate_episodes_flat` work?

Episodes have different numbers of candidates. The collation function concatenates all (query, candidate) pairs from all episodes in the batch into a flat tensor, then tracks which pairs belong to which episode using `lengths`:

```
Episode 1: 21 candidates → 21 pairs
Episode 2: 18 candidates → 18 pairs
Episode 3: 21 candidates → 21 pairs

Flat batch: 60 pairs total
lengths = [21, 18, 21]  → tells the loss function where to split
```

### What are the model inputs?

Each (query, candidate) pair produces 5 tensors:

| Tensor | Shape | Content | Example |
|--------|-------|---------|---------|
| `t_a` | `(T,)` | Query TF-IDF vector | Sparse vector from "q t 1001 us912828xy…" |
| `s_a` | `(2,)` | Query scalars | `[log1p(1000000) = 13.8, date_norm = 55.1]` |
| `t_b` | `(T,)` | Candidate TF-IDF vector | Sparse vector from "t 1001 us912828xy…" |
| `s_b` | `(2,)` | Candidate scalars | `[log1p(1000000) = 13.8, date_norm = 55.1]` |
| `pf` | `(3,)` | Pair features | `[log_amt_diff = 0.0, log_date_diff = 0.0, ref_exact = 0]` |

### The Siamese architecture in detail

```
Query A                              Candidate B
   │                                      │
   ▼                                      ▼
┌──────────────┐                   ┌──────────────┐
│ text_fc      │                   │ text_fc      │  ◄── SAME weights
│ (T → 32)    │                   │ (T → 32)    │      (shared encoder)
│ + ReLU       │                   │ + ReLU       │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│ scalar_fc    │                   │ scalar_fc    │  ◄── SAME weights
│ (2 → 8)     │                   │ (2 → 8)     │
│ + ReLU       │                   │ + ReLU       │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│ concat(32,8) │                   │ concat(32,8) │
│ encode_mix   │                   │ encode_mix   │  ◄── SAME weights
│ (40 → 32)   │                   │ (40 → 32)   │
│ + ReLU       │                   │ + ReLU       │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
  embedding u (32-d)              embedding v (32-d)
       │                                  │
       └──────────┬───────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Comparison     │
         │ |u - v| (32)   │  ← absolute difference
         │ u ⊙ v  (32)   │  ← element-wise product
         │ pair_f  (3)    │  ← amt_diff, date_diff, ref_exact
         │ ────────────── │
         │ Linear(67→16)  │
         │ ReLU           │
         │ Dropout(0.2)   │
         │ Linear(16→1)   │  ← single logit (score)
         └────────────────┘
```

**Why `|u - v|` AND `u ⊙ v`?**
- `|u - v|` captures **how different** the embeddings are in each dimension
- `u ⊙ v` captures **where they agree** (high values = both embeddings have signal in that dimension)
- Together they give the comparison head a richer view than either alone

**Why pair features (`pf`)?**
These are engineered cross-features that provide explicit numerical signals the model doesn't have to learn from scratch:
- `log_amt_diff`: How different are the amounts? (0.0 = identical)
- `log_date_diff`: How far apart are the dates? (0.0 = same day)
- `ref_exact`: Do they share the same reference? (1 or 0)

### What to look for in the outputs

- **Parameter count:** For a `VOCAB_SIZE` of ~5,000 and `embed_dim=32`, expect ~170K–200K parameters. This is very small — fast to train, low overfitting risk.
- **Batch shapes:** `t_as` should be `(N, T)` where N = total pairs in batch and T = vocab size.

---

## Step 8 — Training Loop (Enhanced)

### The loss function explained

**Listwise cross-entropy** treats each episode as a classification problem over K candidates:

$$\mathcal{L} = -\log \frac{e^{s_+}}{\sum_{j=1}^{K} e^{s_j}}$$

**Concrete example:** Suppose an episode has 4 candidates with these model scores (logits):

| Candidate | Score | Label |
|-----------|-------|-------|
| B_positive | 2.1 | ✅ (correct) |
| Negative 1 | 1.5 | ❌ |
| Negative 2 | 0.8 | ❌ |
| Negative 3 | -0.3 | ❌ |

Softmax probabilities:

$$P(\text{positive}) = \frac{e^{2.1}}{e^{2.1} + e^{1.5} + e^{0.8} + e^{-0.3}} = \frac{8.17}{8.17 + 4.48 + 2.23 + 0.74} = \frac{8.17}{15.62} = 0.523$$

Loss = $-\log(0.523) = 0.648$

A **perfect** model would give: $P(\text{positive}) = 1.0$, Loss = $0.0$

A model that gives equal scores to all 4 candidates: $P(\text{positive}) = 0.25$, Loss = $1.386$

### The metrics explained

**P@1 (Precision at 1):**

"Did the model rank the correct match at position 1?"

This is a binary yes/no per episode:

```
Episode 1: scores = [2.1, 1.5, 0.8] → rank of positive = 1 → HIT  ✅
Episode 2: scores = [1.2, 1.5, 0.8] → rank of positive = 2 → MISS ❌
Episode 3: scores = [3.0, 1.0, 0.5] → rank of positive = 1 → HIT  ✅

P@1 = 2 / 3 = 0.667
```

**MRR (Mean Reciprocal Rank):**

"On average, how close to position 1 is the correct match?"

```
Episode 1: correct at rank 1 → reciprocal = 1/1 = 1.000
Episode 2: correct at rank 2 → reciprocal = 1/2 = 0.500
Episode 3: correct at rank 1 → reciprocal = 1/1 = 1.000

MRR = (1.000 + 0.500 + 1.000) / 3 = 0.833
```

MRR is always ≥ P@1 because even when the model doesn't get rank 1, a rank-2 result still contributes (0.5) instead of 0. MRR tells you "the model is close even when it's not perfect."

### Why these metrics and not others?

| Alternative metric | Why NOT used here |
|-------------------|-------------------|
| **Accuracy** | Doesn't apply — this isn't binary classification, it's ranking |
| **AUC-ROC** | Measures discrimination at all thresholds, but we care about the **top** of the ranked list specifically |
| **F1-score** | Requires a binary threshold; ranking is about ordering, not thresholding |
| **NDCG** | Good for graded relevance (relevant/somewhat relevant/irrelevant). Our labels are binary (match/not match), so NDCG reduces to similar signal as MRR |
| **R@K (Recall at K)** | Useful when K > 1. With 1-to-1 matching, R@1 = P@1 |
| **MAP** | Useful for multiple relevant items per query. We have exactly 1, so MAP = MRR |

**P@1 + MRR are the right pair** because:
- P@1 = strict correctness (production threshold)
- MRR = ranking quality (near-miss awareness)

### What to look for during training

- **Epoch 1:** Loss should drop significantly, P@1 should jump from ~0.05 (random baseline for K=21) to something meaningful
- **Later epochs:** Loss should plateau or decrease slowly; P@1/MRR should stabilise
- **Early stopping:** If val loss increases for 3 consecutive epochs, training stops and restores the best model

---

## Step 9 — Evaluation & Analysis (Enhanced)

### Plot 1: 4-Panel Training Curves

**Top-left — Loss curves:**
- Blue line (train) and yellow line (val) should both decrease
- If train drops but val rises → **overfitting** (model memorises training data)
- Green dashed line = test loss (final benchmark)
- Healthy: both curves converge to similar values

**Top-right — Validation metrics per epoch:**
- P@1 and MRR should increase and stabilise
- Dashed lines show where test performance lands relative to validation
- If test metrics are close to val metrics → good generalisation

**Bottom-left — Overfitting gap:**
- Each bar = `train_loss - val_loss`
- Green bars (negative) = healthy (train loss ≥ val loss)
- Red bars (positive) = train loss < val loss → model fits training data more tightly than validation → potential overfit
- Small red bars are normal; large persistent red bars are concerning

**Bottom-right — Summary card:**
- Quick reference for all key numbers: epochs, losses, metrics, model size

### Plot 2: Score Distribution (Positive vs Negative)

This is one of the **most important** visuals. It shows the raw logit scores the model assigns to positive (correct) and negative (wrong) candidates on the test set.

**How to read it:**

```
                 Negatives              Positives
                 (red)                  (green)
    ┌────────────────┐    ┌────────────────────┐
    │  ████████████  │    │       ████████████ │
    │ █████████████  │    │      █████████████ │
    │████████████████│    │     ██████████████ │
    └────────────────┘    └────────────────────┘
   -2    -1     0    1    1    2     3    4    5
                    ◄─── overlap region ───►
```

- **Good model:** Green (positives) shifted far right of red (negatives), minimal overlap
- **Weak model:** Heavy overlap — model can't distinguish positive from negative
- **The overlap region** represents ambiguous cases where the model is uncertain. These would go to a "manual review" queue in production.

**Why logit scores and not probabilities?** Logits are the raw model output before softmax. They show the model's internal confidence scale without being compressed into 0–1.

### Plot 3: Rank Distribution

**"Where does the correct match end up in the model's ranking?"**

```
Rank 1: ████████████████████████████████  85%   ← Model got it right
Rank 2: ████████                           10%   ← Close, but a negative scored higher
Rank 3: ████                                4%   ← The positive was 3rd
Rank 4+: █                                  1%   ← Badly wrong
```

- **Green bars (Rank 1):** Correct → model succeeds → this IS P@1
- **Yellow bars (Rank 2–3):** Near misses → model found the right answer but ranked something similar higher
- **Red bars (Rank 4+):** Failures → blocking or model limitations

**Possible reasons for Rank > 1:**
1. **Very similar negatives:** Two trades with nearly identical ISINs, amounts, and dates — even humans might struggle
2. **Missing features:** The distinguishing information isn't in `combined_text` (e.g., counterparty name wasn't included)
3. **Rule-specific difficulty:** Some match rules rely on fields the model doesn't emphasise
4. **Insufficient training data:** Rare rules have fewer training episodes

### Plot 4: P@1 by Match Rule

**"Which types of matches does the model handle best/worst?"**

This chart breaks down P@1 for each match rule separately:

```
FX Revaluation                           ██████████████████████  95% (n=40)
Matched by Derived Trade Id              ████████████████████    88% (n=35)
Matched by Instrument ID & Economics     ███████████████         72% (n=25)
```

**How to interpret:**
- **High P@1 (>85%):** Model is confident and correct for this rule → safe for auto-matching
- **Medium P@1 (60–85%):** Model struggles sometimes → route to review queue
- **Low P@1 (<60%):** Model needs improvement for this rule → investigate features, data quality, or add specialised training data

**Possible reasons for low P@1 on a specific rule:**
1. **Few training episodes** for that rule → model hasn't learned the pattern well
2. **Rule depends on fields not in combined_text** → missing signal
3. **Rule has inherently ambiguous cases** → multiple candidates look equally valid
4. **Data quality issues** for that rule → noisy labels, mislabeled matches

**What to do about low-performing rules:**
- Increase training episodes for that rule (oversample)
- Add features that distinguish trades for that rule
- Investigate labeling quality
- Consider separate models or thresholds per rule

---

## Deep Dive Addendum — Why 32, Why Shallow, and What Bag-of-Words Size Means

This section answers common interview questions that usually come up after Step 6 and Step 7.

### 1) Why is the Siamese embedding size 32?

Current architecture uses:
- `text_fc: Linear(T → 32)`
- `scalar_fc: Linear(2 → 8)`
- `encode_mix: Linear(40 → 32)`

So each trade is encoded into a **32-dimensional embedding** (`u` for query, `v` for candidate).

**Why 32 is a sensible baseline in this project:**

1. **Capacity vs overfitting tradeoff:** 32 dimensions are usually enough to represent structured matching signals (identifier overlap + economic similarity) without over-parameterising.
2. **Fast training/inference:** Smaller embeddings keep matrix multiplications cheap, which matters when scoring many candidates per query.
3. **Engineered features already help:** Pair features (`log_amt_diff`, `log_date_diff`, `ref_exact`) carry strong domain signal, so the latent embedding does not need to be very large to perform well.
4. **Good for portability:** A compact model is easier to run in constrained environments (local laptop, Databricks shared clusters).

**Rule of thumb:** 32 is not a universal optimum; it is a robust default. If metrics plateau, run an ablation with `{16, 32, 64, 128}` and select the smallest dimension that preserves P@1/MRR.

### 2) Why only a shallow network ("few layers")?

The model intentionally uses a shallow stack:
- one text projection,
- one scalar projection,
- one fusion layer,
- then a small comparison MLP head.

**Why this is appropriate here:**

1. **Task structure is controlled:** This is not open-ended language understanding; it is a constrained trade-matching ranking task.
2. **Inputs are already informative:** TF-IDF char n-grams + explicit numeric pair features reduce the need for deep representation learning.
3. **Stability and debuggability:** Shallow models are easier to train, calibrate, and troubleshoot (especially around ranking metrics).
4. **Better data-efficiency:** With moderate dataset sizes, deeper models often overfit before giving meaningful gains.

In short: this architecture is a pragmatic engineering choice—strong enough to rank accurately, small enough to be reliable and maintainable.

### 3) What does "Bag of Words" mean in this notebook?

In classic NLP, Bag of Words (BoW) means:
- build a vocabulary of tokens,
- represent each text as a vector over that vocabulary.

In this project, it is a **character n-gram bag + TF-IDF weighting** (with `analyzer="char_wb"`, `ngram_range=(2,4)`).

So the vocabulary is not only whole words; it includes many char patterns like:
- `us9`, `912`, `28x`, `8xy`, `usd`, etc.

Each trade text becomes a sparse vector $x \in \mathbb{R}^{V}$ where $V$ is vocabulary size.

### 4) What does "vocab size = 64,000" mean?

It means the vectorizer learned **64,000 unique n-gram features** from training text.

It does **not** mean:
- each trade has 64,000 words,
- your dataset has 64,000 rows,
- every sample is dense.

It means each sample has a vector length of 64,000, but only a small subset of entries are non-zero.

### 5) Is 64,000 big?

Usually: **yes, fairly large** for this use case.

Whether it is *too large* depends on validation performance.

**Cost intuition for this exact model:**

First layer parameter count is approximately:

$$\text{params} \approx V \times 32 + 32$$

If $V = 64{,}000$:

$$\text{params} \approx 64{,}000 \times 32 + 32 = 2{,}048{,}032$$

So just `text_fc` has ~2.05M parameters. This is still manageable, but significantly heavier than smaller vocabularies.

### 6) Worked mini-example of BoW/TF-IDF representation

Assume tiny vocabulary (for illustration only):

| Index | Feature |
|------:|---------|
| 0 | `us9` |
| 1 | `912` |
| 2 | `usd` |
| 3 | `000` |
| 4 | `zz` |
| 5 | `xy` |
| 6 | `2025` |
| 7 | `gb0` |

Query: `"us912828xy usd 1000000 2025"`

$$q=[0.8,\ 1.2,\ 0.4,\ 0.6,\ 0.0,\ 1.1,\ 0.3,\ 0.0]$$

Positive-like candidate: `"us912828xy usd 1000000 2025"`

$$p=[0.8,\ 1.2,\ 0.4,\ 0.6,\ 0.0,\ 1.1,\ 0.3,\ 0.0]$$

Hard negative: `"us912828zz usd 995000 2025"`

$$n_h=[0.8,\ 1.2,\ 0.4,\ 0.5,\ 0.9,\ 0.0,\ 0.3,\ 0.0]$$

Easy negative: `"gb00... eur ..."`

$$n_e=[0.0,\ 0.0,\ 0.0,\ 0.1,\ 0.0,\ 0.0,\ 0.0,\ 1.3]$$

Interpretation:
- $q$ vs $p$: very similar (expected true match)
- $q$ vs $n_h$: still similar (hard case)
- $q$ vs $n_e$: very different (easy reject)

This is exactly why the model needs both TF-IDF features and a learned comparison head.

### 7) Practical guidance for interviews and tuning

If asked "is 64k good or bad?", use this answer:

> "64k means we have 64,000 character n-gram features in the TF-IDF vocabulary. It increases expressiveness but also model size and noise. We keep it only if it materially improves validation P@1/MRR versus smaller vocabularies like 10k–20k."

Recommended quick ablation:
- `max_features`: `10000`, `20000`, `40000`, `64000`
- Compare: Val/Test P@1, MRR, runtime
- Select smallest vocabulary within ~1% of best metric

---

## Extended Q&A — Technical Call Preparation

### Architecture questions

**Q: Why a Siamese network instead of a standard classifier on concatenated features?**

A: A standard classifier would concatenate all features from trade A and trade B into one big vector and classify as match/no-match. This has two problems:
1. It treats the inputs asymmetrically — swapping A and B would give a different result unless explicitly trained for symmetry
2. It doesn't learn reusable trade representations — each new trade pair must be scored from scratch

A Siamese network first creates a **representation** of each trade independently (via the shared encoder), then compares them. This means:
- The same encoder processes both sides → symmetric by construction
- Trade embeddings can be pre-computed and cached → fast inference at scale
- The model generalises better because it learns "what makes a trade" rather than "what makes this specific pair"

**Q: Why share encoder weights between query and candidate?**

A: Weight sharing ensures that both trades are mapped to the **same embedding space**. If each side had its own encoder, the model would need to learn two separate representations and how they relate — requiring more data and more parameters. Shared weights enforce that "a USD 1M bond trade" produces the same embedding regardless of whether it's the query or the candidate.

**Q: Why not just use cosine similarity on TF-IDF vectors directly?**

A: Pure TF-IDF cosine gives a single text-similarity score. It ignores amounts, dates, and pair interactions. The Siamese network can learn nonlinear combinations: "high text similarity + small amount difference → match" vs "high text similarity + large amount difference → not a match (different notional on same instrument)."

### Loss and metrics questions

**Q: Why listwise cross-entropy instead of pairwise (contrastive) loss?**

A: Pairwise loss compares the positive against each negative independently. Listwise CE considers **all candidates together** in one softmax — it directly optimises "rank the positive above everything else simultaneously." This is more aligned with the actual task (select from a list) and converges faster in practice.

**Q: What would P@1 be for a random model?**

A: With `TRAIN_K_NEG = 20`, each episode has 21 candidates (1 positive + 20 negatives). A random model ranks uniformly, so $P@1_{random} = \frac{1}{21} \approx 0.048$ (4.8%). Any P@1 significantly above 5% shows the model is learning. A production-ready model should target P@1 > 0.85.

**Q: If MRR = 0.92, what does that mean practically?**

A: On average, the correct match is at position $\frac{1}{0.92} \approx 1.09$. This means the model almost always ranks the correct match first or second. In production, this means a human reviewer would need to check at most 1–2 candidates per query.

### Data and preprocessing questions

**Q: Why synthetic Trade IDs in the query?**

A: Without the synthetic ID, the query's `combined_text` would be identical to the positive candidate's text (same Trade Id → same n-grams → cosine similarity = 1.0). The model would learn "pick the candidate with similarity 1.0" instead of learning meaningful matching patterns. The synthetic ID forces the text similarity to drop below 1.0, making the model learn from shared identifiers (ISIN, CUSIP, instrument name) rather than exact ID matching.

**Q: How do you prevent data leakage?**

A: Three safeguards:
1. **Group split by Match ID:** Both sides of a match always go in the same split
2. **Train-only vectorizer fit:** TF-IDF vocabulary comes only from training episodes; val/test text may contain unknown n-grams (which get zero weight — this is correct)
3. **Synthetic query IDs:** The model can't memorise Trade IDs as shortcuts

**Q: Why `char_wb` mode instead of `char`?**

A: `char_wb` adds word boundary markers. This means n-grams don't cross word boundaries — `"us912 bond"` produces `us9, s91, 912, bon, ond` but NOT `2 b` or `2 bo`. This reduces noise from meaningless cross-word patterns.

### Deployment questions

**Q: How would this work in production?**

A: Two phases:
1. **Blocking:** For each incoming trade, run the fast currency/date/amount filter to get ~50 candidates from the pool
2. **Scoring:** Run each (query, candidate) pair through the trained model to get a score
3. **Decision:** If the top score is above a confidence threshold → auto-match. If ambiguous → route to manual review queue.

**Q: How fast is inference?**

A: Very fast. The model is ~170K parameters (tiny). The expensive part is TF-IDF vectorisation, which is a sparse matrix multiplication. For a batch of 50 candidates, total inference takes <10ms on CPU.

**Q: How do you decide the confidence threshold?**

A: Use the score distribution plot from Step 9. Find the score where the positive and negative distributions separate cleanly. Trades above the threshold are auto-matched; trades in the overlap zone go to review.

**Q: What happens with a new instrument type the model hasn't seen?**

A: The TF-IDF n-grams for unknown identifiers will have zero weight (out-of-vocabulary). The model falls back on scalar features (amount, date) and learned patterns from similar instruments. Performance will degrade — this is expected and detected via the score distribution (lower confidence → flagged for review).

### Improvement questions

**Q: What would you do next to improve performance?**

A: In priority order:
1. **Error analysis:** Look at the rank 2+ failures — are they always the same rule? Same currency? Same instrument type?
2. **Feature enrichment:** Add more text columns (counterparty, product type) if available
3. **Data augmentation:** Add noise to queries (char typos, amount jitter, date shifts) to make the model more robust
4. **More training data:** More episodes, especially for underperforming rules
5. **Architecture tweaks:** Try larger `embed_dim`, deeper comparison head, or attention-based text encoder

**Q: Could you use transformers / BERT instead of TF-IDF?**

A: Yes, but it's a tradeoff. BERT captures semantic meaning but is 1000x larger and slower. For structured identifiers (ISINs, CUSIPs, numeric IDs), character n-grams are already very effective. BERT would help more if the text contained free-form descriptions. For this use case, TF-IDF is the pragmatic choice — fast, interpretable, and sufficient.

---

## 60-Second Summary for the Call

> "This notebook builds a **leakage-safe, end-to-end trade matching pipeline**. We start with raw trade data, filter to clean 1-to-1 matched pairs, and split by Match ID groups so no match leaks across train/val/test.
>
> We convert matching into a **ranking problem**: for each trade (query), the model must pick the correct match from a list of 21 candidates — 1 positive and 20 hard negatives retrieved via blocking (same currency, similar amount and date).
>
> Trades are represented using **character n-gram TF-IDF** (captures ISIN/CUSIP sub-patterns), **scalar features** (log amount, normalised date), and **pair features** (amount difference, date gap, reference exact match).
>
> The **Siamese network** encodes both query and candidate through a shared encoder into 32-d embeddings, then a comparison head scores each pair. Training uses **listwise cross-entropy** — softmax over all candidates, pushing the positive to rank 1. Early stopping restores the best checkpoint.
>
> Final evaluation shows **score distributions** (positive vs negative separation), **rank distributions** (how often the model gets rank 1), and **per-rule P@1** (which match rules work well and which need attention). The key takeaway metrics are P@1 (strict top-1 accuracy) and MRR (average ranking quality)."
