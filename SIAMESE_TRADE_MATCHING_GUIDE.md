# Siamese Neural Network for Trade Matching

## A Comprehensive Technical Guide

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [What Is a Siamese Network?](#2-what-is-a-siamese-network)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Step 1: Data Cleaning & Filtering](#step-1-data-cleaning--filtering)
5. [Step 2: Text Normalization & Feature Engineering](#step-2-text-normalization--feature-engineering)
6. [Step 3: Stratified Group Splitting](#step-3-stratified-group-splitting)
7. [Step 4: Episode Construction (Learning-to-Rank)](#step-4-episode-construction-learning-to-rank)
8. [Step 5: TF-IDF Vectorization](#step-5-tf-idf-vectorization)
9. [Step 6: Episode Vectorization & Feature Assembly](#step-6-episode-vectorization--feature-assembly)
10. [Step 7: PyTorch Dataset & Collation](#step-7-pytorch-dataset--collation)
11. [Step 8: The Siamese Network Architecture](#step-8-the-siamese-network-architecture)
12. [Step 9: Listwise Cross-Entropy Loss](#step-9-listwise-cross-entropy-loss)
13. [Step 10: Training Loop with Early Stopping](#step-10-training-loop-with-early-stopping)
14. [Step 11: Evaluation Metrics](#step-11-evaluation-metrics)
15. [End-to-End Data Flow Diagram](#end-to-end-data-flow-diagram)
16. [Glossary](#glossary)

---

## 1. Introduction & Motivation

### The Problem: Trade Matching

In financial operations, **trade matching** (also called **trade reconciliation**) is the process of pairing two sides of a transaction — for example, a buy record from System A and a sell record from System B — to confirm they represent the same economic event.

Traditional rule-based matching systems use deterministic logic:

```
IF  Trade_A.Currency == Trade_B.Currency
AND Trade_A.Amount   == -Trade_B.Amount
AND Trade_A.Date     == Trade_B.Date
THEN → MATCH
```

**Limitations of rule-based matching:**
- Brittle to data quality issues (typos, missing fields, format differences)
- Rules must be hand-crafted for each scenario
- Hard to rank among multiple potential matches
- Cannot learn from historical matching decisions

### The Solution: Learning to Match

This notebook implements a **Siamese Neural Network** that learns to score how likely two trades are to be a match, using:

- **Text features** (Trade IDs, instrument identifiers, ISINs, CUSIPs)
- **Numeric features** (amounts, dates)
- **Pairwise features** (amount differences, date differences, reference matches)

The model learns from historical matched pairs to generalize to new, unseen trades.

---

## 2. What Is a Siamese Network?

### Core Concept

A Siamese Network uses **two identical sub-networks** (sharing the same weights) to process two inputs independently, then compares their representations to determine similarity.

```
     Trade A                    Trade B
        │                          │
   ┌────┴────┐                ┌────┴────┐
   │ Encoder │                │ Encoder │    ← Same weights (shared)
   │  f(·)   │                │  f(·)   │
   └────┬────┘                └────┬────┘
        │                          │
     u = f(A)                  v = f(B)
        │                          │
        └──────────┬───────────────┘
                   │
            ┌──────┴──────┐
            │  Comparison │    ← |u - v|, u * v, pair features
            │    Head     │
            └──────┬──────┘
                   │
              Score (logit)
```

### Why "Siamese"?

Named after **Siamese twins** — the two sub-networks are identical (same architecture, same weights). This forces the network to learn a **universal representation** of trades rather than memorizing specific pairings.

### Key Properties

| Property | Benefit |
|----------|---------|
| **Weight sharing** | Learns a single, consistent embedding space |
| **Symmetry** | Can compare any trade to any other trade |
| **Generalization** | New trades are embedded in the same space |
| **Pairwise learning** | Directly optimizes for "is this a match?" |

### How This Differs From Classification

A traditional classifier would ask: *"What category does this trade belong to?"*

A Siamese network asks: *"How similar are these two trades?"*

This is critical because:
- New Match IDs appear constantly (you can't predefine classes)
- The model needs to generalize to unseen trade pairs
- The output is a **ranking** (most likely match), not a category

---

## 3. Pipeline Overview

The notebook follows a systematic pipeline, each step feeding the next:

```
┌──────────────────────────────────────────────────────────────┐
│                    RAW TRADE DATA                            │
│  (Trade IDs, amounts, dates, currencies, instruments, etc.) │
└──────────────────────┬───────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 1: Clean &    │  Filter to 1-to-1 matched pairs
            │  Filter             │  Remove invalid Match IDs
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 2: Normalize  │  Lowercase, strip, combine text
            │  Text Features      │  → "combined_text" column
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 3: Split      │  Train (70%) / Val (15%) / Test (15%)
            │  by Match ID Groups │  No group leakage across splits
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 4: Build      │  Query → [Positive + K Negatives]
            │  Episodes           │  Realistic candidate retrieval
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 5: Fit        │  char_wb n-grams (2,4)
            │  TF-IDF Vectorizer  │  Train-only (no leakage)
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 6: Vectorize  │  Text → TF-IDF vectors
            │  Episodes           │  Scalars, pair features
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 7: PyTorch    │  Dataset, DataLoader, Collation
            │  Data Loading       │  Batch episodes for GPU
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 8: Forward    │  Siamese encoder + comparison head
            │  Pass               │  → logit per candidate
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 9: Listwise   │  Cross-entropy over candidate sets
            │  Loss               │  "Which candidate is the match?"
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 10: Train     │  Adam optimizer
            │  with Early Stop    │  Restore best model
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Step 11: Evaluate  │  P@1, R@1, MRR on test set
            └─────────────────────┘
```

---

## Step 1: Data Cleaning & Filtering

### What Happens

The raw data contains many types of matches (FX Revaluation, Offsetting Journals, etc.) and various Match ID qualities. This step filters down to a clean, reliable training set.

### The Filtering Pipeline

```
Raw Matched Data (all rows where matched == True)
        │
        ▼
(1) Filter by Comments (match type)
    Keep only: {"FX Revaluation", "Matched by Derived Trade Id",
                "Matched by Instrument Identifier and Trade Economics"}
        │
        ▼
(2) Filter by Match ID quality
    Remove: NaN, blank strings, "#REF!" values
        │
        ▼
(3) Keep only 1-to-1 matches
    Select Match ID groups of exactly size 2
    (one trade on each side of the match)
```

### Why 1-to-1 Only?

The Siamese network is designed for **pairwise comparison**. Groups of 3+ trades per Match ID introduce ambiguity about which pairs are the "real" matches. By restricting to groups of exactly 2, we get clean labeled pairs:

```
Match ID "M001":
  Trade A (row 1) ↔ Trade B (row 2)  ← Confirmed positive pair

Match ID "M002":
  Trade C (row 3) ↔ Trade D (row 4)  ← Confirmed positive pair
```

### Why Filter by Comments?

Different match types have different economics:

| Comments | Example Logic |
|----------|---------------|
| FX Revaluation | Same instrument, offsetting amounts due to FX rate change |
| Matched by Derived Trade Id | Trade IDs are algorithmically related |
| Matched by Instrument Identifier | Same ISIN/CUSIP/SEDOL, compatible economics |

By focusing on specific Comments types, the model learns patterns that generalize within those matching contexts. Mixing unrelated match types would confuse the learning signal.

---

## Step 2: Text Normalization & Feature Engineering

### What Happens

Text fields from trades are normalized and combined into a single `combined_text` column that serves as the text input to the model.

### The `normalize()` Function

```python
normalize("  T000000046  ")  → "t000000046"
normalize("US9128283D82")    → "us9128283d82"
normalize("None")            → ""     # null-like strings become empty
normalize("ABC-DEF/123")     → "abc def 123"  # non-word chars → spaces
```

**Steps:**
1. Convert to string, strip whitespace, lowercase
2. Replace null-like values (`"none"`, `"nan"`, `"null"`, `"n/a"`) with empty string
3. Replace non-word characters (`\W+`) with spaces
4. Collapse multiple spaces

### The `normalize_and_combine()` Function

Concatenates normalized text from multiple identifier columns into one string:

```python
columns_to_normalize_reduced = [
    "Trade Id", "Alternate Trade Id", "Alternate Trade Id 2", "Deal ID",
    "Unique Instrument Identifier", "TETB FISS Number",
    "Instrument Name", "ISIN", "CUSIP", "SEDOL",
]
```

**Example:**

| Column | Raw Value | Normalized |
|--------|-----------|------------|
| Trade Id | T000000046 | t000000046 |
| ISIN | US9128283D82 | us9128283d82 |
| Deal ID | None | *(skipped)* |
| CUSIP | 912828D82 | 912828d82 |

**Result `combined_text`:** `"t000000046 us9128283d82 912828d82"`

### Why Combine Into One Field?

1. **Simplifies vectorization** — one TF-IDF transform instead of many
2. **Cross-field matching** — if Trade A's ISIN matches Trade B's CUSIP, character n-grams will overlap
3. **Handles missing data** — empty fields are simply omitted from the concatenation
4. **Consistent representation** — the model sees one unified text string per trade

---

## Step 3: Stratified Group Splitting

### What Happens

The data is split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets with critical constraints.

### Why Group Splitting Matters

**CRITICAL RULE:** Both sides of a matched pair must go into the **same** split.

```
❌ WRONG (Data Leakage):
  Train: Trade A (Match ID M001)
  Test:  Trade B (Match ID M001)    ← Model has seen A's "twin"!

✅ CORRECT (Group Split):
  Train: Trade A + Trade B (Match ID M001)  ← Both sides together
  Test:  Trade C + Trade D (Match ID M002)  ← Entirely unseen pair
```

If matched pairs leak across splits, the model can "cheat" by memorizing partner features from training, producing artificially inflated metrics.

### How It Works

```python
stratified_group_split_3way(
    df,
    group_col="Match ID",    # Keep entire groups together
    strat_col="Comments",    # Balance match types across splits
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)
```

**Two-phase approach:**

1. **Phase 1:** Split groups into Train (70%) vs. Holdout (30%), stratified by Comments
2. **Phase 2:** Split Holdout into Validation vs. Test, stratified by Comments

**Stratification by Comments** ensures each split has a proportional mix of match types (FX Revaluation, Derived Trade Id, etc.), preventing a split from being dominated by one type.

### Fallback for Small Data

When sample sizes are too small for stratification (singleton classes), the code falls back to a simple random group split:

```python
# Groups are shuffled randomly and divided by proportion
groups = df_clean["Match ID"].unique()
rng.shuffle(groups)
# 70% train, 15% val, 15% test
```

---

## Step 4: Episode Construction (Learning-to-Rank)

### What Happens

This is the heart of the training data preparation. Instead of simple positive/negative pairs, we construct **episodes** — each consisting of one query trade, one correct match (positive), and K negative candidates.

### What Is an Episode?

An episode represents a realistic matching scenario:

```
Episode #1:
┌─────────────────────────────────────────────────────────────┐
│ Query (A): Trade T000000046 (synthetic copy)                │
│                                                             │
│ Candidates:                                                 │
│   [0] T000000046  ← POSITIVE (true match)   label = 1      │
│   [1] T000000033  ← negative                label = 0      │
│   [2] T000000025  ← negative                label = 0      │
│   [3] T000000050  ← negative                label = 0      │
│   [4] T000000003  ← negative                label = 0      │
│   ...             ← up to K negatives                       │
└─────────────────────────────────────────────────────────────┘
```

### Why Episodes Instead of Simple Pairs?

**Simple pairs** (Train on `[A, B, label=1]` and `[A, C, label=0]` independently) have limitations:

1. **No ranking context** — the model doesn't learn to differentiate among multiple candidates
2. **Easy negatives** — random negatives are often trivially different from the positive
3. **Calibration issues** — scores aren't comparable across different queries

**Episodes** (listwise approach) solve all three:

1. The model must rank the positive above **all** negatives in the same set
2. Negatives are retrieved using **realistic blocking** (same currency, date window, amount tolerance)
3. Scores are directly comparable within each episode

### How Negatives Are Selected

The episode builder uses a **two-stage negative sampling** strategy:

#### Stage 1: Realistic Retrieval (Hard Negatives)

```python
neg_cands = get_candidates(
    query_row=query,
    pool_df=df_pool,
    top_k=train_k_neg * 2,     # Retrieve 2x needed
    window_days=WINDOW_DAYS,    # Date proximity filter
    amount_tol_pct=AMOUNT_TOL_PCT,  # Amount tolerance filter
    enforce_same_sign=True      # Same debit/credit side
)
```

This uses the same **blocking logic** the model will face at inference:

1. **Currency block** — only same-currency candidates
2. **Date window** — within N days of the query
3. **Amount tolerance** — within X% of the query amount
4. **Ranking** — sorted by reference match, amount diff, date diff

**These are "hard negatives"** — they look plausible but aren't the correct match.

#### Stage 2: Random Top-Up

If blocking returns fewer than K negatives, random trades from the pool fill the gap. These are "easy negatives" that provide baseline learning signal.

### The Candidate Retrieval Pipeline (`get_candidates`)

```
Query Trade (with currency, dates, amount)
        │
        ▼
(1) Currency Block
    Pool → only trades with same Transactional Currency
        │
        ▼
(2) Exclude Self
    Remove the query trade itself from candidates
        │
        ▼
(3) Date Window
    Keep candidates where min date diff ≤ WINDOW_DAYS
    Policy: "any" = at least one date is close
            "all" = all dates must be close
        │
        ▼
(4) Amount Tolerance
    Keep candidates where |amount_diff| / |query_amount| ≤ AMOUNT_TOL_PCT
    Optional: enforce same sign (Dr/Cr matching)
        │
        ▼
(5) Rank by: ref_exact ↓, amount_diff ↑, date_diff ↑
    Keep Top-K
```

### Why This Retrieval Matters

The model learns from the **distribution of candidates it will actually see** at inference time. If training negatives are random, the model learns to distinguish obvious non-matches but fails on close calls. Hard negatives force the model to learn subtle discriminative features.

---

## Step 5: TF-IDF Vectorization

### What Happens

The text features (`combined_text`) are converted into fixed-length numeric vectors using TF-IDF with character n-grams.

### Configuration

```python
vectorizer = TfidfVectorizer(
    analyzer="char_wb",     # Character-level n-grams within word boundaries
    ngram_range=(2, 4),     # 2-grams, 3-grams, and 4-grams
    dtype=np.float32,
)
vectorizer.fit(iter_episode_text(episodes_train))  # Train-only!
```

### Why Character N-Grams?

Trade identifiers are **structured codes** (e.g., `US9128283D82`), not natural language. Character n-grams capture:

| Feature | Word-level TF-IDF | Char n-gram TF-IDF |
|---------|--------------------|--------------------|
| `"T000000046"` | 1 token | `"t0"`, `"00"`, `"000"`, `"0000"`, `"004"`, `"046"`, ... |
| Partial matches | ❌ All or nothing | ✅ Partial overlap in n-grams |
| Typo resilience | ❌ `"T00000046"` ≠ `"T000000046"` | ✅ Most n-grams still match |
| Cross-field | ❌ Different vocabularies | ✅ Shared substrings detected |

**`char_wb` (character within word boundaries)** avoids n-grams that span word boundaries, keeping each identifier's n-grams independent.

**`ngram_range=(2, 4)`** captures:
- **2-grams:** `"t0"`, `"00"`, `"04"`, `"46"` — fine-grained character pairs
- **3-grams:** `"t00"`, `"000"`, `"046"` — trigrams for more context
- **4-grams:** `"t000"`, `"0000"`, `"0046"` — longer patterns for specificity

### Why Train-Only Fitting?

```python
# ✅ Correct: fit only on training episodes
vectorizer.fit(iter_episode_text(episodes_train))

# ❌ Wrong: fitting on all data would leak test vocabulary
# vectorizer.fit(iter_episode_text(all_episodes))
```

Fitting the vectorizer on validation/test data would allow the vocabulary to include n-grams only present in test trades, creating **data leakage**. The vocabulary is frozen after fitting on training data; validation and test texts are transformed using this fixed vocabulary.

### The Corpus Iterator

```python
def iter_episode_text(episodes):
    """Yield all combined_text strings from queries and candidates."""
    for ep in episodes:
        yield ep["query_row"].get("combined_text", "")  # Query text
        for txt in ep["candidates_df"]["combined_text"]:  # Candidate texts
            yield txt
```

This yields every text string (queries and candidates) from all training episodes, building a vocabulary that covers the full range of identifier patterns seen during training.

---

## Step 6: Episode Vectorization & Feature Assembly

### What Happens

Each episode is converted from raw data into numeric tensors suitable for the neural network. Three types of features are created:

### 6a. Text Vectors (TF-IDF)

```python
txt_q = query_row.get("combined_text", "")
vec_q = vectorizer.transform([txt_q]).toarray()  # Shape: (T,) where T = vocab size
vec_C = vectorizer.transform(candidate_texts)     # Shape: (K, T)
```

Each trade becomes a sparse-to-dense vector of TF-IDF weights over the vocabulary.

### 6b. Scalar Features

Two numeric features are extracted per trade:

```python
scal = [log1p(|amount|), min_date_norm]
```

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log1p(|amount|)` | $\log(1 + |amount|)$ | Normalized trade size (log-scale handles wide range: $100 to $1B) |
| `min_date_norm` | $\frac{\min(\text{date\_ints})}{365}$ | Earliest date normalized to years (captures temporal proximity) |

**Why log1p?** Trade amounts span many orders of magnitude. Raw amounts would dominate the loss; log-scaling compresses the range while preserving relative ordering.

### 6c. Pairwise Features

Three features computed **between** the query and each candidate:

```python
pair_C = [log_amt_diff, log_min_date_diff, ref_exact]
```

| Feature | Formula | What It Captures |
|---------|---------|------------------|
| `log_amt_diff` | $\log(1 + |A_{amt} - B_{amt}|)$ | How different the amounts are |
| `log_min_date_diff` | $\log(1 + \min_d |A_d - B_d|)$ | How far apart the dates are |
| `ref_exact` | $\mathbb{1}[A_{ref} = B_{ref}]$ | Whether reference fields match exactly |

**These are the "relationship" features** — they directly measure compatibility between query and candidate, giving the comparison head explicit similarity signals beyond what the encoder learns implicitly.

### Complete Feature Summary Per Candidate Pair

```
For each (Query A, Candidate B) pair:

Text features (from TF-IDF):
  t_a: (T,)  — Query text vector
  t_b: (T,)  — Candidate text vector

Scalar features:
  s_a: (2,)  — [log1p(|amt_A|), min_date_norm_A]
  s_b: (2,)  — [log1p(|amt_B|), min_date_norm_B]

Pairwise features:
  pf:  (3,)  — [log_amt_diff, log_min_date_diff, ref_exact]
```

---

## Step 7: PyTorch Dataset & Collation

### What Happens

Episodes are wrapped in a PyTorch `Dataset` for efficient batching, and a custom `collate_fn` flattens multiple episodes into a single forward pass.

### The `RankingEpisodeDataset`

```python
class RankingEpisodeDataset(Dataset):
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        # Vectorize the episode using the train-fitted vectorizer
        vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode(
            ep["query_row"], ep["candidates_df"], vectorizer=self.vectorizer
        )
        
        K = vec_C.shape[0]  # Number of candidates
        
        # Repeat query K times (one per candidate pair)
        t_as = repeat(vec_q, K times)   # (K, T)
        s_as = repeat(scal_q, K times)  # (K, 2)
        t_bs = vec_C                     # (K, T)
        s_bs = scal_C                    # (K, 2)
        pf   = pair_C                    # (K, 3)
        
        return {t_as, s_as, t_bs, s_bs, pf, length=K, pos_ix}
```

**Key insight:** The query is repeated K times so each `(t_a[i], t_b[i])` pair represents the query compared against candidate $i$.

### The Flat Collation Strategy

Instead of padding to the maximum number of candidates (wasteful), the collator **concatenates** all pairs across episodes:

```
Episode 1: K₁ = 18 candidates  →  18 pair rows
Episode 2: K₂ = 15 candidates  →  15 pair rows
Episode 3: K₃ = 20 candidates  →  20 pair rows
────────────────────────────────────────────
Batch:     N  = 53 total rows   →  One forward pass!

lengths = [18, 15, 20]  ← Used to split logits back into groups
pos_ixs = [0, 0, 0]     ← Position of the positive in each group
```

**Why this approach?**

1. **No padding waste** — every row is a real pair, not a filler
2. **Single forward pass** — all 53 pairs processed together (GPU-efficient)
3. **Variable-size episodes** — different episodes can have different numbers of candidates
4. **`lengths` tensor** — tells the loss function where each episode's candidates start and end

---

## Step 8: The Siamese Network Architecture

### Architecture Diagram

```
                    Trade A                           Trade B
                      │                                 │
          ┌───────────┴───────────┐         ┌───────────┴───────────┐
          │      text_vec (T,)    │         │      text_vec (T,)    │
          │      scalar_vec (2,)  │         │      scalar_vec (2,)  │
          └───────────┬───────────┘         └───────────┬───────────┘
                      │                                 │
   SHARED ENCODER:    │                                 │
   ┌──────────────────┴─────────┐    ┌──────────────────┴─────────┐
   │ text_fc: Linear(T → 32)   │    │ text_fc: Linear(T → 32)   │
   │ → ReLU                     │    │ → ReLU                     │
   │                             │    │                             │
   │ scalar_fc: Linear(2 → 8)  │    │ scalar_fc: Linear(2 → 8)  │
   │ → ReLU                     │    │ → ReLU                     │
   │                             │    │                             │
   │ concat: (32 + 8 = 40)      │    │ concat: (32 + 8 = 40)      │
   │ encode_mix: Linear(40→32)  │    │ encode_mix: Linear(40→32)  │
   │ → ReLU                     │    │ → ReLU                     │
   └──────────────┬──────────────┘    └──────────────┬──────────────┘
                  │                                   │
              u = f(A)  ∈ ℝ³²                    v = f(B)  ∈ ℝ³²
                  │                                   │
                  └────────────┬──────────────────────┘
                               │
                    COMPARISON HEAD:
                    ┌──────────┴──────────┐
                    │ |u - v|  (32,)      │  ← Absolute difference
                    │ u * v    (32,)      │  ← Element-wise product
                    │ pair_feats (3,)     │  ← Pairwise features
                    │                     │
                    │ concat: (32+32+3=67)│
                    │ Linear(67 → 16)     │
                    │ → ReLU              │
                    │ → Dropout(0.2)      │
                    │ Linear(16 → 1)      │
                    └──────────┬──────────┘
                               │
                          logit (scalar)
                     higher = more likely match
```

### The Encoder (`forward_one`)

```python
def forward_one(self, text_vec, scalar_vec):
    t = relu(text_fc(text_vec))       # (T,) → (32,)
    s = relu(scalar_fc(scalar_vec))   # (2,) → (8,)
    combined = concat([t, s])         # (40,)
    embedding = relu(encode_mix(combined))  # (40,) → (32,)
    return embedding  # u ∈ ℝ³²
```

**Design Rationale:**
- **Separate text and scalar projections** — text features are high-dimensional (297 TF-IDF dims), scalars are just 2D. Projecting separately prevents the scalars from being drowned out.
- **Fusion layer (`encode_mix`)** — combines text and scalar representations into a unified 32-D embedding.
- **ReLU activations** — standard non-linearity; introduces the ability to learn complex patterns.

### The Comparison Head

```python
def forward(self, t_a, s_a, t_b, s_b, pair_feats):
    u = forward_one(t_a, s_a)  # Encode A
    v = forward_one(t_b, s_b)  # Encode B
    
    diff_abs = |u - v|     # Where do they differ?
    prod     = u * v       # Where do they agree?
    
    x = concat([diff_abs, prod, pair_feats])  # (67,)
    return classifier(x)  # → logit
```

**Why both `|u - v|` AND `u * v`?**

These two operations capture complementary information:

| Operation | Intuition | Example |
|-----------|-----------|---------|
| $|u - v|$ | **Where do A and B differ?** | Large values → fields don't match |
| $u \cdot v$ | **Where do A and B agree?** | Large values → fields overlap |

Together, they give the classifier a rich comparison signal. Research (e.g., InferSent, SentenceBERT) has shown this combination outperforms either alone.

**Why inject `pair_feats` after encoding?**

Pairwise features (amount diff, date diff, ref match) are **pre-computed relationship signals** that don't need to be learned from embeddings. Injecting them directly into the comparison head gives the model explicit access to structured similarity signals alongside the learned representations.

---

## Step 9: Listwise Cross-Entropy Loss

### What Happens

The loss function treats each episode as a **multi-class classification** problem: "Which of these K candidates is the correct match?"

### Mathematical Formulation

For an episode with K candidates and logits $z_1, z_2, \ldots, z_K$, where the positive candidate is at index $y$:

$$\mathcal{L} = -\log \frac{e^{z_y}}{\sum_{k=1}^{K} e^{z_k}}$$

This is equivalent to **softmax cross-entropy** over the candidate set.

### Implementation

```python
def listwise_ce_from_groups(logits, lengths, pos_ixs):
    start = 0
    losses = []
    
    for i in range(B):  # B = number of episodes in batch
        K = lengths[i]                          # Candidates in this episode
        z = logits[start : start+K]             # Logits for this episode
        y = pos_ixs[i]                          # Index of positive candidate
        losses.append(cross_entropy(z, y))      # CE loss for this episode
        start += K
    
    return mean(losses)  # Average across episodes
```

### Why Listwise (Not Pointwise or Pairwise)?

| Approach | Loss Function | Limitation |
|----------|---------------|------------|
| **Pointwise** | BCE on individual pairs | No ranking context; scores aren't comparable |
| **Pairwise** | Hinge/margin on (pos, neg) pairs | Quadratic in candidates; doesn't optimize for top-1 |
| **Listwise (✅ used here)** | Softmax CE over full candidate set | Directly optimizes: "push the positive to rank 1" |

**Listwise CE is optimal for our use case** because we ultimately care about **one thing**: is the correct match ranked first?

### Worked Example

```
Episode: Query A vs. [Candidate₀=positive, Candidate₁, Candidate₂]

Model outputs logits: [2.1, 0.5, -0.3]
Positive index: 0

Softmax: [exp(2.1), exp(0.5), exp(-0.3)] / sum
       = [8.166, 1.649, 0.741] / 10.556
       = [0.774, 0.156, 0.070]

Loss = -log(0.774) = 0.256  ← Low loss (correct candidate ranked first)

Now imagine logits: [0.5, 2.1, -0.3]  (positive NOT ranked first)
Softmax: [0.156, 0.774, 0.070]
Loss = -log(0.156) = 1.858  ← High loss (model penalized)
```

---

## Step 10: Training Loop with Early Stopping

### What Happens

The model is trained using Adam optimizer with a training/validation split. Early stopping prevents overfitting by restoring the best model when validation performance degrades.

### Training Flow

```
For each epoch:
  ┌─── TRAIN ───────────────────────────────────────┐
  │ model.train()                                    │
  │ For each batch:                                  │
  │   Forward pass → logits                          │
  │   Compute listwise CE loss                       │
  │   Backward pass → gradients                      │
  │   Optimizer step → update weights                │
  └──────────────────────────────────────────────────┘
  
  ┌─── VALIDATE ────────────────────────────────────┐
  │ model.eval()                                     │
  │ torch.no_grad()                                  │
  │ For each batch:                                  │
  │   Forward pass → logits                          │
  │   Compute loss + metrics (P@1, MRR)              │
  └──────────────────────────────────────────────────┘
  
  ┌─── EARLY STOPPING CHECK ────────────────────────┐
  │ If val_loss improved:                            │
  │   Save model state (best_model_state)            │
  │   Reset patience counter                         │
  │ Else:                                            │
  │   Increment patience counter                     │
  │   If counter >= patience: STOP                   │
  └──────────────────────────────────────────────────┘

After training:
  Restore best_model_state
```

### Why Early Stopping?

Without early stopping, the model follows a typical learning curve:

```
Epoch:    1    2    3    4    5    6    7    8    9   10
          │    │    │    │    │    │    │    │    │    │
Train:    ████ ███  ██   █    █    ▌    ▌    ▎    ▎    ▏  → Keeps decreasing
Val:      ████ ███  ██   █    █    █▌   ██   ██▌  ███  ████ → Starts increasing!
                              ▲
                    OPTIMAL POINT (lowest val loss)
                    
                    After this, the model OVERFITS:
                    - Memorizes training quirks
                    - Loses generalization ability
```

Early stopping automatically identifies the optimal point and restores the model to that state.

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EPOCHS` | 4 (configurable) | Maximum training epochs |
| `patience` | 3 | Stop after 3 epochs without improvement |
| `lr` | 1e-3 | Adam learning rate |
| `BATCH_SIZE` | 8 | Episodes per batch |

---

## Step 11: Evaluation Metrics

### What Happens

The trained model is evaluated on the held-out test set using ranking metrics.

### Metrics Explained

#### Precision@1 (P@1)

> *"In what fraction of episodes did the model rank the correct match first?"*

$$P@1 = \frac{\text{Number of episodes where positive is ranked \#1}}{\text{Total episodes}}$$

**Example:**
```
Episode 1: Correct match ranked #1 ✅
Episode 2: Correct match ranked #3 ❌
Episode 3: Correct match ranked #1 ✅
Episode 4: Correct match ranked #2 ❌
Episode 5: Correct match ranked #1 ✅

P@1 = 3/5 = 0.600
```

**Interpretation:** P@1 = 0.60 means "60% of the time, the model's top recommendation is correct."

#### Recall@1 (R@1)

In our setting (exactly 1 positive per episode), R@1 = P@1. This would differ if there were multiple correct matches per query.

#### Mean Reciprocal Rank (MRR)

> *"On average, how high does the correct match appear in the ranking?"*

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

**Example:**
```
Episode 1: Correct match at rank 1 → 1/1 = 1.000
Episode 2: Correct match at rank 3 → 1/3 = 0.333
Episode 3: Correct match at rank 1 → 1/1 = 1.000
Episode 4: Correct match at rank 2 → 1/2 = 0.500
Episode 5: Correct match at rank 1 → 1/1 = 1.000

MRR = (1.0 + 0.333 + 1.0 + 0.5 + 1.0) / 5 = 0.767
```

**Interpretation:** MRR = 0.767 means "on average, the correct match appears between rank 1 and rank 2."

**Why MRR > P@1?** MRR gives partial credit for near-misses. If the correct match is ranked #2 instead of #1, P@1 counts it as a failure (0), but MRR gives it 0.5 credit.

### Implementation

```python
def batch_metrics_from_logits(logits, lengths, pos_ixs):
    for each episode:
        z = logits for this episode       # (K,)
        sorted_indices = argsort(z, descending=True)
        rank = position of positive in sorted_indices
        
        if rank == 0: hits += 1           # P@1 numerator
        mrr_sum += 1 / (rank + 1)         # MRR numerator (1-indexed)
    
    return hits, episodes, mrr_sum
```

---

## End-to-End Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      RAW TRADE DATA                            │
│  Each row: Trade Id, ISIN, Currency, Amount, Dates, Match ID   │
└────────────────────────────┬───────────────────────────────────┘
                             │
                    Filter & Clean (Step 1)
                    Keep 1-to-1 matched pairs
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                   CLEAN MATCHED PAIRS                          │
│  Each Match ID has exactly 2 rows (Trade A ↔ Trade B)          │
└────────────────────────────┬───────────────────────────────────┘
                             │
                    Normalize Text (Step 2)
                    Create combined_text column
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              Trade Id  │  ISIN          │  combined_text       │
│              T00000046 │  US9128283D82  │  "t00000046 us91..." │
└────────────────────────┬───────────────────────────────────────┘
                         │
                Split by Match ID Groups (Step 3)
                70/15/15, no group leakage
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          df_train    df_val    df_test
              │          │          │
              │     Build Episodes (Step 4)
              │     Query → [Pos + K Neg]
              ▼          ▼          ▼
         episodes    episodes    episodes
          _train      _val       _test
              │
         Fit TF-IDF (Step 5)
         Train-only!
              │
              ▼
         vectorizer (frozen)
              │
              │   Transform all episodes (Step 6)
              │   text → (T,), scalars → (2,), pairs → (3,)
              ▼
         ┌─────────────────────────────────────────────┐
         │  Per candidate pair:                         │
         │    t_a: (297,)  t_b: (297,)  — TF-IDF       │
         │    s_a: (2,)    s_b: (2,)    — Scalars       │
         │    pf:  (3,)                  — Pair features │
         └──────────────────────┬──────────────────────┘
                                │
                    DataLoader + Collation (Step 7)
                    Flatten episodes into batches
                                │
                                ▼
                    ┌───────────────────────┐
                    │   SIAMESE NETWORK     │
                    │   Encoder (shared)     │ (Step 8)
                    │   + Comparison Head    │
                    └───────────┬───────────┘
                                │
                          logits (N,)
                                │
                    ┌───────────┴───────────┐
                    │                       │
            Listwise CE Loss          Ranking Metrics
            (Step 9)                  (Step 11)
                    │                  P@1, R@1, MRR
            Backpropagation
            + Early Stopping
            (Step 10)
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Siamese Network** | Neural architecture with two identical sub-networks sharing weights, used for similarity comparison |
| **Episode** | One query trade + its set of candidate matches (1 positive, K negatives) |
| **Blocking** | Pre-filtering candidates by hard constraints (currency, date, amount) before scoring |
| **Hard Negative** | A candidate that passes all blocking filters but is NOT the correct match — forces the model to learn subtle distinctions |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; weights terms by how important they are to a document relative to the corpus |
| **char_wb** | Character-level n-grams within word boundaries; ideal for structured codes and identifiers |
| **Listwise Loss** | Loss function that operates over a full candidate set, optimizing the ranking directly |
| **P@1 (Precision at 1)** | Fraction of queries where the top-ranked candidate is correct |
| **MRR (Mean Reciprocal Rank)** | Average of $1/\text{rank}$ of the correct answer across all queries |
| **Early Stopping** | Halting training when validation performance stops improving, then restoring the best model |
| **Data Leakage** | Contamination of training with information from the test set, producing artificially high metrics |
| **Group Split** | Splitting data so that entire Match ID groups stay in the same fold, preventing leakage |
| **Embedding** | A learned fixed-length vector representation of a trade in a shared space |
| **Comparison Head** | The part of the network that takes two embeddings and predicts a match score |
| **Pairwise Features** | Pre-computed relationship signals (amount diff, date diff) injected directly into the comparison head |
| **Match ID** | Identifier linking two trades that form a confirmed match |
| **Comments** | High-level category of how a match was determined (e.g., "FX Revaluation") |
| **Match Rule** | Specific logic used to establish a match (e.g., "Sub Ledger Account + Absolute of Transactional Amount") |
