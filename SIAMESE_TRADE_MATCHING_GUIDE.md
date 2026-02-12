# Siamese Neural Network for Trade Matching

## A Theoretical Guide

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [What Is a Siamese Network?](#2-what-is-a-siamese-network)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Data Cleaning & Filtering](#4-data-cleaning--filtering)
5. [Text Normalisation & Feature Engineering](#5-text-normalisation--feature-engineering)
6. [Stratified Group Splitting](#6-stratified-group-splitting)
7. [Episode Construction](#7-episode-construction)
8. [TF-IDF Vectorisation](#8-tf-idf-vectorisation)
9. [Feature Assembly](#9-feature-assembly)
10. [PyTorch Dataset & Collation](#10-pytorch-dataset--collation)
11. [Network Architecture](#11-network-architecture)
12. [Listwise Cross-Entropy Loss](#12-listwise-cross-entropy-loss)
13. [Training with Early Stopping](#13-training-with-early-stopping)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [Glossary](#15-glossary)

---

## 1. Introduction

### The problem

Trade matching (trade reconciliation) pairs two sides of a financial transaction — e.g. a buy from System A and a sell from System B — to confirm they represent the same economic event.

Rule-based systems use deterministic logic like "same currency AND same amount AND same date". They are brittle to data-quality issues, require manual rule maintenance, and cannot rank among multiple plausible candidates.

### The approach

This project replaces (or augments) rule-based matching with a **learning-to-rank** pipeline built on a Siamese neural network. The model learns from historical matched pairs and generalises to unseen trades. Given a query trade and a set of candidates, it produces a ranking — the most likely match appears at the top.

---

## 2. What Is a Siamese Network?

A Siamese network processes two inputs through **identical sub-networks** (same architecture, same weights) and compares the resulting representations.

```
     Trade A                    Trade B
        │                          │
   ┌────┴────┐                ┌────┴────┐
   │ Encoder │                │ Encoder │    ← shared weights
   │  f(·)   │                │  f(·)   │
   └────┬────┘                └────┬────┘
        │                          │
     u = f(A)                  v = f(B)
        │                          │
        └──────────┬───────────────┘
                   │
            ┌──────┴──────┐
            │  Comparison │
            │    Head     │
            └──────┬──────┘
                   │
              score (logit)
```

Weight sharing forces the network to learn a **universal embedding** of trades rather than memorising specific pairings. Any new trade is embedded in the same space and can be compared to any other.

This differs from classification: a classifier asks "what category does this belong to?", which fails when new Match IDs appear constantly. A Siamese network asks "how similar are these two?" — a question that generalises to any unseen pair.

---

## 3. Pipeline Overview

```
Raw trade data
  │
  ├─ 1. Clean & filter → valid 1-to-1 matched pairs
  ├─ 2. Normalise text  → combined_text column
  ├─ 3. Split by group  → train / val / test (no leakage)
  ├─ 4. Build episodes  → query + [positive, K negatives]
  ├─ 5. Fit TF-IDF      → char n-grams on train only
  ├─ 6. Assemble features → text vectors, scalars, pair features
  ├─ 7. DataLoader      → batch and collate episodes
  ├─ 8. Forward pass    → shared encoder + comparison head → logit
  ├─ 9. Listwise CE     → softmax over candidates
  ├─ 10. Train + early stop → restore best model
  └─ 11. Evaluate       → P@1, MRR on test set
```

Each step feeds the next. The pipeline is modular: data preparation, candidate generation, vectorisation, model, and loss live in separate modules.

---

## 4. Data Cleaning & Filtering

### Goal

Reduce the raw data to a clean set of confirmed 1-to-1 matched pairs that provide a reliable training signal.

### Steps

1. **Filter by match type** — keep only match categories where the pairing logic is well-defined (e.g. matches by instrument identifier, by derived trade ID, by FX revaluation).
2. **Remove invalid Match IDs** — drop rows where the Match ID is null, blank, or corrupt.
3. **Keep groups of size 2** — a Siamese network compares pairs. Match IDs with exactly two rows give one unambiguous pair; groups of 3+ introduce ambiguity about which rows form the "real" match.

### Why filter by match type?

Different match categories have different economics. Mixing unrelated categories dilutes the training signal. Restricting to specific types lets the model learn patterns that generalise within those matching contexts.

---

## 5. Text Normalisation & Feature Engineering

### Normalisation

Each text field is cleaned:

1. Convert to lowercase, strip whitespace.
2. Replace null-like values (`"none"`, `"nan"`, `"n/a"`) with empty string.
3. Replace non-word characters with spaces and collapse whitespace.

### Combined text

Multiple identifier columns — trade IDs, ISINs, CUSIPs, SEDOLs, instrument names — are normalised and concatenated into a single `combined_text` string per trade.

This serves three purposes:

- **Single vectorisation step** — one TF-IDF transform instead of many.
- **Cross-field matching** — if Trade A's ISIN overlaps with Trade B's CUSIP, the shared character n-grams will surface naturally.
- **Missing-data tolerance** — empty fields are simply omitted from the concatenation.

---

## 6. Stratified Group Splitting

The data is split into **train (70 %)**, **validation (15 %)**, and **test (15 %)**.

### Group constraint

Both sides of a matched pair must go into the **same** split. If Trade A is in train and its partner Trade B is in test, the model has already seen A's "twin" during training — this is data leakage and inflates metrics.

The split is performed at the Match ID level: entire groups stay together.

### Stratification

Within each split, the distribution of match types is kept roughly proportional. This prevents any split from being dominated by a single match category.

If the data is too small for stratification (singleton classes), the code falls back to a random group shuffle.

---

## 7. Episode Construction

### What is an episode?

An episode is one realistic matching scenario:

```
Query (A):  synthetic copy of a known matched trade
Candidates: [positive (original trade), neg₁, neg₂, …, negₖ]
```

The model must rank the positive above all negatives.

### Synthetic query creation

The episode builder clones a matched row B to create query A, then:

1. Replaces A's Trade ID with a synthetic identifier.
2. Rebuilds A's `combined_text` from scratch.

Without this step, query and positive would have identical text — the model would learn trivial exact-string matching instead of useful shared-identifier patterns. The synthetic ID forces the model to rely on ISINs, CUSIPs, instrument names, amounts, and dates.

### Hard-negative retrieval (blocking)

Negatives are not random. They are retrieved using the same **blocking** pipeline the model will face at inference:

1. **Currency block** — only same-currency candidates.
2. **Date window** — candidates within a configurable number of days.
3. **Amount tolerance** — candidates within a configurable percentage of the query amount.
4. **Rank & top-K** — sort by reference-field match (descending), amount difference (ascending), date difference (ascending); take the top K.

These "hard negatives" pass all blocking filters but are not the correct match. They force the model to learn subtle discriminative features.

If blocking returns fewer than K negatives, random trades from the pool fill the gap.

### Builder variants

Three interchangeable builders exist. All call the same `_build_one_episode` worker and produce identical results:

| Builder | Execution | Best for |
|---------|-----------|----------|
| `build_training_episodes_parallel` | `ProcessPoolExecutor` (multi-core) | Local machines |
| `build_training_episodes_sequential` | Plain for-loop | Restricted environments, no extra dependencies |
| `build_training_episodes_spark` | Spark RDD + broadcast | Databricks |

The notebook auto-selects based on environment detection.

---

## 8. TF-IDF Vectorisation

### Configuration

- **Analyser:** `char_wb` — character n-grams within word boundaries.
- **N-gram range:** (2, 4) — bigrams, trigrams, and 4-grams.
- **Fitted on training episodes only.**

### Why character n-grams?

Trade identifiers are structured codes (e.g. `US9128283D82`), not natural language. Character n-grams:

- Capture **partial overlap** between similar codes.
- Are **typo-resilient** — most n-grams survive a single-character error.
- Enable **cross-field matching** — shared substrings between different identifier types produce overlapping n-grams.

Word-level TF-IDF would treat each identifier as a single token — either an exact match or nothing.

### Train-only fitting

Fitting the vectoriser on all data would let the vocabulary include n-grams that exist only in test trades. This is data leakage. The vocabulary must come exclusively from training; validation and test texts are transformed with this fixed vocabulary.

---

## 9. Feature Assembly

Each query–candidate pair is represented by three feature groups.

### Text vectors

The query's and each candidate's `combined_text` are transformed by the (frozen) TF-IDF vectoriser into dense float vectors of dimension V (the vocabulary size).

### Scalar features (per trade)

| Feature | Formula | Purpose |
|---------|---------|---------|
| Log-amount | $\log(1 + |\text{amount}|)$ | Normalised trade size; log-scaling compresses the wide range of financial amounts |
| Date norm | $\min(\text{date\_ints}) / 365$ | Earliest date normalised to approximate years |

### Pairwise features (per query–candidate pair)

| Feature | Formula | Purpose |
|---------|---------|---------|
| Log amount diff | $\log(1 + |A_{\text{amt}} - B_{\text{amt}}|)$ | How different the amounts are |
| Log min date diff | $\log(1 + \min_d |A_d - B_d|)$ | How far apart the closest dates are |
| Reference exact | $\mathbb{1}[A_{\text{ref}} = B_{\text{ref}}]$ | Whether reference fields match exactly |

Pairwise features are **pre-computed relationship signals** — they give the comparison head direct access to structured similarity measures that would be difficult to recover from embeddings alone.

---

## 10. PyTorch Dataset & Collation

### Dataset

Each episode is vectorised on-the-fly in `__getitem__`: the query's features are repeated K times (once per candidate) so that every row represents one (query, candidate) pair.

### Flat collation

Instead of padding episodes to the same length, the collator **concatenates** all pair-rows across episodes:

```
Episode 1:  K₁ candidates  →  K₁ pair-rows
Episode 2:  K₂ candidates  →  K₂ pair-rows
─────────────────────────────────────────────
Batch:      N = K₁ + K₂     →  one forward pass
```

A `lengths` tensor records how many candidates belong to each episode. A `pos_ixs` tensor records the position of the positive in each group. The loss function uses these to split logits back into per-episode groups.

Benefits: no padding waste, variable-size episodes, and all pairs processed in a single forward pass.

---

## 11. Network Architecture

### Shared encoder

Both query and candidate pass through the same encoder:

```
text_vec (V,)  ─→  Linear(V → 32) + ReLU  ─┐
                                              ├─ concat (40,) ─→ Linear(40 → 32) + ReLU ─→ embedding (32,)
scalar_vec (2,) ─→  Linear(2 → 8)  + ReLU  ─┘
```

Text and scalar projections are separate because text is high-dimensional (V features) while scalars are 2-D. Projecting independently prevents the scalars from being dominated. The fusion layer combines them into a 32-dimensional embedding.

### Comparison head

Given query embedding $u$ and candidate embedding $v$:

```
|u − v|      (32,)   ← where do they differ?
u ⊙ v        (32,)   ← where do they agree?
pair_feats   (3,)    ← amount diff, date diff, ref match
──────────────────
concat       (67,)
Linear(67 → 16) + ReLU + Dropout(0.2)
Linear(16 → 1) → logit
```

Using both $|u - v|$ and $u \cdot v$ captures complementary information: absolute difference highlights mismatches; element-wise product highlights agreement. Research on sentence-pair models (InferSent, SentenceBERT) has shown this combination outperforms either alone.

Pairwise features are injected **after** encoding — they are pre-computed relationship signals that do not need to be learned from raw inputs.

---

## 12. Listwise Cross-Entropy Loss

Each episode is treated as a K-way classification: "which of these K candidates is the correct match?"

For an episode with logits $z_1, \ldots, z_K$ and positive at index $y$:

$$\mathcal{L} = -\log \frac{e^{z_y}}{\sum_{k=1}^{K} e^{z_k}}$$

This is standard softmax cross-entropy over the candidate set.

### Why listwise?

| Approach | What it optimises | Limitation |
|----------|-------------------|------------|
| **Pointwise** (BCE per pair) | "Is this pair a match?" independently | No ranking context; scores not comparable across queries |
| **Pairwise** (margin on pos vs neg) | "Score positive above this negative" | Quadratic in candidates; does not directly optimise top-1 |
| **Listwise** (softmax over set) | "Push the positive to rank 1 among all candidates" | Directly optimises the ranking objective |

### Implementation detail

Because episodes in a batch have different numbers of candidates, logits are concatenated into a flat vector. The loss function uses the `lengths` tensor to slice logits back into per-episode groups and computes cross-entropy for each group independently. The final loss is the mean across all episodes in the batch.

---

## 13. Training with Early Stopping

### Loop

Each epoch:

1. **Train** — forward pass, listwise CE loss, backpropagation, Adam weight update.
2. **Validate** — forward pass (no gradients), compute loss and ranking metrics (P@1, MRR).
3. **Early stopping check** — if validation loss improved, save model state and reset patience. Otherwise increment counter. If patience exceeded, stop.

After training, the best model state is restored.

### Why early stopping?

Without it, the model eventually overfits: training loss keeps decreasing while validation loss starts increasing. Early stopping identifies the point of best generalisation automatically.

---

## 14. Evaluation Metrics

### Ranking metrics (currently computed)

**Precision@1 (P@1):**

$$P@1 = \frac{\text{episodes where positive is ranked \#1}}{\text{total episodes}}$$

"What fraction of the time is the model's top recommendation correct?"

**Mean Reciprocal Rank (MRR):**

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

"On average, how high does the correct match appear?" MRR gives partial credit for near-misses — ranking the correct match 2nd contributes 0.5 rather than 0.

In our setting there is exactly one positive per episode, so R@1 = P@1.

### Threshold metrics (with a score cut-off)

When a threshold τ routes pairs to auto-match vs manual review:

| Metric | Notes |
|--------|-------|
| PR-AUC | Precision–recall trade-off across all thresholds |
| ROC-AUC | Can be misleadingly high under class imbalance |
| F1 / Fβ | Balance or bias precision vs recall at a chosen operating point |

The model outputs raw logits consumed by softmax. For per-pair scores in [0, 1], a sigmoid would be applied to individual logits.

### Operational metrics

| Metric | Description |
|--------|-------------|
| Coverage | % of queries where blocking produces at least one candidate |
| Avg candidates / query | Review workload and compute cost |
| Latency | End-to-end time per query |

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Siamese network** | Architecture with two identical sub-networks sharing weights, for similarity comparison |
| **Episode** | One query trade + its candidate set (1 positive, K negatives) |
| **Blocking** | Pre-filtering candidates by hard constraints (currency, date, amount) before scoring |
| **Hard negative** | A candidate that passes blocking filters but is not the correct match |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; weights terms by importance relative to the corpus |
| **char_wb** | Character n-grams within word boundaries; suited for structured identifiers |
| **Listwise loss** | Loss over a full candidate set, directly optimising the ranking |
| **P@1** | Fraction of queries where the top-ranked candidate is the true match |
| **MRR** | Average of 1 / rank of the correct answer across queries |
| **Early stopping** | Halting training when validation loss plateaus; restoring the best model |
| **Data leakage** | Contamination of training with information from the test set |
| **Group split** | Splitting data so entire Match ID groups stay in the same fold |
| **Embedding** | Learned fixed-length vector representation of a trade |
| **Comparison head** | Network component that takes two embeddings and outputs a match score |
| **Pairwise features** | Pre-computed signals (amount diff, date diff, ref match) injected into the comparison head |
