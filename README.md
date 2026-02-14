# Siamese Neural Network for Trade Matching

A learning-to-rank pipeline that uses a Siamese neural network to match financial trades. Given a query trade, the model scores and ranks a set of candidates to identify the most likely match.

---

## How It Works

1. **Data cleaning** — filter to valid 1-to-1 matched pairs with known match rules.
2. **Text normalisation** — lowercase, strip, and combine identifier columns (Trade Id, ISIN, CUSIP, SEDOL, etc.) into a single `combined_text` field.
3. **Stratified group split** — divide data into train / val / test by Match ID groups so no matched pair leaks across splits.
4. **Episode construction** — for each training example, clone a matched trade as the query (with a synthetic ID), keep the original as the positive, and retrieve hard negatives via blocking (currency, date window, amount tolerance).
5. **Data augmentation** — apply on-the-fly noise (typos, token swaps, date shifts) to query trades during training to improve robustness.
6. **TF-IDF vectorisation** — fit character n-gram (2–4) TF-IDF on training episodes only.
7. **Feature assembly** — each query–candidate pair gets: TF-IDF text vectors, scalar features (log-amount, date norm), and pairwise features (amount diff, date diff, reference match).
8. **Siamese network** — shared encoder (text → 32-d, scalars → 8-d, fused → 32-d) produces embeddings; comparison head combines |u−v|, u⊙v, and pair features to output a logit per candidate.
9. **Listwise cross-entropy loss** — softmax over candidate logits pushes the positive to rank 1.
10. **Early stopping** — training halts when validation loss stops improving; best model is restored.
11. **Evaluation** — P@1, MRR on the held-out test set.

---

## Project Structure

```
.
├── README.md                          ← This file
├── SIAMESE_TRADE_MATCHING_GUIDE.md    ← Detailed theoretical guide
├── CHANGE_LOG.md                      ← Bug-fix history
│
├── src/model/nn_matching/
│   ├── pipeline/
│   │   ├── data_prep.py               ← Normalisation, date handling, group splitting
│   │   ├── candidate_generation.py    ← Blocking, retrieval, episode builders
│   │   ├── vectorization.py           ← TF-IDF iteration, scalar/pair features
│   │   └── augmentation.py            ← Query noise simulation (typos, swaps, etc.)
│   ├── models/
│   │   ├── siamese_network.py         ← SiameseMatchingNet, Dataset, collation
│   │   └── losses.py                  ← Listwise CE loss, P@1 / MRR metrics
│   ├── notebooks/
│   │   ├── siamese_txn_matching.ipynb ← End-to-end training notebook
│   │   └── analysis_and_improvements.ipynb
│   ├── configs/                       ← (reserved for config files)
│   └── utils/
│
├── docs/
│   ├── DATA_AUGMENTATION_GUIDE.md     ← Guide to noise simulation techniques
│   ├── IMPORT_PATTERNS.md
│   ├── NOTEBOOK_WALKTHROUGH.md
│   └── visuals/
│       ├── README.md                  ← Visual documentation (Mermaid diagrams)
│   ├── pipeline_overview.mmd
│   ├── candidate_generation.mmd
│   ├── siamese_architecture.mmd
│   └── eval_metrics.mmd
│
├── scripts/
│   └── plot_eval.py                   ← Generates evaluation plots
│
└── plots_outputs/                     ← Generated plots
```

---

## Episode Builders

The episode construction step has three interchangeable builders. All produce identical results — they share the same `_build_one_episode` worker.

| Builder | How it runs | Best for |
|---------|------------|----------|
| `build_training_episodes_parallel` | `ProcessPoolExecutor` (multi-core) | Local machines with multiple cores |
| `build_training_episodes_sequential` | Plain for-loop, no extra dependencies | Restricted environments, work laptops |
| `build_training_episodes_spark` | Spark RDD + broadcast variables | Databricks clusters |

The notebook auto-detects the environment (`DATABRICKS_RUNTIME_VERSION`) and picks the appropriate builder.

---

## Data Augmentation

To improve model robustness, the pipeline supports online data augmentation during training. This simulates real-world errors by perturbing the *Query* side of the match.

Techniques implemented in `pipeline/augmentation.py`:
- **Token Dropout**: Randomly removing words (simulating data loss).
- **Token Swap**: Swapping adjacent words (simulating entry errors).
- **Character Noise**: Keyboard typos (qwerty adjacency) or random insertions/deletions.
- **Synonym Substitution**: Replacing words with domain equivalents (e.g., "CORP" -> "CORPORATION").
- **Field Omission**: Dropping entire fields from the combined text string.
- **Scalar Perturbation**: Jittering numeric amounts and shifting dates slightly.

Configuration for these can be found in the notebook's `AUGMENT_PARAMS` dictionary.

---

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, torch, matplotlib

Install:

```bash
pip install numpy pandas scikit-learn torch matplotlib
```

---

## Running

Open `src/model/nn_matching/notebooks/siamese_txn_matching.ipynb` and run all cells. The notebook handles path setup, data loading, training, and evaluation.

> **Note:** The data generation and column configuration files (`private_config.py`, `private_data_generation.py`) are gitignored. You need to provide your own data or create these files following the patterns in the notebook.
