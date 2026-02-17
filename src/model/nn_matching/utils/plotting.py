"""
plotting — Visualisation functions for the Siamese Trade Matching pipeline.

Extracted from siamese_txn_matching.ipynb to keep the notebook concise.
Each function encapsulates one chart cell and prints interpretive guidance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


__all__ = [
    "plot_data_filtering_funnel",
    "plot_split_distribution",
    "plot_episode_inspection",
    "plot_tfidf_similarity",
    "plot_augmentation_impact",
    "plot_training_curves",
    "plot_test_score_distribution",
    "plot_embedding_evolution",
    "plot_weight_evolution",
    "plot_layer_activations",
    "plot_feature_importance",
]


# ─── Helper ──────────────────────────────────────────────────────────────


def _apply_style():
    """Apply consistent seaborn styling."""
    sns.set_theme(style="whitegrid", context="notebook")


# ─── 1. Data Filtering Funnel ────────────────────────────────────────────


def plot_data_filtering_funnel(
    n_raw_matched: int,
    n_after_rule: int,
    n_after_id: int,
    n_1to1: int,
    df_matched_1to1: pd.DataFrame,
    comments_col: str = "Comments",
):
    """Data filtering funnel (left) and match-rule distribution (right)."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    stages = ["Raw matched", "After rule filter", "After ID filter", "1-to-1 pairs"]
    counts = [n_raw_matched, n_after_rule, n_after_id, n_1to1]
    colors = ["#bbb", "#8cb4d9", "#4a90d9", "#2563eb"]
    bars = axes[0].barh(stages[::-1], counts[::-1], color=colors[::-1], edgecolor="white")
    for bar, c in zip(bars, counts[::-1]):
        axes[0].text(
            bar.get_width() + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,}", va="center", fontsize=10, fontweight="bold",
        )
    axes[0].set_xlabel("Rows")
    axes[0].set_title("Data Filtering Funnel")
    axes[0].set_xlim(0, max(counts) * 1.25)

    rule_counts = df_matched_1to1[comments_col].value_counts()
    axes[1].barh(rule_counts.index, rule_counts.values, color="#4a90d9", edgecolor="white")
    for i, (v, _) in enumerate(zip(rule_counts.values, rule_counts.index)):
        axes[1].text(v + max(rule_counts.values) * 0.02, i, f"{v:,}", va="center", fontsize=10)
    axes[1].set_xlabel("Rows")
    axes[1].set_title("Match Rules (1-to-1 pairs)")

    plt.tight_layout()
    plt.show()


# ─── 2. Split Distribution ───────────────────────────────────────────────


def plot_split_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    comments_col: str = "Comments",
):
    """Split sizes (left) and rule distribution per split (right)."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    split_names = ["Train", "Val", "Test"]
    split_sizes = [len(df_train), len(df_val), len(df_test)]
    colors = ["#2563eb", "#f59e0b", "#10b981"]
    bars = axes[0].bar(split_names, split_sizes, color=colors, edgecolor="white", width=0.5)
    for bar, s in zip(bars, split_sizes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(split_sizes) * 0.02,
            f"{s:,}", ha="center", fontsize=11, fontweight="bold",
        )
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Split Sizes")
    axes[0].set_ylim(0, max(split_sizes) * 1.2)

    all_rules = sorted(
        set(df_train[comments_col].unique())
        | set(df_val[comments_col].unique())
        | set(df_test[comments_col].unique())
    )
    x = np.arange(len(all_rules))
    w = 0.25
    for i, (name, df_split, color) in enumerate(
        zip(split_names, [df_train, df_val, df_test], colors)
    ):
        counts = [len(df_split[df_split[comments_col] == r]) for r in all_rules]
        axes[1].bar(x + i * w, counts, w, label=name, color=color, edgecolor="white")

    axes[1].set_xticks(x + w)
    axes[1].set_xticklabels(all_rules, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Rule Distribution per Split")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ─── 3. Episode Inspection ───────────────────────────────────────────────


def plot_episode_inspection(episodes_train: list):
    """Episode structure summary and candidate-count / rule distributions."""
    _apply_style()
    if not episodes_train:
        print("No training episodes to inspect.")
        return

    ep = episodes_train[0]
    print("📊 EPISODE INSPECTION (Example: Episode 0)")
    print("─" * 40)
    print(f"  • Query ID:       {ep['query_id']}")
    print(f"  • Match Rule:     {ep['rule']}")
    print(f"  • Positive Index: {ep['positive_index']} (Always 0)")
    print(f"  • Total Items:    {len(ep['candidates_df'])} (1 Positive + {len(ep['candidates_df'])-1} Negatives)")
    print(f"  • Candidate IDs:  {ep['candidate_ids'][:3]} ...")

    cand_counts = [len(e["candidates_df"]) for e in episodes_train]
    rule_counts = pd.Series([e["rule"] for e in episodes_train]).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(
        cand_counts,
        bins=np.arange(min(cand_counts), max(cand_counts) + 2) - 0.5,
        kde=False, color="#3498db", ax=axes[0], edgecolor="white", alpha=0.9,
    )
    axes[0].axvline(
        np.mean(cand_counts), color="#e74c3c", ls="--", lw=2,
        label=f"Avg: {np.mean(cand_counts):.1f}",
    )
    axes[0].set_title("Distribution of Episode Sizes (Train)", fontsize=13, fontweight="bold", loc="left")
    axes[0].set_xlabel("Number of Candidates (Pos + Negs)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    sns.despine(ax=axes[0])

    sns.barplot(x=rule_counts.values, y=rule_counts.index, palette="viridis", ax=axes[1], orient="h")
    for i, v in enumerate(rule_counts.values):
        axes[1].text(v + max(rule_counts.values) * 0.02, i, f"{v}", va="center", fontweight="bold")
    axes[1].set_title("Episode Composition by Match Rule", fontsize=13, fontweight="bold", loc="left")
    axes[1].set_xlabel("Number of Episodes")
    axes[1].grid(axis="x", alpha=0.3)
    sns.despine(ax=axes[1], left=True, bottom=True)

    plt.tight_layout()
    plt.show()

    print("\n💡 INTERPRETATION:")
    print("• Left Chart: Shows how many candidates (hard negatives + 1 positive) are in each training episode.")
    print("  - If consistent (e.g., spike at 21), the blocker successfully found the requested 'k' negatives.")
    print("  - If skewed left, the blocker struggled to find enough hard negatives (easy queries).")
    print("• Right Chart: Shows the diversity of match logic. A balanced dataset helps the model learn generalized matching.")


# ─── 4. TF-IDF Similarity ────────────────────────────────────────────────


def plot_tfidf_similarity(episodes_train: list, vectorizer, n_episodes: int = 1000):
    """Positive vs negative cosine-similarity distributions in TF-IDF space."""
    _apply_style()
    pos_sims, neg_sims = [], []

    for ep in episodes_train[:n_episodes]:
        q_text = ep["query_row"].get("combined_text", "")
        cands = ep["candidates_df"]
        if cands is None or cands.empty or "combined_text" not in cands.columns:
            continue
        texts = [q_text] + cands["combined_text"].fillna("").tolist()
        vecs = vectorizer.transform(texts).toarray()
        sims = cos_sim(vecs[0:1], vecs[1:])[0]
        pos_sims.append(sims[0])
        neg_sims.extend(sims[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bins = np.linspace(0, 1, 40)
    axes[0].hist(neg_sims, bins=bins, alpha=0.6, density=True, label=f"Negatives (n={len(neg_sims):,})", color="#e74c3c")
    axes[0].hist(pos_sims, bins=bins, alpha=0.6, density=True, label=f"Positives (n={len(pos_sims):,})", color="#27ae60")
    axes[0].set_title("TF-IDF Similarity: Positives vs Negatives", fontsize=13, fontweight="bold", loc="left")
    axes[0].set_xlabel("Cosine Similarity (0-1)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    sns.despine(ax=axes[0])

    data_sim = (
        [{"Type": "Positive", "Similarity": x} for x in pos_sims]
        + [{"Type": "Negative", "Similarity": x} for x in neg_sims]
    )
    df_sim = pd.DataFrame(data_sim)
    sns.boxplot(
        data=df_sim, x="Type", y="Similarity",
        hue="Type", palette={"Positive": "#2ecc71", "Negative": "#e74c3c"},
        ax=axes[1], width=0.5, legend=False,
    )
    axes[1].set_title("Similarity Distribution Stats", fontsize=13, fontweight="bold", loc="left")
    axes[1].set_ylabel("Cosine Similarity")
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.show()

    sep_gap = np.mean(pos_sims) - np.mean(neg_sims)
    print(f"📊 STATS:")
    print(f"  • Positive Mean Sim: {np.mean(pos_sims):.3f} ± {np.std(pos_sims):.3f}")
    print(f"  • Negative Mean Sim: {np.mean(neg_sims):.3f} ± {np.std(neg_sims):.3f}")
    print(f"  • Separation Gap:    {sep_gap:.3f}")
    print("\n💡 INTERPRETATION:")
    if sep_gap > 0.3:
        print("  • High Separation (>0.3): TF-IDF alone is quite strong. The model will converge quickly.")
    else:
        print("  • Low Separation (<0.3): The Hard Negatives are textually very similar to the Query.")
        print("    The Neural Network (NN) is essential here to use other features (Amounts, Dates, Patterns) to distinguish them.")


# ─── 5. Augmentation Impact ──────────────────────────────────────────────


def plot_augmentation_impact(
    original_row,
    augment_query_fn,
    vectorizer,
    columns_to_normalize: list,
    amount_col: str,
    date_cols: list,
    date_int_cols: list,
    augment_params: dict,
):
    """Demonstrate each perturbation individually, then combined + TF-IDF impact."""
    _apply_style()

    rng_demo = np.random.default_rng(42)
    orig_text = str(original_row.get("combined_text", ""))
    orig_amount = original_row.get(amount_col, 0.0)
    orig_dates = {c: original_row.get(f"{c}_int", None) for c in date_cols}

    print("=" * 80)
    print("AUGMENTATION DEMO — What happens to a real query row")
    print("=" * 80)
    print(f"\n📋 ORIGINAL QUERY:")
    print(f"   combined_text : '{orig_text[:100]}{'...' if len(orig_text) > 100 else ''}'")
    print(f"   {amount_col}: {orig_amount}")
    for c, v in orig_dates.items():
        print(f"   {c}_int: {v}")

    perturbations = {
        "1. Token Dropout": {"enable_token_dropout": True, "token_drop_prob": 0.25},
        "2. Token Swap": {"enable_token_swap": True, "token_swap_prob": 1.0},
        "3. Char Noise": {"enable_char_noise": True, "char_noise_prob": 0.15},
        "4. Synonym Sub": {"enable_synonym_sub": True, "synonym_sub_prob": 1.0},
        "5. Field Omission": {"enable_field_omission": True, "field_drop_prob": 1.0},
        "6. Scalar Noise": {"enable_scalar_noise": True, "amount_jitter_std": 0.01, "date_shift_prob": 1.0, "date_shift_max": 2},
    }

    results = {}
    for name, params in perturbations.items():
        rng_demo = np.random.default_rng(42)
        aug_row = augment_query_fn(
            original_row,
            columns_to_normalize=columns_to_normalize,
            amount_col=amount_col,
            date_int_cols=date_int_cols,
            rng=rng_demo,
            **params,
        )
        aug_text = str(aug_row.get("combined_text", ""))
        aug_amount = aug_row.get(amount_col, 0.0)
        aug_dates = {c: aug_row.get(f"{c}_int", None) for c in date_cols}
        results[name] = {"text": aug_text, "amount": aug_amount, "dates": aug_dates, "row": aug_row}

        print(f"\n{'─' * 70}")
        print(f"🔧 {name}")

        if aug_text != orig_text:
            orig_tokens = set(orig_text.split())
            aug_tokens = set(aug_text.split())
            removed = orig_tokens - aug_tokens
            added = aug_tokens - orig_tokens
            print(f"   Text: '{aug_text[:100]}{'...' if len(aug_text) > 100 else ''}'")
            if removed:
                print(f"   ❌ Removed tokens: {removed}")
            if added:
                print(f"   ✅ Added tokens: {added}")
            if not removed and not added:
                print("   🔄 Token order changed (same tokens, different order)")
        else:
            print("   Text: (unchanged)")

        if abs(float(aug_amount) - float(orig_amount)) > 1e-6:
            pct = abs(float(aug_amount) - float(orig_amount)) / max(abs(float(orig_amount)), 1e-12) * 100
            print(f"   Amount: {float(orig_amount):.2f} → {float(aug_amount):.2f} ({pct:.3f}% change)")
        else:
            print("   Amount: (unchanged)")

        for c in date_cols:
            ov = orig_dates[c]
            av = aug_dates[c]
            if ov is not None and av is not None and abs(float(av) - float(ov)) > 0.1:
                print(f"   {c}_int: {int(ov)} → {int(av)} ({int(float(av) - float(ov)):+d} days)")

    # ── Combined ──
    print(f"\n{'=' * 80}")
    print("🚀 COMBINED — All enabled perturbations (your current AUGMENT_PARAMS config):")
    print(f"{'=' * 80}")

    rng_demo = np.random.default_rng(42)
    combined_row = augment_query_fn(
        original_row,
        columns_to_normalize=columns_to_normalize,
        amount_col=amount_col,
        date_int_cols=date_int_cols,
        rng=rng_demo,
        **augment_params,
    )
    comb_text = str(combined_row.get("combined_text", ""))
    comb_amount = combined_row.get(amount_col, 0.0)

    print(f"   Original text:   '{orig_text[:80]}...'")
    print(f"   Augmented text:  '{comb_text[:80]}...'")
    pct = abs(float(comb_amount) - float(orig_amount)) / max(abs(float(orig_amount)), 1e-12) * 100
    print(f"   Original amount: {float(orig_amount):.2f}")
    print(f"   Augmented amount: {float(comb_amount):.2f} ({pct:.3f}% change)")
    for c in date_cols:
        ov = float(orig_dates[c]) if orig_dates[c] is not None else 0
        av = float(combined_row.get(f"{c}_int", ov))
        if abs(av - ov) > 0.1:
            print(f"   {c}_int: {int(ov)} → {int(av)} ({int(av - ov):+d} days)")

    # ── Charts ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    orig_vec = vectorizer.transform([orig_text]).toarray()[0]
    sim_scores = {}
    for name, res in results.items():
        aug_vec = vectorizer.transform([res["text"]]).toarray()[0]
        sim = cos_sim(orig_vec.reshape(1, -1), aug_vec.reshape(1, -1))[0, 0]
        sim_scores[name.split(". ")[1]] = sim

    comb_vec = vectorizer.transform([comb_text]).toarray()[0]
    sim_scores["Combined"] = cos_sim(orig_vec.reshape(1, -1), comb_vec.reshape(1, -1))[0, 0]

    colors_sim = ["#27ae60" if v > 0.95 else "#f39c12" if v > 0.85 else "#e74c3c" for v in sim_scores.values()]
    bars = axes[0].barh(list(sim_scores.keys()), list(sim_scores.values()), color=colors_sim, edgecolor="white")
    axes[0].set_xlim(0.5, 1.05)
    axes[0].axvline(1.0, ls="--", color="#bbb", alpha=0.5)
    for bar, v in zip(bars, sim_scores.values()):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_title("TF-IDF Cosine Sim vs Original", fontsize=12, fontweight="bold", loc="left")
    axes[0].set_xlabel("Cosine Similarity")
    sns.despine(ax=axes[0])

    orig_nz = np.count_nonzero(orig_vec)
    comb_nz = np.count_nonzero(comb_vec)
    axes[1].bar(["Original", "Augmented"], [orig_nz, comb_nz], color=["#2563eb", "#e74c3c"], edgecolor="white", width=0.5)
    for i, v in enumerate([orig_nz, comb_nz]):
        axes[1].text(i, v + max(orig_nz, comb_nz) * 0.02, str(v), ha="center", fontsize=11, fontweight="bold")
    axes[1].set_title("Active TF-IDF Dimensions", fontsize=12, fontweight="bold", loc="left")
    axes[1].set_ylabel("Non-zero features")
    sns.despine(ax=axes[1])

    amounts = []
    scalar_params = {k: v for k, v in augment_params.items() if k in ["enable_scalar_noise", "amount_jitter_std", "date_shift_max", "date_shift_prob"]}
    for seed in range(100):
        r = np.random.default_rng(seed)
        aug_r = augment_query_fn(
            original_row,
            columns_to_normalize=columns_to_normalize,
            amount_col=amount_col,
            date_int_cols=date_int_cols,
            rng=r,
            **scalar_params,
        )
        amounts.append(float(aug_r.get(amount_col, orig_amount)))

    pct_changes = [(a - float(orig_amount)) / max(abs(float(orig_amount)), 1e-12) * 100 for a in amounts]
    axes[2].hist(pct_changes, bins=25, color="#2563eb", edgecolor="white", alpha=0.8)
    axes[2].axvline(0, ls="--", color="#e74c3c", lw=2, label="Original")
    axes[2].set_title("Amount Noise Distribution (100 samples)", fontsize=12, fontweight="bold", loc="left")
    axes[2].set_xlabel("% Change from Original")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    sns.despine(ax=axes[2])

    plt.suptitle("Data Augmentation: Impact on Model Inputs", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n💡 HOW TO READ THIS:")
    print("• TF-IDF Cosine Similarity: How 'close' the augmented text is to the original in TF-IDF space.")
    print("  Green (>0.95) = very mild noise.  Yellow (0.85-0.95) = moderate.  Red (<0.85) = aggressive.")
    print("  Goal: Keep similarity high enough that the model CAN still match, but low enough to add diversity.")
    print("• Active TF-IDF Dimensions: Token dropout/field omission REDUCES non-zero dimensions.")
    print("  Char noise may CHANGE some but keeps roughly the same count.")
    print("• Amount Noise: Shows the distribution of percentage changes across 100 random augmentations.")
    print("  With std=0.005, most changes are within ±1% — realistic for rounding/FX differences.")


# ─── 6. Training Curves ──────────────────────────────────────────────────


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_precisions: list,
    val_mrrs: list,
    score_margins: list,
    gradient_norms: list,
    per_rule_history: dict,
):
    """6-panel training introspection dashboard."""
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    epochs_range = range(1, len(train_losses) + 1)

    # 1 — Loss
    axes[0, 0].plot(epochs_range, train_losses, "o-", label="Train", color="#2563eb", lw=2)
    axes[0, 0].plot(epochs_range, val_losses, "s-", label="Val", color="#f59e0b", lw=2)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Listwise CE Loss")
    axes[0, 0].set_title("Loss Convergence", fontsize=13, fontweight="bold", loc="left")
    axes[0, 0].legend(); sns.despine(ax=axes[0, 0])

    # 2 — Metrics
    axes[0, 1].plot(epochs_range, val_precisions, "o-", label="P@1", color="#2563eb", lw=2)
    axes[0, 1].plot(epochs_range, val_mrrs, "s-", label="MRR", color="#f59e0b", lw=2)
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Ranking Quality", fontsize=13, fontweight="bold", loc="left")
    axes[0, 1].set_ylim(0, 1.05); axes[0, 1].legend(); sns.despine(ax=axes[0, 1])

    # 3 — Margin
    colors_m = ["#ef4444" if m < 0 else "#22c55e" for m in score_margins]
    axes[0, 2].bar(epochs_range, score_margins, color=colors_m, edgecolor="white", width=0.6)
    axes[0, 2].axhline(0, color="black", lw=0.5)
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Avg Margin (pos − max neg)")
    axes[0, 2].set_title("Score Margin Evolution", fontsize=13, fontweight="bold", loc="left")
    sns.despine(ax=axes[0, 2])

    # 4 — Gradients
    axes[1, 0].plot(epochs_range, gradient_norms, "o-", color="#8b5cf6", lw=2)
    axes[1, 0].fill_between(epochs_range, 0, gradient_norms, alpha=0.15, color="#8b5cf6")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("L2 Gradient Norm")
    axes[1, 0].set_title("Gradient Flow", fontsize=13, fontweight="bold", loc="left")
    sns.despine(ax=axes[1, 0])

    # 5 — Per-rule P@1
    rule_colors = sns.color_palette("husl", len(per_rule_history))
    for i, (rule, history) in enumerate(sorted(per_rule_history.items())):
        rule_epochs = range(1, len(history) + 1)
        axes[1, 1].plot(rule_epochs, history, "o-", label=rule[:25], color=rule_colors[i], lw=2, markersize=4)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("P@1")
    axes[1, 1].set_title("P@1 by Match Rule", fontsize=13, fontweight="bold", loc="left")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(fontsize=7, loc="lower right"); sns.despine(ax=axes[1, 1])

    # 6 — Overfitting gap
    gap = [t - v for t, v in zip(train_losses, val_losses)]
    colors_gap = ["#22c55e" if g <= 0 else "#ef4444" for g in gap]
    axes[1, 2].bar(epochs_range, gap, color=colors_gap, edgecolor="white", width=0.6)
    axes[1, 2].axhline(0, color="black", lw=0.5)
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("Train − Val Loss")
    axes[1, 2].set_title("Overfitting Monitor", fontsize=13, fontweight="bold", loc="left")
    sns.despine(ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

    print("\n💡 HOW TO READ THESE CHARTS:")
    print("• Loss Convergence: Both curves should decrease. If train drops but val rises → overfitting.")
    print("• Ranking Quality: P@1 = 'did the model pick the right match first?'. MRR rewards partial correctness.")
    print("• Score Margin: Positive margin means the model scores the correct match HIGHER than all negatives.")
    print("  - Growing margin = increasing confidence. Negative margin = model is confused.")
    print("• Gradient Flow: Should start high (learning) and decrease (converging). Flat = saturated/stuck.")
    print("• P@1 by Rule: Shows which match types the model learns fastest. Lagging rules need more data.")
    print("• Overfitting Monitor: Green bars = healthy. Red bars = train loss << val loss (memorising).")


# ─── 7. Test Score Distribution ───────────────────────────────────────────


def plot_test_score_distribution(
    episodes_test: list,
    model,
    vectorizer,
    amount_col: str,
    date_cols: list,
    ref_col,
    device,
    vectorize_episode_fn,
):
    """Score separation, rank distribution, and P@1 per rule on the test set."""
    _apply_style()
    model.eval()
    all_pos_scores, all_neg_scores = [], []
    all_ranks, all_rules = [], []

    with torch.no_grad():
        for ep in episodes_test:
            q = ep["query_row"]
            c = ep["candidates_df"]
            vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode_fn(
                q, c, vectorizer=vectorizer, amount_col=amount_col,
                date_cols=date_cols, ref_col=ref_col,
            )
            K = vec_C.shape[0]
            t_a = torch.from_numpy(np.repeat(vec_q[None, :], K, axis=0)).to(device)
            s_a = torch.from_numpy(np.repeat(scal_q[None, :], K, axis=0)).to(device)
            t_b = torch.from_numpy(vec_C).to(device)
            s_b = torch.from_numpy(scal_C).to(device)
            pf = torch.from_numpy(pair_C).to(device)
            logits = model(t_a, s_a, t_b, s_b, pf).squeeze(-1)
            scores = logits.cpu().numpy()
            all_pos_scores.append(scores[0])
            all_neg_scores.extend(scores[1:].tolist())
            rank = int((np.argsort(-scores) == 0).argmax())
            all_ranks.append(rank + 1)
            all_rules.append(ep.get("rule", "UNKNOWN"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bins = np.linspace(
        min(min(all_neg_scores), min(all_pos_scores)),
        max(max(all_neg_scores), max(all_pos_scores)), 50,
    )
    axes[0].hist(all_neg_scores, bins=bins, density=True, alpha=0.6, label=f"Negatives (n={len(all_neg_scores):,})", color="#e74c3c")
    axes[0].hist(all_pos_scores, bins=bins, density=True, alpha=0.6, label=f"Positives (n={len(all_pos_scores):,})", color="#27ae60")
    axes[0].set_xlabel("Logit Score", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Density", fontsize=11, fontweight="bold")
    axes[0].set_title("Score Separation", fontsize=13, fontweight="bold", loc="left")
    axes[0].legend(frameon=True, framealpha=0.9, facecolor="white", loc="upper right")
    sns.despine(ax=axes[0], left=True)

    rank_counts = pd.Series(all_ranks).value_counts().sort_index()
    colors = ["#27ae60" if r == 1 else "#95a5a6" for r in rank_counts.index]
    bp = sns.barplot(x=rank_counts.index, y=rank_counts.values, hue=rank_counts.index, palette=colors, ax=axes[1], legend=False)
    axes[1].set_xlabel("Rank of Correct Match", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[1].set_title("Rank Distribution", fontsize=13, fontweight="bold", loc="left")
    for p in bp.patches:
        if p.get_height() > 0:
            bp.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height()), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#34495e")
    sns.despine(ax=axes[1], left=True)

    rule_df = pd.DataFrame({"rule": all_rules, "rank": all_ranks})
    rule_df["hit"] = (rule_df["rank"] == 1).astype(int)
    per_rule = rule_df.groupby("rule").agg(p_at_1=("hit", "mean"), count=("hit", "count")).sort_values("p_at_1", ascending=True)
    y_pos = np.arange(len(per_rule))
    axes[2].barh(y_pos, per_rule["p_at_1"], color="#3498db", height=0.6, edgecolor="white")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(per_rule.index, fontsize=10)
    axes[2].set_xlabel("Precision @ 1", fontsize=11, fontweight="bold")
    axes[2].set_title("Performance by Rule", fontsize=13, fontweight="bold", loc="left")
    axes[2].set_xlim(0, 1.25)
    sns.despine(ax=axes[2], bottom=True, left=True)
    axes[2].grid(axis="x", alpha=0.3)
    for i, (p, n) in enumerate(zip(per_rule["p_at_1"], per_rule["count"])):
        axes[2].text(p + 0.02, i, f"{p:.1%} (n={n})", va="center", fontsize=10, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    plt.show()

    print(f"\nTest Rank Distribution:")
    for r in sorted(rank_counts.index):
        pct_val = 100 * rank_counts[r] / len(all_ranks)
        print(f"  Rank {r}: {rank_counts[r]} episodes ({pct_val:.1f}%)")


# ─── 8. Embedding Evolution ──────────────────────────────────────────────


def plot_embedding_evolution(embedding_snapshots: list):
    """PCA-projected embeddings at each snapshot epoch."""
    _apply_style()
    if not embedding_snapshots:
        print("No embedding snapshots captured. Run the training loop first.")
        return

    n_snaps = len(embedding_snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(6 * n_snaps, 5))
    if n_snaps == 1:
        axes = [axes]

    palette = {"Query": "#2563eb", "Positive": "#27ae60", "Negative": "#e74c3c"}

    for idx, snap in enumerate(embedding_snapshots):
        embs = snap["embeddings"]
        labels = snap["labels"]
        ep_num = snap["epoch"]
        if len(embs) < 3:
            continue
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        df_pca = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Type": labels})
        for t in ["Negative", "Positive", "Query"]:
            subset = df_pca[df_pca["Type"] == t]
            axes[idx].scatter(subset["PC1"], subset["PC2"], c=palette[t], label=t, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
        axes[idx].set_title(f"Epoch {ep_num}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        axes[idx].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        if idx == 0:
            axes[idx].legend(loc="upper right", fontsize=8)
        sns.despine(ax=axes[idx])

    plt.suptitle("Embedding Space Evolution During Training", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n💡 HOW TO READ THIS:")
    print("• Each dot is a 32-d embedding projected onto 2D via PCA.")
    print("• Blue (Query) and Green (Positive) should CONVERGE over epochs — the model learns they are the same trade.")
    print("• Red (Negative) should SEPARATE from the Query/Positive cluster.")
    print("• If everything is clumped together → the model hasn't learned meaningful differences yet.")


# ─── 9. Weight Evolution ─────────────────────────────────────────────────


def plot_weight_evolution(weight_snapshots: dict):
    """Weight distribution (mean ± σ, min-max) for key layers over training."""
    _apply_style()
    key_layers = ["text_fc.weight", "encode_mix.weight", "classifier.0.weight", "classifier.3.weight"]
    available_layers = [k for k in key_layers if k in weight_snapshots]

    if not available_layers:
        print("No weight snapshots captured. Run the training loop first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, layer_name in enumerate(available_layers[:4]):
        stats = weight_snapshots[layer_name]
        epochs_range_w = range(1, len(stats["mean"]) + 1)
        ax = axes_flat[idx]
        ax.fill_between(epochs_range_w, stats["min"], stats["max"], alpha=0.15, color="#3498db", label="Min–Max range")
        ax.fill_between(
            epochs_range_w,
            [m - s for m, s in zip(stats["mean"], stats["std"])],
            [m + s for m, s in zip(stats["mean"], stats["std"])],
            alpha=0.3, color="#3498db", label="Mean ± 1σ",
        )
        ax.plot(epochs_range_w, stats["mean"], "o-", color="#2c3e50", lw=2, markersize=4, label="Mean")
        ax.set_title(layer_name, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Weight Value")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")
        sns.despine(ax=ax)

    for idx in range(len(available_layers), 4):
        axes_flat[idx].axis("off")

    plt.suptitle("Weight Distribution Evolution", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    print("\n💡 HOW TO READ THIS:")
    print("• Each panel shows ONE layer's weight statistics over training epochs.")
    print("• text_fc.weight: The first layer that compresses TF-IDF (64k dims) → 32 dims.")
    print("  This layer learns WHICH character n-grams matter for matching.")
    print("• encode_mix.weight: Fuses text (32d) + scalars (8d) → final embedding (32d).")
    print("  This layer learns HOW to combine text similarity with amount/date signals.")
    print("• classifier.0.weight / classifier.3.weight: The comparison head.")
    print("  These learn WHAT patterns in |u-v| and u⊙v indicate a match.")
    print("• Healthy: Weights should spread out (σ increases) then stabilise.")
    print("• Unhealthy: Weights collapse to 0 (dying ReLU) or explode (divergence).")


# ─── 10. Layer Activations ────────────────────────────────────────────────


def plot_layer_activations(
    model,
    episodes_test: list,
    vectorizer,
    amount_col: str,
    date_cols: list,
    ref_col,
    device,
    vectorize_episode_fn,
):
    """Feed one test episode through the network and visualise activations at every stage."""
    _apply_style()
    model.eval()
    sample_ep = episodes_test[0]
    q = sample_ep["query_row"]
    c = sample_ep["candidates_df"]

    vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode_fn(
        q, c, vectorizer=vectorizer, amount_col=amount_col,
        date_cols=date_cols, ref_col=ref_col,
    )

    K = vec_C.shape[0]
    t_a = torch.from_numpy(np.repeat(vec_q[None, :], K, axis=0)).to(device)
    s_a = torch.from_numpy(np.repeat(scal_q[None, :], K, axis=0)).to(device)
    t_b = torch.from_numpy(vec_C).to(device)
    s_b = torch.from_numpy(scal_C).to(device)
    pf_t = torch.from_numpy(pair_C).to(device)

    with torch.no_grad():
        text_out_q = torch.relu(model.text_fc(t_a))
        text_out_c = torch.relu(model.text_fc(t_b))
        u = model.forward_one(t_a, s_a)
        v = model.forward_one(t_b, s_b)
        diff_abs = torch.abs(u - v)
        prod = u * v
        logits = model(t_a, s_a, t_b, s_b, pf_t).squeeze(-1)

    activations = {
        "Text Encoder (Query)": text_out_q[0].cpu().numpy(),
        "Text Encoder (Pos)": text_out_c[0].cpu().numpy(),
        "Text Encoder (Neg)": text_out_c[1].cpu().numpy() if K > 1 else text_out_c[0].cpu().numpy(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Panel 1 — Text encoder
    ax = axes[0, 0]
    x_dims = np.arange(32)
    ax.bar(x_dims - 0.25, activations["Text Encoder (Query)"], width=0.25, label="Query", color="#2563eb", alpha=0.8)
    ax.bar(x_dims, activations["Text Encoder (Pos)"], width=0.25, label="Positive", color="#27ae60", alpha=0.8)
    ax.bar(x_dims + 0.25, activations["Text Encoder (Neg)"], width=0.25, label="Neg #1", color="#e74c3c", alpha=0.8)
    ax.set_title("Text Encoder Output (32 dims)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Dimension"); ax.set_ylabel("Activation"); ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Panel 2 — Fused embedding heatmap
    ax = axes[0, 1]
    embed_matrix = np.vstack([u[0].cpu().numpy(), v[0].cpu().numpy()])
    if K > 1:
        embed_matrix = np.vstack([embed_matrix, v[1].cpu().numpy()])
        row_labels = ["Query", "Positive", "Neg #1"]
    else:
        row_labels = ["Query", "Positive"]
    sns.heatmap(embed_matrix, cmap="RdYlGn", center=0, ax=ax, yticklabels=row_labels, cbar_kws={"shrink": 0.8})
    ax.set_title("Fused Embeddings (32-d Fingerprint)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Embedding Dimension")

    # Panel 3 — |u-v|
    ax = axes[1, 0]
    ax.bar(x_dims - 0.15, diff_abs[0].cpu().numpy(), width=0.3, label="|u-v| Positive", color="#27ae60", alpha=0.8)
    if K > 1:
        ax.bar(x_dims + 0.15, diff_abs[1].cpu().numpy(), width=0.3, label="|u-v| Negative", color="#e74c3c", alpha=0.8)
    ax.set_title("Absolute Difference |u-v| (Classifier Input)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Dimension"); ax.set_ylabel("|Difference|"); ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Panel 4 — Final logits
    ax = axes[1, 1]
    all_logits = logits.cpu().numpy()
    colors_l = ["#27ae60"] + ["#e74c3c"] * (len(all_logits) - 1)
    ax.bar(range(len(all_logits)), all_logits, color=colors_l, edgecolor="white")
    ax.set_title("Final Logits (All Candidates)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Candidate Index (0 = Positive)"); ax.set_ylabel("Logit Score")
    ax.axhline(all_logits[0], ls="--", color="#27ae60", alpha=0.5, label=f"Positive: {all_logits[0]:.3f}")
    if len(all_logits) > 1:
        ax.axhline(max(all_logits[1:]), ls="--", color="#e74c3c", alpha=0.5, label=f"Max Neg: {max(all_logits[1:]):.3f}")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    plt.suptitle("NN Internals: How One Episode Flows Through the Network", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    print("\n💡 HOW TO READ THIS:")
    print("• Text Encoder: Shows the 32-d compressed representation of each trade's TF-IDF text.")
    print("  Query and Positive should have SIMILAR activation patterns (same trade, different IDs).")
    print("• Fused Embeddings: The final 32-d 'identity vector' for each trade after combining text + scalars.")
    print("  Similar rows in the heatmap = similar trades.")
    print("• |u-v| (Absolute Difference): This is what the classifier looks at to decide 'match or not'.")
    print("  For the POSITIVE pair, |u-v| should be SMALL (near zero). For NEGATIVES, it should be LARGE.")
    print("• Final Logits: The model's confidence score for each candidate.")
    print("  The Positive (index 0, green) should have the HIGHEST logit → model ranks it first.")


# ─── 11. Feature Importance ──────────────────────────────────────────────


def plot_feature_importance(
    model,
    episodes_test: list,
    vectorizer,
    amount_col: str,
    date_cols: list,
    ref_col,
    device,
    vectorize_episode_fn,
):
    """Gradient-based feature importance: which inputs drive the match decision."""
    _apply_style()
    model.eval()
    sample_ep = episodes_test[0]
    q = sample_ep["query_row"]
    c = sample_ep["candidates_df"]

    vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode_fn(
        q, c, vectorizer=vectorizer, amount_col=amount_col,
        date_cols=date_cols, ref_col=ref_col,
    )

    K = vec_C.shape[0]
    t_a = torch.from_numpy(np.repeat(vec_q[None, :], K, axis=0)).to(device).requires_grad_(True)
    s_a = torch.from_numpy(np.repeat(scal_q[None, :], K, axis=0)).to(device).requires_grad_(True)
    t_b = torch.from_numpy(vec_C).to(device).requires_grad_(True)
    s_b = torch.from_numpy(scal_C).to(device).requires_grad_(True)
    pf_t = torch.from_numpy(pair_C).to(device).requires_grad_(True)

    logits = model(t_a, s_a, t_b, s_b, pf_t).squeeze(-1)
    logit_pos = logits[0]
    logit_pos.backward()

    text_grad_q = t_a.grad[0].abs().cpu().numpy()
    text_grad_c = t_b.grad[0].abs().cpu().numpy()
    scalar_grad_q = s_a.grad[0].abs().cpu().numpy()
    scalar_grad_c = s_b.grad[0].abs().cpu().numpy()
    pair_grad = pf_t.grad[0].abs().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Panel 1 — TF-IDF sensitivity
    ax = axes[0, 0]
    combined_text_grad = text_grad_q + text_grad_c
    top_20 = np.argsort(combined_text_grad)[-20:]
    ax.barh(range(20), combined_text_grad[top_20], color="#2563eb", alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"dim {i}" for i in top_20], fontsize=8)
    ax.set_title("Top 20 Sensitive TF-IDF Dims", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("|∂logit/∂input|"); sns.despine(ax=ax)

    # Panel 2 — Scalar sensitivity
    ax = axes[0, 1]
    s_dim = scal_q.shape[0]
    s_labels = ["amt_zscore", "recency"] if s_dim == 2 else [f"s{i}" for i in range(s_dim)]
    scalar_names_q = [f"Query: {n}" for n in s_labels]
    scalar_names_c = [f"Cand: {n}" for n in s_labels]
    all_scalar_names = scalar_names_q + scalar_names_c
    all_scalar_grads = np.concatenate([scalar_grad_q, scalar_grad_c])
    sorted_idx = np.argsort(all_scalar_grads)
    ax.barh(range(len(all_scalar_names)), all_scalar_grads[sorted_idx], color="#f39c12", alpha=0.8)
    ax.set_yticks(range(len(all_scalar_names)))
    ax.set_yticklabels([all_scalar_names[i] for i in sorted_idx], fontsize=9)
    ax.set_title("Scalar Feature Sensitivity", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("|∂logit/∂input|"); sns.despine(ax=ax)

    # Panel 3 — Pair feature sensitivity
    ax = axes[1, 0]
    pair_names = [f"pair_{i}" for i in range(pair_grad.shape[0])]
    sorted_idx_p = np.argsort(pair_grad)
    ax.barh(range(len(pair_names)), pair_grad[sorted_idx_p], color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels([pair_names[i] for i in sorted_idx_p], fontsize=9)
    ax.set_title("Pair Feature Sensitivity", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("|∂logit/∂input|"); sns.despine(ax=ax)

    # Panel 4 — Element-wise product
    ax = axes[1, 1]
    with torch.no_grad():
        u = model.forward_one(t_a[:1], s_a[:1])
        all_v = model.forward_one(t_b, s_b)
        pos_prod = (u * all_v[0:1]).squeeze().cpu().numpy()
        neg_prods = (u * all_v[1:]).mean(dim=0).cpu().numpy() if K > 1 else pos_prod * 0

    x_d = np.arange(min(32, len(pos_prod)))
    ax.bar(x_d - 0.15, pos_prod[:32], width=0.3, label="u⊙v (Positive)", color="#27ae60", alpha=0.8)
    if K > 1:
        ax.bar(x_d + 0.15, neg_prods[:32], width=0.3, label="u⊙v (Neg avg)", color="#e74c3c", alpha=0.8)
    ax.set_title("Element-wise Product: Pos vs Neg", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("u ⊙ v"); ax.legend(fontsize=8)
    sns.despine(ax=ax)

    plt.suptitle("Feature Importance: What Drives the Match Score?", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    print("\n💡 FEATURE IMPORTANCE INTERPRETATION:")
    print("• TF-IDF Sensitivity: Shows which text dimensions the classifier relies on most.")
    print("  High-gradient dims correspond to n-grams that are discriminative for matching.")
    print("• Scalar Sensitivity: How much each numeric feature (amount z-score, recency) affects the score.")
    print("• Pair Feature Sensitivity: Direct comparison features (log-amount-diff, etc.) and their importance.")
    print("• u⊙v Product: The element-wise agreement between query and candidate embeddings.")
    print("  Positive matches should have HIGHER agreement (larger u⊙v values) than negatives.")
