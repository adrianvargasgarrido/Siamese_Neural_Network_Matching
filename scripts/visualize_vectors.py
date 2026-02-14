#!/usr/bin/env python3
"""
scripts/visualize_vectors.py

Loads a trained Siamese model and visualises the learned embeddings for a single episode.
Produces two plots:
1. 2D PCA Scatter: Shows the "cluster" of Query vs Positive vs Negatives.
2. Vector Fingerprint: Heatmap of the raw 32-d embeddings to see activation patterns.

Usage:
    python scripts/visualize_vectors.py
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Add src to path so we can import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Adjust these imports to match your precise struct
from src.model.nn_matching.models.siamese_network import SiameseMatchingNet, RankingEpisodeDataset
from src.model.nn_matching.pipeline.candidate_generation import build_training_episodes_sequential

# --- CONFIG ---
MODEL_PATH = PROJECT_ROOT / "best_siamese_model.pt"
# We need a dummy pool/config to generate ONE episode live if we don't have a cached one
# Or we can just mock some data. Let's try to load the model first.

def generate_mock_episode():
    """Generates a synthetic episode for visualisation if no real data is handy."""
    print("Generating mock data for visualisation...")
    # Mock data structure matching the pipeline
    rows = []
    # Positive
    rows.append({
        "Trade Id": "T-POS", 
        "Amount": 1000.0, 
        "Currency": "USD", 
        "DateInt": 20250101, 
        "combined_text": "US912828XY USD 1000 BOND",
        "Match Rule": "Test"
    })
    # Hard negative (similar text, diff amount)
    rows.append({
        "Trade Id": "T-NEG1", 
        "Amount": 1000.0, 
        "Currency": "USD", 
        "DateInt": 20250101, 
        "combined_text": "US912828ZZ USD 1000 BOND", # ZZ instead of XY
        "Match Rule": "Test"
    })
    # Easy negative (diff everything)
    rows.append({
        "Trade Id": "T-NEG2", 
        "Amount": 500.0, 
        "Currency": "EUR", 
        "DateInt": 20250105, 
        "combined_text": "GB00123 EUR 500 EQUITY", 
        "Match Rule": "Test"
    })
    
    # Fill more random negatives
    for i in range(10):
        rows.append({
            "Trade Id": f"T-RND{i}", 
            "Amount": 100.0 * i, 
            "Currency": "USD", 
            "DateInt": 20250101, 
            "combined_text": f"RANDOM TEXT {i}", 
            "Match Rule": "Test"
        })

    df = pd.DataFrame(rows)
    
    # Use builder to make an episode dict
    # We must ensure logic aligns with the real pipeline
    episodes, _ = build_training_episodes_sequential(
        df,
        id_col="Trade Id",
        currency_col="Currency",
        amount_col="Amount",
        date_int_cols=["DateInt"],
        columns_to_normalize=["Trade Id", "Currency", "Amount"],
        id_norm_col="Trade Id",
        n_episodes=1,
        train_k_neg=10, 
        verbose=False
    )
    if not episodes:
        raise ValueError("Could not build a mock episode.")
    
    return episodes[0]

def visualize_episode(model, episode, vectorizer):
    """Runs the episode through the model and plots the vectors."""
    model.eval()
    
    # 1. Prepare inputs (vectorize)
    # This logic mimics RankingEpisodeDataset.__getitem__
    # We provide the required cols to the constructor
    ds = RankingEpisodeDataset(
        [episode], 
        vectorizer=vectorizer, 
        amount_col="Amount", 
        date_cols=["DateInt"]
    )
    
    # Get tensors
    batch = ds[0] # dict with t_a, s_a, t_b, s_b, pair_feats
    
    # Get tensors
    batch = ds[0] # dict with t_as, s_as, t_bs, s_bs, pf
    
    # In RankingEpisodeDataset, 't_as' is already (K, T) (repeated query)
    # We don't need to unsqueeze or repeat.
    # We just grab the tensors directly.
    
    with torch.no_grad():
        print(f"Batch t_as shape: {batch['t_as'].shape}")
        
        # We need the underlying 'forward_one' method to get embeddings.
        # If the model doesn't expose it publicly, we might need to access submodules.
        # The SiameseMatchingNet usually has 'text_fc', 'scalar_fc', 'encode_mix'.
        # Let's define a helper to encode efficiently.

        def encode(t_batch, s_batch):
            # t_batch: (B, T), s_batch: (B, 2)
            # Reimplement forward pass for one branch
            # 1. Text branch
            txt_emb = F.relu(model.text_fc(t_batch))
            # 2. Scalar branch
            scl_emb = F.relu(model.scalar_fc(s_batch))
            # 3. Mixing
            combined = torch.cat([txt_emb, scl_emb], dim=1)
            u = F.relu(model.encode_mix(combined))
            return u

        # Encode Query (from first row of t_as, since they are all identical)
        # t_as is (K, T)
        t_q_single = batch['t_as'][0:1] # (1, T)
        s_q_single = batch['s_as'][0:1] # (1, 2)
        u_vec = encode(t_q_single, s_q_single)[0].numpy() # (32,)
        
        # Encode Candidates (all rows of t_bs)
        # t_bs is (K, T)
        t_c_batch = batch['t_bs']
        s_c_batch = batch['s_bs']
        v_vecs = encode(t_c_batch, s_c_batch).numpy() # (K, 32)

    # 2. Prepare plot data
    # Matrix: [Query, Pos, Neg1, Neg2, ...]
    vectors = np.vstack([u_vec, v_vecs])
    
    # Labels
    labels = ["Query"] + ["Positive"] + [f"Neg {i}" for i in range(len(v_vecs)-1)]
    types = ["Query"] + ["Positive"] + ["Negative"] * (len(v_vecs)-1)
    
    # --- Plot 1: Vector Fingerprint (Heatmap) ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(vectors[:10], cmap="viridis", cbar=True, annot=False) # Show top 10 rows only
    plt.yticks(np.arange(10) + 0.5, labels[:10], rotation=0)
    plt.xlabel("Embedding Dimension (0-31)")
    plt.title("Vector Fingerprint: Query vs Positive vs Negatives")
    plt.tight_layout()
    plt.savefig("visuals_fingerprint.png")
    print("Saved visuals_fingerprint.png")
    
    # --- Plot 2: PCA Projection ---
    pca = PCA(n_components=2)
    vec_2d = pca.fit_transform(vectors)
    
    df_plot = pd.DataFrame({
        "x": vec_2d[:, 0],
        "y": vec_2d[:, 1],
        "Type": types,
        "Label": labels
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x="x", y="y", hue="Type", style="Type", s=200, palette={"Query": "blue", "Positive": "green", "Negative": "red"})
    
    # Draw line from Query to Positive
    q_pt = df_plot[df_plot["Type"]=="Query"].iloc[0]
    p_pt = df_plot[df_plot["Type"]=="Positive"].iloc[0]
    plt.plot([q_pt.x, p_pt.x], [q_pt.y, p_pt.y], 'k--', alpha=0.5, label="Target Match")
    
    plt.title("2D PCA Projection of Trade Embeddings")
    plt.grid(True, alpha=0.3)
    plt.savefig("visuals_pca.png")
    print("Saved visuals_pca.png")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        print("Using random initialized model for demo purposes.")
        
    # Load vectorizer (pickled)
    # In a real run we reload it. For this script, we might need to recreate fit on mock data
    # if we don't have the pickle.
    import pickle
    VEC_PATH = PROJECT_ROOT / "params/tfidf_vectorizer.pkl"
    
    if os.path.exists(VEC_PATH):
        with open(VEC_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        vocab_size = len(vectorizer.vocabulary_)
    else:
        print("No vectorizer found. Fitting a dummy one on mock data.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
        # We need to fit it to make it work
        episode = generate_mock_episode()
        texts = [episode["query_row"]["combined_text"]] + episode["candidates_df"]["combined_text"].tolist()
        vectorizer.fit(texts)
        vocab_size = len(vectorizer.vocabulary_)

    # Init Model
    model = SiameseMatchingNet(text_input_dim=vocab_size, embed_dim=32)
    
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"]) 
            else:
                model.load_state_dict(state)
            print("Loaded trained metrics.")
        except Exception as e:
            print(f"Could not load state dict: {e}")
            
    # Generate data if we haven't already
    if 'episode' not in locals():
        episode = generate_mock_episode()
        
    # Run viz
    visualize_episode(model, episode, vectorizer)

if __name__ == "__main__":
    main()
