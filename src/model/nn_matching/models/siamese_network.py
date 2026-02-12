"""
siamese_network.py — Siamese neural network, dataset, and collation.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..pipeline.vectorization import vectorize_episode


# ─── Network ─────────────────────────────────────────────────────────

class SiameseMatchingNet(nn.Module):
    """Siamese network with text + scalar encoder and comparison head."""

    def __init__(self, text_input_dim, scalar_input_dim=2, pair_feat_dim=3, embed_dim=32):
        super(SiameseMatchingNet, self).__init__()

        # Encoder
        self.text_fc = nn.Linear(text_input_dim, embed_dim)
        self.scalar_fc = nn.Linear(scalar_input_dim, 8)
        self.encode_mix = nn.Linear(embed_dim + 8, embed_dim)

        # Comparison Head
        comparison_dim = (embed_dim * 2) + pair_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(comparison_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward_one(self, text_vec, scalar_vec):
        t = torch.relu(self.text_fc(text_vec))
        s = torch.relu(self.scalar_fc(scalar_vec))
        combined = torch.cat([t, s], dim=1)
        embedding = torch.relu(self.encode_mix(combined))
        return embedding

    def forward(self, t_a, s_a, t_b, s_b, pair_feats):
        u = self.forward_one(t_a, s_a)
        v = self.forward_one(t_b, s_b)

        diff_abs = torch.abs(u - v)
        prod = u * v

        x = torch.cat([diff_abs, prod, pair_feats], dim=1)
        return self.classifier(x)


# ─── Dataset & Collation ─────────────────────────────────────────────

class RankingEpisodeDataset(Dataset):
    """Episode-based dataset for listwise training."""

    def __init__(self, episodes, vectorizer, amount_col, date_cols, ref_col=None):
        self.episodes = episodes
        self.vectorizer = vectorizer
        self.amount_col = amount_col
        self.date_cols = date_cols
        self.ref_col = ref_col

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        query_row = ep["query_row"]
        candidates_df = ep["candidates_df"]

        vec_q, scal_q, vec_C, scal_C, pair_C, pos_ix = vectorize_episode(
            query_row, candidates_df,
            vectorizer=self.vectorizer,
            amount_col=self.amount_col,
            date_cols=self.date_cols,
            ref_col=self.ref_col,
        )

        K = vec_C.shape[0]
        assert K >= 1, "Each episode must contain at least 1 candidate."

        t_as = np.repeat(vec_q[None, :], K, axis=0).astype(np.float32)
        s_as = np.repeat(scal_q[None, :], K, axis=0).astype(np.float32)
        t_bs = vec_C.astype(np.float32)
        s_bs = scal_C.astype(np.float32)
        pf = pair_C.astype(np.float32)

        return {
            "t_as": torch.from_numpy(t_as),
            "s_as": torch.from_numpy(s_as),
            "t_bs": torch.from_numpy(t_bs),
            "s_bs": torch.from_numpy(s_bs),
            "pf": torch.from_numpy(pf),
            "length": torch.tensor(K, dtype=torch.long),
            "pos_ix": torch.tensor(int(pos_ix), dtype=torch.long),
        }


def collate_episodes_flat(batch):
    """Concatenate all pair rows from episodes in the batch."""
    t_as = torch.cat([b["t_as"] for b in batch], dim=0)
    s_as = torch.cat([b["s_as"] for b in batch], dim=0)
    t_bs = torch.cat([b["t_bs"] for b in batch], dim=0)
    s_bs = torch.cat([b["s_bs"] for b in batch], dim=0)
    pf = torch.cat([b["pf"] for b in batch], dim=0)

    lengths = torch.stack([b["length"] for b in batch], dim=0)
    pos_ixs = torch.stack([b["pos_ix"] for b in batch], dim=0)

    return {
        "t_as": t_as, "s_as": s_as,
        "t_bs": t_bs, "s_bs": s_bs,
        "pf": pf,
        "lengths": lengths, "pos_ixs": pos_ixs,
    }
