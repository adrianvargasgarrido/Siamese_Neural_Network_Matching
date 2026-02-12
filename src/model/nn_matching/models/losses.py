"""
losses.py — Loss functions and metrics.
"""
import torch
import torch.nn.functional as F

# ─── Loss ─────────────────────────────────────────────────────────────

def listwise_ce_from_groups(logits, lengths, pos_ixs):
    """
    Listwise cross-entropy over grouped logits.

    Args:
        logits:  (N,) logits over all pairs in the batch
        lengths: (B,) number of candidates per episode
        pos_ixs: (B,) correct indices per episode
    """
    start = 0
    losses = []
    B = lengths.numel()

    for i in range(B):
        K = int(lengths[i].item())
        z = logits[start : start + K].unsqueeze(0)
        y = pos_ixs[i].unsqueeze(0)
        losses.append(F.cross_entropy(z, y))
        start += K

    return torch.stack(losses).mean()


# ─── Metrics ──────────────────────────────────────────────────────────

def batch_metrics_from_logits(logits, lengths, pos_ixs):
    """
    Compute ranking metrics (Hits@1, MRR) from grouped logits.

    Returns: (hits, episodes, mrr_sum)
    """
    start = 0
    hits = 0.0
    episodes = lengths.numel()
    mrr_sum = 0.0

    for i in range(episodes):
        K = int(lengths[i].item())
        z = logits[start : start + K]
        target_idx = int(pos_ixs[i].item())

        _, sorted_indices = torch.sort(z, descending=True)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item()

        if rank == 0:
            hits += 1.0
        mrr_sum += 1.0 / (rank + 1)

        start += K

    return hits, episodes, mrr_sum
