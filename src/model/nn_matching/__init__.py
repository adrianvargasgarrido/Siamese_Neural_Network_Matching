"""
nn_matching — Siamese neural network for trade matching.

This package provides a complete pipeline for training and deploying
Siamese networks to match financial trades across systems.

Main components:
  - models: Neural network architecture, dataset, and loss functions
  - pipeline: Data preprocessing, candidate generation, and vectorization
  - utils: Shared utility functions
"""

# Import key classes and functions for convenience
from .models import (
    SiameseMatchingNet,
    RankingEpisodeDataset,
    collate_episodes_flat,
    listwise_ce_from_groups,
    batch_metrics_from_logits,
)

from .pipeline import (
    normalize,
    normalize_and_combine,
    add_date_int_cols,
    stratified_group_split_3way,
    get_candidates,
    build_training_episodes_parallel,
    build_training_episodes_sequential,
    build_training_episodes_spark,
    iter_episode_text,
    vectorize_episode,
)

__version__ = "0.1.0"

__all__ = [
    # models
    "SiameseMatchingNet",
    "RankingEpisodeDataset",
    "collate_episodes_flat",
    "listwise_ce_from_groups",
    "batch_metrics_from_logits",
    # pipeline
    "normalize",
    "normalize_and_combine",
    "add_date_int_cols",
    "stratified_group_split_3way",
    "get_candidates",
    "build_training_episodes_parallel",
    "build_training_episodes_sequential",
    "build_training_episodes_spark",
    "iter_episode_text",
    "vectorize_episode",
]
