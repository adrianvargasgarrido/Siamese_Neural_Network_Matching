"""
pipeline — Data preprocessing, candidate generation, and vectorization.
"""

from .data_prep import (
    normalize,
    normalize_and_combine,
    rebuild_combined_text_for_row,
    add_date_int_cols,
    stratified_group_split_3way,
)
from .candidate_generation import (
    get_candidates,
    build_training_episodes_single_df_debug,
    build_training_episodes_parallel,
    build_training_episodes_sequential,
    build_training_episodes_spark,
)
from .vectorization import (
    iter_episode_text,
    vectorize_episode,
)

__all__ = [
    # data_prep
    "normalize",
    "normalize_and_combine",
    "rebuild_combined_text_for_row",
    "add_date_int_cols",
    "stratified_group_split_3way",
    # candidate_generation
    "get_candidates",
    "build_training_episodes_single_df_debug",
    "build_training_episodes_parallel",
    "build_training_episodes_sequential",
    "build_training_episodes_spark",
    # vectorization
    "iter_episode_text",
    "vectorize_episode",
]
