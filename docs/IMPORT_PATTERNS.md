# Import Patterns Guide

## Overview

The `__init__.py` files have been populated to expose clean public APIs. You can now use shorter, more Pythonic import patterns.

---

## Available Import Patterns

### Pattern 1: Direct Package Imports (Recommended)

```python
# Import from main package
from src.model.nn_matching import (
    SiameseMatchingNet,
    RankingEpisodeDataset,
    collate_episodes_flat,
    listwise_ce_from_groups,
    batch_metrics_from_logits,
    normalize,
    normalize_and_combine,
    add_date_int_cols,
    get_candidates,
    build_training_episodes_parallel,
    vectorize_episode,
)
```

### Pattern 2: Subpackage Imports

```python
# Import from subpackages
from src.model.nn_matching.models import (
    SiameseMatchingNet,
    RankingEpisodeDataset,
    listwise_ce_from_groups,
)

from src.model.nn_matching.pipeline import (
    normalize,
    get_candidates,
    build_training_episodes_parallel,
    vectorize_episode,
)
```

### Pattern 3: Module Imports (Most Explicit)

```python
# Import from individual modules (current notebook style)
from src.model.nn_matching.pipeline.data_prep import normalize, normalize_and_combine
from src.model.nn_matching.pipeline.candidate_generation import get_candidates
from src.model.nn_matching.models.siamese_network import SiameseMatchingNet
```

---

## What's Exported

### `src.model.nn_matching` (Top Level)

All key functions and classes from models and pipeline subpackages.

### `src.model.nn_matching.models`

- `SiameseMatchingNet` — Neural network model
- `RankingEpisodeDataset` — PyTorch dataset for episodes
- `collate_episodes_flat` — Batch collation function
- `listwise_ce_from_groups` — Listwise cross-entropy loss
- `batch_metrics_from_logits` — MRR/Recall metrics

### `src.model.nn_matching.pipeline`

**Data Prep:**
- `normalize` — Single text normalization
- `normalize_and_combine` — Normalize columns and create combined_text
- `rebuild_combined_text_for_row` — Rebuild text for single row
- `add_date_int_cols` — Convert dates to integer format
- `stratified_group_split_3way` — Train/val/test split

**Candidate Generation:**
- `get_candidates` — Retrieve candidates for single query
- `build_training_episodes_single_df_debug` — Debug episode builder
- `build_training_episodes_parallel` — Multi-process episode builder
- `build_training_episodes_sequential` — Simple for-loop builder
- `build_training_episodes_spark` — Spark-distributed builder

**Vectorization:**
- `iter_episode_text` — Iterate over all text in episodes
- `vectorize_episode` — Convert episode to model inputs

### `src.model.nn_matching.utils`

Currently empty, reserved for shared utilities.

---

## Migration Guide

### Before (Current Notebook Style)

```python
from src.model.nn_matching.pipeline.data_prep import (
    normalize, normalize_and_combine, rebuild_combined_text_for_row,
    add_date_int_cols, stratified_group_split_3way,
)
from src.model.nn_matching.pipeline.candidate_generation import (
    get_candidates,
    build_training_episodes_single_df_debug,
    build_training_episodes_parallel,
    build_training_episodes_sequential,
    build_training_episodes_spark,
)
from src.model.nn_matching.pipeline.vectorization import (
    iter_episode_text, vectorize_episode
)
from src.model.nn_matching.models.siamese_network import (
    SiameseMatchingNet, RankingEpisodeDataset, collate_episodes_flat
)
from src.model.nn_matching.models.losses import (
    listwise_ce_from_groups, batch_metrics_from_logits
)
```

### After (Cleaner)

```python
# Option A: Import from subpackages
from src.model.nn_matching.pipeline import (
    normalize, normalize_and_combine, rebuild_combined_text_for_row,
    add_date_int_cols, stratified_group_split_3way,
    get_candidates,
    build_training_episodes_single_df_debug,
    build_training_episodes_parallel,
    build_training_episodes_sequential,
    build_training_episodes_spark,
    iter_episode_text, vectorize_episode,
)
from src.model.nn_matching.models import (
    SiameseMatchingNet, RankingEpisodeDataset, collate_episodes_flat,
    listwise_ce_from_groups, batch_metrics_from_logits,
)

# Option B: Import from top-level package
from src.model.nn_matching import (
    # Data prep
    normalize, normalize_and_combine, add_date_int_cols, stratified_group_split_3way,
    # Candidate generation
    get_candidates, build_training_episodes_parallel,
    # Vectorization
    iter_episode_text, vectorize_episode,
    # Models
    SiameseMatchingNet, RankingEpisodeDataset, collate_episodes_flat,
    # Losses
    listwise_ce_from_groups, batch_metrics_from_logits,
)
```

---

## Benefits

1. **Cleaner code:** Shorter import statements
2. **Better IDE support:** Autocomplete shows available exports
3. **Clearer API boundaries:** `__all__` explicitly defines public interface
4. **Easier refactoring:** Internal module structure can change without breaking imports
5. **More Pythonic:** Follows standard Python package conventions

---

## Backward Compatibility

✅ **Old imports still work!** The existing notebook imports from specific modules (e.g., `from src.model.nn_matching.pipeline.data_prep import ...`) continue to function exactly as before.

You can migrate gradually or keep the current style — both work.

---

## Best Practices

1. **For notebooks:** Use Pattern 1 or 2 (subpackage imports) for cleaner cells
2. **For library code:** Use Pattern 3 (module imports) for maximum clarity
3. **For external users:** Document Pattern 1 as the recommended approach
4. **Update gradually:** No need to refactor existing code immediately

---

## Example: Simplified Notebook Cell

```python
# Before: 20 lines of imports
from src.model.nn_matching.pipeline.data_prep import normalize, normalize_and_combine
from src.model.nn_matching.pipeline.candidate_generation import get_candidates
# ... 15 more lines ...

# After: 6 lines of imports
from src.model.nn_matching.models import SiameseMatchingNet, RankingEpisodeDataset
from src.model.nn_matching.pipeline import (
    normalize, get_candidates, build_training_episodes_parallel, vectorize_episode
)
from torch.utils.data import DataLoader
```
