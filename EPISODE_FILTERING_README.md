# Episode Score-Based Filtering for RLDS Datasets

## Overview

This feature allows you to filter RLDS datasets to only include episodes with performance scores above the overall mean. This helps improve training data quality by focusing on higher-quality demonstrations.

## How It Works

The filtering is based on the `episode_scores_full_results.json` file which contains:
- **Episode-level scores**: Mean score for each episode computed across all samples
- **Overall mean score**: The average score across all episodes (0.265048 in current dataset)

When enabled, the dataset loader will:
1. Load episode scores from the JSON file
2. Identify episodes with `mean_score > overall_mean_score`
3. Filter the RLDS dataset to only include those high-scoring episodes

## Impact

Based on the current dataset analysis:
- **Episodes kept**: 2,728 out of 5,147 (53.00%)
- **Samples kept**: 93,052 out of 183,376 (50.74%)
- **Score improvement**: Average score increases from 0.265 to 0.312 (+17.88%)

## Usage

### Automatic (Default Behavior)

The filtering is **enabled by default**. When you load a dataset, it will automatically:
```python
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds

dataset, stats = make_dataset_from_rlds(
    name="your_dataset",
    data_dir="/path/to/data",
    train=True,
    # filter_high_scoring_episodes=True,  # This is the default
    # episode_scores_path=None,  # Uses default path: /root/openvla/episode_scores_full_results.json
    ...
)
```

### Disabling Filtering

To disable episode filtering and use all episodes:
```python
dataset, stats = make_dataset_from_rlds(
    name="your_dataset",
    data_dir="/path/to/data",
    train=True,
    filter_high_scoring_episodes=False,  # Disable filtering
    ...
)
```

### Custom Episode Scores Path

To use a different episode scores file:
```python
dataset, stats = make_dataset_from_rlds(
    name="your_dataset",
    data_dir="/path/to/data",
    train=True,
    filter_high_scoring_episodes=True,
    episode_scores_path="/path/to/custom_episode_scores.json",
    ...
)
```

### Manual Initialization

You can also manually initialize the episode filter before loading datasets:
```python
from prismatic.vla.datasets.rlds.dataset import initialize_episode_filter

# Initialize once at the start
initialize_episode_filter("/path/to/episode_scores_full_results.json")

# Then load datasets normally (filtering will be applied)
dataset, stats = make_dataset_from_rlds(...)
```

## Implementation Details

### Episode Index Matching

The filtering works by:
1. Loading episodes **in order** (with `shuffle=False` initially)
2. Assigning each episode an index (0, 1, 2, ...)
3. Matching these indices with episode IDs in the JSON file
4. Filtering out episodes not in the high-scoring set
5. Removing the temporary episode_index field after filtering

**Important**: Episode indices in the RLDS dataset must match the episode IDs in the JSON file. The JSON file should contain scores for episodes in the same order as they appear in the RLDS dataset.

### Score Calculation

Episodes are kept if:
```
episode_data['mean_score'] > overall_mean_score
```

Where `overall_mean_score` is computed as the average score across all samples in all episodes.

## Testing

To analyze the filtering statistics without loading the full dataset:
```bash
python test_episode_filtering.py
```

This will show:
- Overall statistics
- Number of episodes/samples kept vs filtered
- Score distributions
- Top highest and lowest scoring episodes

## Example Output

When filtering is enabled, you'll see log messages like:
```
[INFO] Loaded episode scores: 2728/5147 episodes above mean score 0.2650
[INFO] Using split: val
[INFO] Filtered dataset to keep only high-scoring episodes (score > 0.2650)
```

## File Structure

- `episode_scores_full_results.json`: Contains episode scores and statistics
  - `episode_scores`: Dict mapping episode ID to score data
    - Each episode has: `mean_score`, `scores`, `num_samples`, etc.
  - `summary`: Overall statistics including `overall_mean_score`
  
## Notes

- Filtering happens at the trajectory (episode) level before any frame-level transformations
- The feature is backward compatible - if the JSON file is not found, a warning is logged and no filtering is applied
- Episode indices are temporarily added during filtering and removed afterward to avoid affecting downstream processing
- When filtering is enabled, datasets are initially loaded without shuffling to maintain episode order, but shuffling is still applied later in the pipeline if requested

## Future Improvements

Possible enhancements:
- Support for custom threshold percentiles (e.g., keep top 25%)
- Support for filtering by other metrics (std_score, min_score, etc.)
- Support for sample-level filtering (instead of episode-level)
- Caching of filtered datasets for faster subsequent loads

