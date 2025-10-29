# VLA-CLIP Scoring for DROID Dataset

This guide explains how to compute VLA-CLIP scores for the DROID dataset episodes to evaluate action quality.

## Overview

The `compute_droid_episode_scores.py` script computes average VLA-CLIP scores for each episode in the DROID dataset. This helps identify:
- High-quality episodes (higher scores)
- Low-quality episodes (lower scores)
- Episodes suitable for fine-tuning

## Prerequisites

### 1. Extracted DROID Dataset
You need the DROID dataset extracted with images:
```bash
./extract_droid_example.sh
```

This creates:
- `droid_extracted.json` - Dataset with DROID-scale actions
- `droid_extracted_images/` - Folder with exterior camera images (256x256)

### 2. Bridge-Scale Converted Actions
Convert DROID actions to Bridge scale for VLA-CLIP compatibility:
```bash
./convert_droid_example.sh
```

This creates:
- `droid_bridge_converted.json` - Dataset with Bridge-scale actions

### 3. VLA-CLIP Ensemble Checkpoint
You need a trained VLA-CLIP ensemble checkpoint (merged_checkpoint.pt).

If you don't have one, train using the vla-clip framework or obtain a pre-trained checkpoint.

## Usage

### Quick Start

1. **Edit the example script** to set your checkpoint path:
```bash
nano compute_droid_scores_example.sh
# Update MERGED_CHECKPOINT="/path/to/merged_checkpoint.pt"
```

2. **Run the script**:
```bash
./compute_droid_scores_example.sh
```

### Manual Usage

```bash
python compute_droid_episode_scores.py \
  --merged_checkpoint /path/to/merged_checkpoint.pt \
  --droid_dataset droid_bridge_converted.json \
  --images_folder droid_extracted_images \
  --output droid_episode_scores.json
```

### Testing with Limited Episodes

Process only first 10 episodes for testing:
```bash
python compute_droid_episode_scores.py \
  --merged_checkpoint /path/to/merged_checkpoint.pt \
  --droid_dataset droid_bridge_converted.json \
  --images_folder droid_extracted_images \
  --output droid_test_scores.json \
  --max_episodes 10
```

## Arguments

- `--merged_checkpoint` **(required)**: Path to VLA-CLIP ensemble checkpoint
- `--droid_dataset`: Path to DROID dataset JSON with Bridge-scale actions (default: `droid_bridge_converted.json`)
- `--images_folder`: Path to folder with exterior camera images (default: `droid_extracted_images`)
- `--output`: Output JSON file path (default: auto-generated based on dataset name)
- `--max_episodes`: Limit number of episodes to process (for testing)

## Output Format

The script produces a JSON file with the following structure:

```json
{
  "episode_scores": {
    "0": {
      "num_samples": 45,
      "mean_score": 0.8234,
      "std_score": 0.0456,
      "min_score": 0.7123,
      "max_score": 0.9012,
      "median_score": 0.8301,
      "scores": [0.8234, 0.8456, ...],
      "valid_sample_indices": [0, 1, 2, ...]
    },
    "1": {...},
    ...
  },
  "failed_episodes": [5, 12, ...],
  "summary": {
    "total_episodes_processed": 95,
    "total_episodes_failed": 5,
    "total_samples": 4567,
    "mean_of_episode_means": 0.7891,
    "std_of_episode_means": 0.0623,
    "overall_mean_score": 0.7923,
    "overall_std_score": 0.0789,
    "overall_median_score": 0.8012
  },
  "metadata": {
    "merged_checkpoint": "...",
    "droid_dataset": "...",
    "images_folder": "...",
    "dataset_metadata": {...}
  }
}
```

### Fields Explained

**Per-Episode (`episode_scores`):**
- `num_samples`: Number of valid samples in the episode
- `mean_score`: Average VLA-CLIP score across all samples in the episode
- `std_score`: Standard deviation of scores
- `min_score`, `max_score`: Range of scores
- `median_score`: Median score
- `scores`: List of all individual sample scores
- `valid_sample_indices`: Which samples were successfully processed

**Summary Statistics:**
- `mean_of_episode_means`: Average of all episode mean scores
- `overall_mean_score`: Average across all individual samples
- `total_episodes_processed`: Number of successfully processed episodes
- `total_episodes_failed`: Number of episodes that failed processing

## Console Output

The script prints detailed statistics during execution:

```
================================================================================
Computing Average VLA-CLIP Scores per Episode (DROID Dataset)
================================================================================

Loading ensemble model from: /path/to/checkpoint.pt
Loading DROID dataset from: droid_bridge_converted.json

Dataset contains:
  - 2345 unique action histories
  - 150 unique instructions
  - 12567 total samples

✓ Dataset converted to Bridge scale:
  - Method: token_based
  - Source: droid
  - Target: bridge
  - Action dim: 7

Processing 100 episodes
Processing episodes: 100%|████████████████| 100/100

================================================================================
Results Summary
================================================================================

Successfully processed: 95 episodes
Failed episodes: 5

Per-Episode Statistics (averaging scores within each episode):
  Mean of episode means: 0.7891
  Std of episode means: 0.0623
  Min episode mean: 0.5234
  Max episode mean: 0.9123
  Median of episode means: 0.8012

Overall Sample Statistics (all 4567 samples):
  Mean score: 0.7923
  Std score: 0.0789
  Min score: 0.3456
  Max score: 0.9876
  Median score: 0.8012

Top 5 Episodes by Mean Score:
  Episode 42: 0.9123 (n=56)
  Episode 23: 0.9001 (n=48)
  Episode 67: 0.8923 (n=52)
  Episode 15: 0.8834 (n=45)
  Episode 89: 0.8756 (n=51)

Bottom 5 Episodes by Mean Score:
  Episode 5: 0.5234 (n=32)
  Episode 12: 0.5678 (n=28)
  Episode 78: 0.6012 (n=41)
  Episode 91: 0.6234 (n=39)
  Episode 34: 0.6456 (n=44)
```

## Use Cases

### 1. Episode Quality Filtering

Filter high-quality episodes for fine-tuning:

```python
import json

with open('droid_episode_scores.json', 'r') as f:
    results = json.load(f)

# Get episodes with mean score > 0.8
high_quality_episodes = [
    int(episode_id)
    for episode_id, scores in results['episode_scores'].items()
    if scores['mean_score'] > 0.8
]

print(f"High-quality episodes: {high_quality_episodes}")
```

### 2. Sample Filtering

Filter individual samples by score:

```python
# Get all samples with score > 0.85
high_quality_samples = []
for episode_id, scores in results['episode_scores'].items():
    for idx, score in enumerate(scores['scores']):
        if score > 0.85:
            sample_idx = scores['valid_sample_indices'][idx]
            high_quality_samples.append({
                'episode_id': int(episode_id),
                'sample_idx': sample_idx,
                'score': score
            })
```

### 3. Dataset Analysis

Analyze score distribution:

```python
import numpy as np
import matplotlib.pyplot as plt

all_scores = []
for scores in results['episode_scores'].values():
    all_scores.extend(scores['scores'])

plt.hist(all_scores, bins=50)
plt.xlabel('VLA-CLIP Score')
plt.ylabel('Count')
plt.title('Score Distribution Across DROID Dataset')
plt.savefig('droid_score_distribution.png')
```

## Key Differences from Bridge Version

The DROID version differs from the Bridge version in:

1. **Image Key**: Uses `exterior_image_file` instead of `agent_view_image_file`
2. **Camera**: Uses exterior camera (exterior_image_1_left) instead of agent view
3. **Actions**: Expects Bridge-scale converted actions (7D: 6D + binary gripper)
4. **Conversion Check**: Validates that dataset has been converted to Bridge scale

## Troubleshooting

### Issue: "Dataset not found"

**Solution**: Make sure you've converted to Bridge scale:
```bash
./convert_droid_example.sh
```

### Issue: "Images folder not found"

**Solution**: Extract the dataset first:
```bash
./extract_droid_example.sh
```

### Issue: "Could not import EfficientEnsembleMerged"

**Solution**: Make sure vla-clip is installed at `/root/vla-clip/bridge_verifier/`

### Issue: "Checkpoint not found"

**Solution**: Provide valid path to merged VLA-CLIP checkpoint:
```bash
python compute_droid_episode_scores.py \
  --merged_checkpoint /correct/path/to/merged_checkpoint.pt \
  ...
```

### Issue: Low processing speed

**Solution**: The script uses batch inference for efficiency. If still slow:
- Use `--max_episodes` for testing
- Ensure GPU is available (check with `nvidia-smi`)
- Check that checkpoint is on GPU

### Issue: All episodes failed

**Check:**
1. Dataset has correct structure with `exterior_image_file` keys
2. Images exist in the specified folder
3. Actions are in Bridge scale (check metadata)
4. Checkpoint is compatible with action format

## Performance

- **Batch Size**: Automatically determined by ensemble model
- **Speed**: ~10-50 episodes/minute depending on episode length and GPU
- **Memory**: Requires ~4-8GB GPU memory for inference

## Integration with Training Pipeline

After computing scores, you can:

1. **Filter dataset** by episode or sample scores
2. **Create subset** for fine-tuning:
```python
# Filter dataset based on scores
filtered_dataset = {
    'action_histories': original_dataset['action_histories'],
    'instructions': original_dataset['instructions'],
    'samples': [
        s for s in original_dataset['samples']
        if s['episode_id'] in high_quality_episodes
    ],
    '_metadata': original_dataset['_metadata']
}
```

3. **Weight samples** by score during training
4. **Monitor** score distribution changes during training

## Files

- `compute_droid_episode_scores.py` - Main scoring script
- `compute_droid_scores_example.sh` - Quick-start example script
- `DROID_VLA_CLIP_SCORING_README.md` - This documentation

## Citations

If you use VLA-CLIP scoring, please cite:

```bibtex
@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

