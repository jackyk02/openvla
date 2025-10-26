# Episode ID Alignment Fix

## Problem Identified

The user correctly identified a critical alignment issue between the episode scores JSON and the RLDS dataset loading.

### The Issue

**In `augment_bridge_dataset.py` (score calculation):**
```python
for episode_id in episode_ids:
    ds = builder.as_dataset(split=f"val[{episode_id}:{episode_id + 1}]")
    
    # Skip episodes with empty instructions
    if original_instruction is None:
        print(f"Warning: Empty instruction in episode {episode_id}, skipping...")
        continue
    
    # Store with original episode_id
    episode_data['episode_id'] = episode_id
```

This means:
- Episode IDs in the JSON are the **original RLDS episode IDs**
- Some episodes (e.g., episode 2, 5, 8, etc.) were **skipped** due to empty instructions
- The JSON has episodes: 0, 1, 3, 4, 6, 7, 9, ... (with gaps)

**Original implementation (INCORRECT):**
```python
# Used enumeration to assign sequential indices
dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False)
dataset = dataset.enumerate().traj_map(add_episode_index, ...)
# This creates: 0, 1, 2, 3, 4, 5, 6, ...
# But JSON has:   0, 1, 3, 4, 6, 7, 9, ...
# MISMATCH! ❌
```

**Example of the mismatch:**
| RLDS Load Order | Enum Index | Actual Episode ID | JSON Has Score For |
|----------------|------------|-------------------|-------------------|
| 1st episode | 0 | 0 | 0 ✓ |
| 2nd episode | 1 | 1 | 1 ✓ |  
| 3rd episode | 2 | 2 | **NONE** (skipped) |
| 4th episode | 3 | 3 | 3 ✓ |
| 5th episode | 4 | 4 | 4 ✓ |

If we used enumeration:
- Enum index 2 would try to match JSON episode "2" (which doesn't exist)
- Enum index 3 would try to match JSON episode "3" (but it's actually episode 4 in RLDS)
- **Everything after the first gap is misaligned!**

## Solution Implemented

### Approach: Explicit Episode Loading

Instead of loading all episodes and trying to filter, we **directly load only the episodes we want** using their original RLDS episode IDs:

```python
if filter_high_scoring_episodes and HIGH_SCORING_EPISODES is not None:
    # HIGH_SCORING_EPISODES = {0, 1, 3, 4, 6, 7, 9, ...}  (from JSON)
    
    episode_ids_to_load = sorted(list(HIGH_SCORING_EPISODES))
    episode_datasets = []
    
    for ep_id in episode_ids_to_load:
        # Load THIS SPECIFIC episode by its ID
        ep_dataset = dl.DLataset.from_rlds(
            builder, 
            split=f"{split}[{ep_id}:{ep_id+1}]",  # e.g., "val[3:4]" for episode 3
            shuffle=False,
            num_parallel_reads=1
        )
        episode_datasets.append(ep_dataset)
    
    # Concatenate all loaded episodes
    dataset = concatenate(episode_datasets)
```

### Why This Works

1. **Direct Addressing**: We use `split=f"{split}[{ep_id}:{ep_id+1}]"` which loads episode by its actual RLDS index
2. **No Enumeration**: We don't rely on sequential ordering
3. **Perfect Alignment**: We load episode 0, 1, 3, 4, 6, 7, 9, ... exactly matching the JSON keys
4. **Handles Gaps**: Missing episodes (2, 5, 8, ...) are simply not loaded

### Verification

To verify the alignment is correct, run:
```bash
python check_episode_id_alignment.py
```

This will:
1. Check if RLDS trajectories contain episode IDs (they usually don't)
2. Check for gaps in the scores JSON (there are gaps)
3. Confirm that explicit loading is the correct solution

## Implementation Details

### Key Changes

**File**: `/root/openvla/prismatic/vla/datasets/rlds/dataset.py`

1. **Explicit Loading** (lines 304-346):
   - Load each high-scoring episode individually
   - Use actual episode IDs from JSON
   - Concatenate into single dataset

2. **No Filtering Needed**:
   - Since we load only high-scoring episodes, no post-load filtering required
   - Simpler and more efficient

3. **Shuffling**:
   - Applied after concatenation if requested
   - Uses appropriate buffer size for the number of episodes

### Example Flow

```
JSON Episode Scores: {0: 0.31, 1: 0.28, 3: 0.35, 4: 0.22, 6: 0.38, ...}
                           ↓
Overall Mean: 0.265
                           ↓
High-Scoring: {0, 1, 3, 6, ...}  (scores > 0.265)
                           ↓
Load from RLDS:
  - Load val[0:1] → Episode 0
  - Load val[1:2] → Episode 1
  - Load val[3:4] → Episode 3  (skip 2 - it has empty instruction)
  - Load val[6:7] → Episode 6  (skip 4, 5 - they're low-scoring or empty)
  - ...
                           ↓
Concatenate → Filtered Dataset ✓
```

## Benefits of This Approach

1. **Correct Alignment**: Episodes match their JSON scores exactly
2. **Efficient**: Only loads needed episodes, skips low-scoring ones
3. **Robust**: Works even when episodes are missing/skipped
4. **Simple**: No complex index tracking or filtering logic
5. **Explicit**: Clear which episodes are being loaded

## Edge Cases Handled

1. **Missing Episodes**: If episode IDs have gaps (they do), we skip them automatically
2. **Empty Instructions**: Already filtered out in JSON, won't be loaded
3. **Out of Range**: If JSON contains invalid episode IDs, RLDS will raise an error
4. **Empty High-Scoring Set**: Raises clear error message

## Testing

### Manual Test
```python
from prismatic.vla.datasets.rlds.dataset import initialize_episode_filter, make_dataset_from_rlds

# Initialize filtering
initialize_episode_filter("/root/openvla/episode_scores_full_results.json")

# Load dataset with filtering
dataset, stats = make_dataset_from_rlds(
    name="bridge_dataset",
    data_dir="/path/to/data",
    train=False,  # Use validation split
    filter_high_scoring_episodes=True,
)

# Should see log message:
# [INFO] Loading 2728 high-scoring episodes explicitly...
# [INFO] Successfully loaded 2728 high-scoring episodes
# [INFO] Loaded dataset with 2728 high-scoring episodes (score > 0.2650)
```

### Diagnostic Test
```bash
python check_episode_id_alignment.py --num_episodes 100
```

## Comparison: Old vs New

| Aspect | Old (Enumeration) | New (Explicit Loading) |
|--------|------------------|----------------------|
| **Alignment** | ❌ Wrong after first gap | ✅ Perfect alignment |
| **Handles gaps** | ❌ Misaligns | ✅ Skips naturally |
| **Efficiency** | Load all, filter later | ✅ Load only needed |
| **Complexity** | Index tracking required | ✅ Simple and direct |
| **Correctness** | ❌ **CRITICAL BUG** | ✅ Correct |

## Conclusion

The explicit loading approach ensures that:
- Episode 0 from RLDS matches episode "0" in JSON ✓
- Episode 1 from RLDS matches episode "1" in JSON ✓
- Episode 3 from RLDS matches episode "3" in JSON ✓ (2 is skipped)
- Episode 6 from RLDS matches episode "6" in JSON ✓ (4, 5 are skipped)

**No enumeration, no index mapping, no room for error.** We load exactly the episodes identified in the JSON by their actual IDs.

## Thank You!

Thank you for catching this critical alignment issue. The explicit loading approach is more robust and correct than the original enumeration-based approach.

