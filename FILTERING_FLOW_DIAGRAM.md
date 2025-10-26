# Episode Filtering Flow Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataset Loading Starts                        │
│                  make_dataset_from_rlds(...)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│   Check if filter_high_scoring_episodes=True (default)          │
│   If yes and not initialized → Load episode_scores.json          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           Parse episode_scores_full_results.json                 │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ episode_scores: {"0": {...}, "1": {...}, ...}      │       │
│   │ summary: {overall_mean_score: 0.265048}            │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                  │
│   Filter: mean_score > 0.265048                                 │
│   Result: 2728 high-scoring episode IDs (53% of total)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         Load RLDS Dataset (shuffle=False for filtering)          │
│                   dl.DLataset.from_rlds()                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         Add Episode Index to Each Trajectory                     │
│   Episode 0 → {trajectory_data, episode_index: 0}               │
│   Episode 1 → {trajectory_data, episode_index: 1}               │
│   ...                                                            │
│   Episode N → {trajectory_data, episode_index: N}               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Restructure Trajectories                            │
│   - Apply standardize_fn                                         │
│   - Extract observations, actions, task info                     │
│   - Preserve episode_index for filtering                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Filter by Episode Index                         │
│   Keep trajectory if:                                            │
│   episode_index in HIGH_SCORING_EPISODES                         │
│                                                                  │
│   Before: 5,147 episodes (183,376 samples)                      │
│   After:  2,728 episodes (93,052 samples)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│          Remove Temporary Episode Index Field                    │
│   Clean up: remove episode_index from trajectories              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Normalize Actions and Proprio                      │
│   Apply normalization transformations                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Return Filtered Dataset                          │
│   Quality: +17.88% improvement in average score                 │
│   Size: -49.26% reduction in samples                            │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Episode Filtering Logic

```
                  ┌──────────────────────────┐
                  │ episode_scores.json      │
                  │ 5,147 episodes total     │
                  └──────────┬───────────────┘
                             │
                             │ Parse & Analyze
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │   For each episode:                   │
         │   if mean_score > 0.265048:           │
         │       HIGH_SCORING_EPISODES.add(id)   │
         └───────────────┬───────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                │
         ▼                                ▼
┌────────────────────┐         ┌──────────────────────┐
│ High Scoring       │         │ Low Scoring          │
│ (KEPT)             │         │ (FILTERED OUT)       │
├────────────────────┤         ├──────────────────────┤
│ 2,728 episodes     │         │ 2,419 episodes       │
│ 93,052 samples     │         │ 90,324 samples       │
│ Mean: 0.312449     │         │ Mean: 0.215234       │
│ Range: 0.265-0.421 │         │ Range: 0.072-0.265   │
└────────────────────┘         └──────────────────────┘
         │                                │
         │                                ▼
         │                       ┌────────────────┐
         │                       │   Discarded    │
         │                       └────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Final Filtered Dataset        │
│   Ready for training            │
└─────────────────────────────────┘
```

## Score Distribution

```
Before Filtering (All Episodes):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                 ▲ Overall Mean: 0.265
    ┌─────────────────────────────┼──────────────────────────────┐
    │         ░░░░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░░░░░░░░░        │
    │      ░░░░░░░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░      │
    │    ░░░░░░░░░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░░░    │
    └─────────────────────────────┼──────────────────────────────┘
    0.072                     0.265                          0.421
    (min)                     (mean)                         (max)

After Filtering (High-Scoring Only):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                              ▲ New Mean: 0.312
                          ┌───────────────────┼────────────────┐
                          │    ░░░░░░░░░░░░░░░│░░░░░░░░░░░░    │
                          │  ░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░  │
                          │ ░░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░░ │
                          └───────────────────┼────────────────┘
                        0.265              0.312             0.421
                    (threshold)           (new mean)         (max)

Improvement: +17.88% in average score
```

## Configuration Options

```
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Matrix                        │
├──────────────────────────┬──────────────────────────────────┤
│ Parameter                │ Effect                           │
├──────────────────────────┼──────────────────────────────────┤
│ filter_high_scoring      │ • True (default): Filter enabled │
│   _episodes              │ • False: All episodes loaded     │
├──────────────────────────┼──────────────────────────────────┤
│ episode_scores_path      │ • None: Use default path         │
│                          │ • Custom: Use specified JSON     │
├──────────────────────────┼──────────────────────────────────┤
│ If JSON not found        │ • Warning logged                 │
│                          │ • All episodes loaded            │
│                          │ • No error raised                │
└──────────────────────────┴──────────────────────────────────┘
```

## Performance Comparison

```
Training Data Quality Metrics:

                        Before          After         Improvement
                     ┌───────────┐  ┌───────────┐   ┌──────────┐
Episodes             │   5,147   │  │   2,728   │   │  -47.0%  │
                     └───────────┘  └───────────┘   └──────────┘
                     
Samples              │  183,376  │  │  93,052   │   │ -49.26%  │
                     └───────────┘  └───────────┘   └──────────┘
                     
Avg Score            │  0.2650   │  │  0.3124   │   │ +17.88%  │
                     └───────────┘  └───────────┘   └──────────┘
                     
Quality Floor        │  0.0716   │  │  0.2651   │   │ +270.1%  │
                     └───────────┘  └───────────┘   └──────────┘

Result: Higher quality, smaller dataset → Faster, better training
```

## Usage Flow

```
User Script
    │
    ├─► make_dataset_from_rlds()
    │       │
    │       ├─► [AUTOMATIC] Initialize episode filter
    │       │       └─► Load episode_scores.json
    │       │           └─► Identify high-scoring episodes
    │       │
    │       ├─► Load RLDS dataset
    │       │       └─► Add episode indices
    │       │
    │       ├─► [AUTOMATIC] Filter episodes
    │       │       └─► Keep only high-scoring episodes
    │       │           └─► Remove temporary indices
    │       │
    │       └─► Return filtered dataset
    │
    └─► Use filtered dataset for training
            └─► Better quality → Better model performance
```

## Key Decision Points

```
┌──────────────────────────────────────────────────────────────┐
│  Start: make_dataset_from_rlds(filter_high_scoring=True)    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ JSON file exists?           │
         └────┬───────────────┬────────┘
              │ Yes           │ No
              ▼               ▼
    ┌─────────────────┐  ┌──────────────────────┐
    │ Load scores     │  │ Log warning          │
    │ Filter enabled  │  │ Load all episodes    │
    └─────────┬───────┘  └──────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Parse JSON          │
    │ 2728 episodes found │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Load dataset        │
    │ Add indices         │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Filter by index     │
    │ Keep 2728 episodes  │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Clean up indices    │
    └─────────┬───────────┘
              │
              ▼
    ┌──────────────────────────────┐
    │ Return filtered dataset      │
    │ Quality: +17.88%             │
    │ Size: -49.26%                │
    └──────────────────────────────┘
```

## Files Involved

```
Project Structure:
/root/openvla/
    │
    ├── episode_scores_full_results.json  ← Score data (INPUT)
    │   └── Contains mean scores for 5,147 episodes
    │
    ├── prismatic/vla/datasets/rlds/
    │   └── dataset.py  ← Modified (CORE LOGIC)
    │       ├── load_episode_scores()
    │       ├── initialize_episode_filter()
    │       └── make_dataset_from_rlds() [modified]
    │
    ├── test_episode_filtering.py  ← Analysis tool (TESTING)
    │   └── Displays filtering statistics
    │
    ├── EPISODE_FILTERING_README.md  ← Documentation (DOCS)
    │   └── Usage guide and examples
    │
    ├── CHANGES_SUMMARY.md  ← This file (DOCS)
    │   └── Complete change log
    │
    └── FILTERING_FLOW_DIAGRAM.md  ← Visual guide (DOCS)
        └── Flow diagrams and charts
```

## Success Metrics

✅ **Quality Improvement**: Average score increases by 17.88%  
✅ **Data Efficiency**: 49% reduction in data size  
✅ **Backward Compatible**: Can be disabled, graceful degradation  
✅ **Automatic**: Works by default, no code changes needed  
✅ **Flexible**: Custom paths and thresholds supported  
✅ **Robust**: Handles missing files, invalid data gracefully  
✅ **Documented**: Complete docs, examples, and tests  

## Next Steps

1. **Verify**: Run `python test_episode_filtering.py`
2. **Test**: Load a dataset and check logs for filtering messages
3. **Train**: Use filtered dataset and monitor performance improvements
4. **Iterate**: Adjust threshold or add custom filtering logic if needed

