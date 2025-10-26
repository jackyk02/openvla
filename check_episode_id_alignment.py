#!/usr/bin/env python3
"""
Diagnostic script to check if RLDS dataset contains episode IDs and verify alignment with scores JSON.
"""
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], "GPU")

def check_episode_id_in_trajectory(builder_dir="/root/bridge_dataset/1.0.0", num_episodes_to_check=10):
    """
    Check if RLDS trajectories contain episode_id field.
    """
    print("="*70)
    print("CHECKING EPISODE ID AVAILABILITY IN RLDS DATASET")
    print("="*70)
    
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    dataset = dl.DLataset.from_rlds(builder, split=f"val[:{num_episodes_to_check}]", shuffle=False)
    
    print(f"\nChecking first {num_episodes_to_check} episodes from validation split...")
    print(f"Builder directory: {builder_dir}")
    
    episode_count = 0
    has_episode_id = False
    
    for i, traj in enumerate(dataset):
        episode_count += 1
        print(f"\n--- Episode {i} ---")
        print(f"Keys in trajectory: {list(traj.keys())}")
        
        # Check for episode_id
        if "episode_id" in traj:
            print(f"✓ Found 'episode_id': {traj['episode_id']}")
            has_episode_id = True
        elif "episode_index" in traj:
            print(f"✓ Found 'episode_index': {traj['episode_index']}")
            has_episode_id = True
        elif "episode_metadata" in traj:
            print(f"✓ Found 'episode_metadata': {traj['episode_metadata']}")
            if isinstance(traj["episode_metadata"], dict) and "episode_id" in traj["episode_metadata"]:
                print(f"  └─ Contains 'episode_id': {traj['episode_metadata']['episode_id']}")
                has_episode_id = True
        else:
            print("✗ No episode_id or episode_index found in trajectory")
            print(f"  Available keys: {list(traj.keys())}")
        
        # Check steps if available
        if "steps" in traj:
            steps = traj["steps"]
            if hasattr(steps, '__iter__'):
                first_step = next(iter(steps))
                print(f"Keys in first step: {list(first_step.keys())}")
        
        if episode_count >= num_episodes_to_check:
            break
    
    print("\n" + "="*70)
    if has_episode_id:
        print("✓ Episode IDs are available in the dataset")
        print("  The filtering should work correctly!")
    else:
        print("✗ Episode IDs are NOT directly available in the dataset")
        print("  This means we need an alternative approach for filtering!")
    print("="*70)
    
    return has_episode_id


def check_alignment_with_scores(scores_json_path="/root/openvla/episode_scores_full_results.json"):
    """
    Check alignment between scores JSON and expected episode structure.
    """
    print("\n" + "="*70)
    print("CHECKING EPISODE SCORES JSON STRUCTURE")
    print("="*70)
    
    with open(scores_json_path, 'r') as f:
        data = json.load(f)
    
    episode_ids = sorted([int(k) for k in data['episode_scores'].keys()])
    
    print(f"\nTotal episodes in JSON: {len(episode_ids)}")
    print(f"Episode ID range: {min(episode_ids)} to {max(episode_ids)}")
    print(f"\nFirst 20 episode IDs: {episode_ids[:20]}")
    
    # Check for gaps
    gaps = []
    for i in range(len(episode_ids) - 1):
        if episode_ids[i+1] - episode_ids[i] > 1:
            missing = list(range(episode_ids[i] + 1, episode_ids[i+1]))
            gaps.extend(missing)
    
    if gaps:
        print(f"\n⚠ Found {len(gaps)} gaps (missing episode IDs)")
        print(f"Missing episodes: {gaps[:50]}" + (" ..." if len(gaps) > 50 else ""))
        print("\nThis means:")
        print("  - Episode IDs are NOT sequential")
        print("  - Some episodes were skipped during scoring (likely due to empty instructions)")
        print("  - We MUST use actual episode IDs, not enumeration indices")
    else:
        print("\n✓ No gaps - episode IDs are sequential")
        print("  - Enumeration indices would work")
    
    return len(gaps) > 0


def suggest_solution(has_rlds_episode_id, has_gaps_in_scores):
    """
    Suggest the appropriate solution based on findings.
    """
    print("\n" + "="*70)
    print("RECOMMENDED SOLUTION")
    print("="*70)
    
    if not has_rlds_episode_id and has_gaps_in_scores:
        print("\n⚠ PROBLEM DETECTED:")
        print("  1. RLDS dataset does NOT provide episode IDs in trajectories")
        print("  2. Score JSON has gaps (some episodes were skipped)")
        print("  3. Cannot reliably match episodes using enumeration")
        
        print("\n💡 SOLUTION OPTIONS:")
        print("\nOption A: Load with explicit episode IDs (RECOMMENDED)")
        print("  - Modify dataset.py to load episodes individually with known IDs")
        print("  - Load only episodes present in the scores JSON")
        print("  - Code example:")
        print("    ```python")
        print("    episode_ids = list(HIGH_SCORING_EPISODES)")
        print("    datasets = []")
        print("    for ep_id in episode_ids:")
        print("        ds = builder.as_dataset(split=f'val[{ep_id}:{ep_id+1}]')")
        print("        datasets.append(ds)")
        print("    dataset = concatenate(datasets)")
        print("    ```")
        
        print("\nOption B: Add episode ID during data creation")
        print("  - Modify augment_bridge_dataset.py to save episode ID mapping")
        print("  - Use this mapping to inject episode IDs when loading")
        
        print("\nOption C: Use sequential loading with ID mapping")
        print("  - Create a mapping: {sequential_index: actual_episode_id}")
        print("  - Load all episodes and filter using the mapping")
        
    elif has_rlds_episode_id:
        print("\n✓ GOOD NEWS:")
        print("  - RLDS dataset provides episode IDs")
        print("  - Current implementation should work correctly")
        
    else:
        print("\n⚠ EDGE CASE:")
        print("  - RLDS doesn't provide episode IDs")
        print("  - But scores JSON has no gaps")
        print("  - Enumeration should work fine")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check episode ID alignment')
    parser.add_argument('--builder_dir', type=str, default='/root/bridge_dataset/1.0.0',
                       help='Path to RLDS dataset builder directory')
    parser.add_argument('--scores_json', type=str, default='/root/openvla/episode_scores_full_results.json',
                       help='Path to episode scores JSON')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to check in RLDS')
    
    args = parser.parse_args()
    
    # Check if RLDS has episode IDs
    has_episode_id = check_episode_id_in_trajectory(args.builder_dir, args.num_episodes)
    
    # Check for gaps in scores JSON
    has_gaps = check_alignment_with_scores(args.scores_json)
    
    # Suggest solution
    suggest_solution(has_episode_id, has_gaps)
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Review the findings above")
    print("2. Run with more episodes: --num_episodes 100")
    print("3. Apply the recommended solution")
    print("="*70)

