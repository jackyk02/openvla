#!/usr/bin/env python3
"""
Compute average VLA-CLIP scores for each episode in the DROID dataset.
Uses batch inference for efficiency.
Works with Bridge-scale converted DROID actions.
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# Add vla-clip path for imports
VLA_CLIP_PATH = '/root/vla-clip/bridge_verifier'
VLA_CLIP_ENSEMBLE_PATH = os.path.join(VLA_CLIP_PATH, 'ensemble_eval')

if os.path.exists(VLA_CLIP_ENSEMBLE_PATH):
    sys.path.insert(0, VLA_CLIP_ENSEMBLE_PATH)
elif os.path.exists(VLA_CLIP_PATH):
    sys.path.insert(0, VLA_CLIP_PATH)
else:
    print(f"Error: vla-clip not found at: {VLA_CLIP_PATH}")
    sys.exit(1)

try:
    from efficient_ensemble_merged import EfficientEnsembleMerged
except ImportError as e:
    print("Error: Could not import EfficientEnsembleMerged")
    print(f"Tried paths: {VLA_CLIP_ENSEMBLE_PATH}, {VLA_CLIP_PATH}")
    print(f"Import error: {e}")
    print("\nMake sure vla-clip is properly installed.")
    sys.exit(1)


def group_samples_by_episode(samples):
    """
    Group samples by episode_id
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        dict: {episode_id: [sample1, sample2, ...]}
    """
    episode_to_samples = defaultdict(list)
    for sample in samples:
        episode_id = sample.get('episode_id')
        if episode_id is not None:
            episode_to_samples[episode_id].append(sample)
    return episode_to_samples


def compute_episode_scores(merged_checkpoint_path, droid_dataset_path, images_folder, 
                          output_path=None, max_episodes=None):
    """
    Compute average VLA-CLIP scores for each episode using batch inference
    
    Args:
        merged_checkpoint_path: Path to merged ensemble checkpoint
        droid_dataset_path: Path to DROID dataset JSON (Bridge-scale converted)
        images_folder: Path to folder containing exterior_image files
        output_path: Path to save results JSON (optional)
        max_episodes: Maximum number of episodes to process (for testing)
        
    Returns:
        dict: Episode scores and statistics
    """
    print("="*80)
    print("Computing Average VLA-CLIP Scores per Episode (DROID Dataset)")
    print("="*80)
    
    # Load the ensemble model
    print(f"\nLoading ensemble model from: {merged_checkpoint_path}")
    inference_model = EfficientEnsembleMerged(merged_checkpoint_path)
    
    # Load the dataset
    print(f"\nLoading DROID dataset from: {droid_dataset_path}")
    with open(droid_dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    
    action_histories = dataset_dict['action_histories']
    instructions = dataset_dict['instructions']
    samples = dataset_dict['samples']
    metadata = dataset_dict.get('_metadata', {})
    
    print(f"Dataset contains:")
    print(f"  - {len(action_histories)} unique action histories")
    print(f"  - {len(instructions)} unique instructions")
    print(f"  - {len(samples)} total samples")
    
    # Check if dataset has been converted to Bridge scale
    conversion_info = metadata.get('conversion', {})
    if conversion_info:
        print(f"\n✓ Dataset converted to Bridge scale:")
        print(f"  - Method: {conversion_info.get('method', 'unknown')}")
        print(f"  - Source: {conversion_info.get('source_scale', 'unknown')}")
        print(f"  - Target: {conversion_info.get('target_scale', 'unknown')}")
        print(f"  - Action dim: {metadata.get('action_dim', 'unknown')}")
    else:
        print(f"\n⚠ Warning: Dataset may not be converted to Bridge scale")
        print(f"  Make sure to use droid_bridge_converted.json, not droid_extracted.json")
    
    # Group samples by episode
    print("\nGrouping samples by episode...")
    episode_to_samples = group_samples_by_episode(samples)
    episode_ids = sorted(episode_to_samples.keys())
    
    if max_episodes is not None and max_episodes < len(episode_ids):
        print(f"Limiting to first {max_episodes} episodes (for testing)")
        episode_ids = episode_ids[:max_episodes]
    
    print(f"Processing {len(episode_ids)} episodes")
    
    # Process each episode with batch inference
    episode_results = {}
    failed_episodes = []
    
    for episode_id in tqdm(episode_ids, desc="Processing episodes"):
        episode_samples = episode_to_samples[episode_id]
        
        # Prepare batch data for this episode
        images_list = []
        instructions_list = []
        action_histories_list = []
        valid_sample_indices = []
        
        for idx, sample in enumerate(episode_samples):
            action_history_id = sample.get('action_history_id')
            instruction_id = sample.get('instruction_id')
            
            # DROID uses exterior_image_file instead of agent_view_image_file
            exterior_image_file = sample.get('exterior_image_file')
            
            # Validate sample has all required fields
            if not all([action_history_id, instruction_id, exterior_image_file]):
                continue
            
            # Get action history and instruction
            if action_history_id not in action_histories:
                continue
            if instruction_id not in instructions:
                continue
                
            action_hist = np.array(action_histories[action_history_id])
            instruction = instructions[instruction_id]
            
            # Load image
            image_path = os.path.join(images_folder, exterior_image_file)
            if not os.path.exists(image_path):
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                continue
            
            # Add to batch
            images_list.append(image)
            instructions_list.append(instruction)
            action_histories_list.append(action_hist)
            valid_sample_indices.append(idx)
        
        # Skip episodes with no valid samples
        if len(images_list) == 0:
            failed_episodes.append(episode_id)
            continue
        
        # Compute scores in batch for this episode
        try:
            scores = inference_model.compute_scores_batch(
                images_list, 
                instructions_list, 
                action_histories_list
            )
            
            # Compute statistics
            episode_results[episode_id] = {
                'num_samples': len(scores),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores)),
                'scores': scores.tolist(),  # All individual scores
                'valid_sample_indices': valid_sample_indices,  # Which samples were valid
            }
        except Exception as e:
            print(f"\nError processing episode {episode_id}: {e}")
            failed_episodes.append(episode_id)
            continue
    
    # Compute overall statistics
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    if len(episode_results) == 0:
        print("❌ No episodes processed successfully!")
        return None
    
    all_mean_scores = [result['mean_score'] for result in episode_results.values()]
    all_individual_scores = []
    for result in episode_results.values():
        all_individual_scores.extend(result['scores'])
    
    print(f"\nSuccessfully processed: {len(episode_results)} episodes")
    print(f"Failed episodes: {len(failed_episodes)}")
    
    print(f"\nPer-Episode Statistics (averaging scores within each episode):")
    print(f"  Mean of episode means: {np.mean(all_mean_scores):.4f}")
    print(f"  Std of episode means: {np.std(all_mean_scores):.4f}")
    print(f"  Min episode mean: {np.min(all_mean_scores):.4f}")
    print(f"  Max episode mean: {np.max(all_mean_scores):.4f}")
    print(f"  Median of episode means: {np.median(all_mean_scores):.4f}")
    
    print(f"\nOverall Sample Statistics (all {len(all_individual_scores)} samples):")
    print(f"  Mean score: {np.mean(all_individual_scores):.4f}")
    print(f"  Std score: {np.std(all_individual_scores):.4f}")
    print(f"  Min score: {np.min(all_individual_scores):.4f}")
    print(f"  Max score: {np.max(all_individual_scores):.4f}")
    print(f"  Median score: {np.median(all_individual_scores):.4f}")
    
    # Show top and bottom episodes by mean score
    sorted_episodes = sorted(episode_results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    
    print(f"\nTop 5 Episodes by Mean Score:")
    for episode_id, result in sorted_episodes[:5]:
        print(f"  Episode {episode_id}: {result['mean_score']:.4f} (n={result['num_samples']})")
    
    print(f"\nBottom 5 Episodes by Mean Score:")
    for episode_id, result in sorted_episodes[-5:]:
        print(f"  Episode {episode_id}: {result['mean_score']:.4f} (n={result['num_samples']})")
    
    # Prepare final results dictionary
    final_results = {
        'episode_scores': episode_results,
        'failed_episodes': failed_episodes,
        'summary': {
            'total_episodes_processed': len(episode_results),
            'total_episodes_failed': len(failed_episodes),
            'total_samples': len(all_individual_scores),
            'mean_of_episode_means': float(np.mean(all_mean_scores)),
            'std_of_episode_means': float(np.std(all_mean_scores)),
            'overall_mean_score': float(np.mean(all_individual_scores)),
            'overall_std_score': float(np.std(all_individual_scores)),
            'overall_median_score': float(np.median(all_individual_scores)),
        },
        'metadata': {
            'merged_checkpoint': merged_checkpoint_path,
            'droid_dataset': droid_dataset_path,
            'images_folder': images_folder,
            'dataset_metadata': metadata,
        }
    }
    
    # Save results if output path specified
    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print("✅ Results saved!")
    
    print("\n" + "="*80)
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute average VLA-CLIP scores for each episode in DROID dataset using batch inference'
    )
    
    parser.add_argument('--merged_checkpoint', type=str, required=True,
                       help='Path to merged trainable components checkpoint')
    parser.add_argument('--droid_dataset', type=str, default='droid_bridge_converted.json',
                       help='Path to DROID dataset JSON (Bridge-scale converted, default: droid_bridge_converted.json)')
    parser.add_argument('--images_folder', type=str, default='droid_extracted_images',
                       help='Path to folder containing exterior camera images (default: droid_extracted_images)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON (optional)')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum number of episodes to process (for testing)')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.merged_checkpoint):
        print(f"❌ Error: Merged checkpoint not found: {args.merged_checkpoint}")
        print(f"\nPlease provide path to VLA-CLIP ensemble checkpoint")
        exit(1)
    if not os.path.exists(args.droid_dataset):
        print(f"❌ Error: Dataset not found: {args.droid_dataset}")
        print(f"\nMake sure you've converted DROID to Bridge scale:")
        print(f"  ./convert_droid_example.sh")
        exit(1)
    if not os.path.exists(args.images_folder):
        print(f"❌ Error: Images folder not found: {args.images_folder}")
        print(f"\nMake sure you've extracted DROID dataset:")
        print(f"  ./extract_droid_example.sh")
        exit(1)
    
    # Set default output path if not specified
    if args.output is None:
        dataset_name = os.path.splitext(os.path.basename(args.droid_dataset))[0]
        args.output = f"episode_scores_{dataset_name}.json"
        print(f"Output path not specified, using: {args.output}")
    
    # Run computation
    compute_episode_scores(
        merged_checkpoint_path=args.merged_checkpoint,
        droid_dataset_path=args.droid_dataset,
        images_folder=args.images_folder,
        output_path=args.output,
        max_episodes=args.max_episodes
    )

