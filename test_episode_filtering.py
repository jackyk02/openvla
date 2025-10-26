#!/usr/bin/env python3
"""
Test script to verify episode filtering based on scores.
"""
import json
import sys

def analyze_episode_filtering(json_path="/root/openvla/episode_scores_full_results.json"):
    """Analyze which episodes will be kept vs filtered out."""
    
    print(f"Loading episode scores from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    overall_mean_score = data['summary']['overall_mean_score']
    overall_std_score = data['summary']['overall_std_score']
    total_episodes = data['summary']['total_episodes_processed']
    total_samples = data['summary']['total_samples']
    
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Mean Score: {overall_mean_score:.6f}")
    print(f"Overall Std Score: {overall_std_score:.6f}")
    print(f"Overall Median Score: {data['summary']['overall_median_score']:.6f}")
    
    # Analyze filtering
    high_scoring_episodes = []
    low_scoring_episodes = []
    high_scoring_samples = 0
    low_scoring_samples = 0
    
    for episode_id, episode_data in data['episode_scores'].items():
        num_samples = episode_data['num_samples']
        mean_score = episode_data['mean_score']
        
        if mean_score > overall_mean_score:
            high_scoring_episodes.append({
                'id': int(episode_id),
                'mean_score': mean_score,
                'num_samples': num_samples
            })
            high_scoring_samples += num_samples
        else:
            low_scoring_episodes.append({
                'id': int(episode_id),
                'mean_score': mean_score,
                'num_samples': num_samples
            })
            low_scoring_samples += num_samples
    
    print(f"\n{'='*70}")
    print(f"FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Episodes KEPT (score > {overall_mean_score:.6f}):")
    print(f"  Count: {len(high_scoring_episodes)} episodes ({len(high_scoring_episodes)/total_episodes*100:.2f}%)")
    print(f"  Samples: {high_scoring_samples} samples ({high_scoring_samples/total_samples*100:.2f}%)")
    
    print(f"\nEpisodes FILTERED OUT (score <= {overall_mean_score:.6f}):")
    print(f"  Count: {len(low_scoring_episodes)} episodes ({len(low_scoring_episodes)/total_episodes*100:.2f}%)")
    print(f"  Samples: {low_scoring_samples} samples ({low_scoring_samples/total_samples*100:.2f}%)")
    
    # Show score distribution for kept episodes
    kept_scores = [ep['mean_score'] for ep in high_scoring_episodes]
    if kept_scores:
        print(f"\n{'='*70}")
        print(f"KEPT EPISODES - SCORE DISTRIBUTION")
        print(f"{'='*70}")
        print(f"  Mean: {sum(kept_scores)/len(kept_scores):.6f}")
        print(f"  Min: {min(kept_scores):.6f}")
        print(f"  Max: {max(kept_scores):.6f}")
        print(f"  Median: {sorted(kept_scores)[len(kept_scores)//2]:.6f}")
    
    # Show top 10 highest scoring episodes
    high_scoring_episodes.sort(key=lambda x: x['mean_score'], reverse=True)
    print(f"\n{'='*70}")
    print(f"TOP 10 HIGHEST SCORING EPISODES (will be kept)")
    print(f"{'='*70}")
    print(f"{'Episode ID':>12} | {'Mean Score':>12} | {'Num Samples':>12}")
    print(f"{'-'*70}")
    for i, ep in enumerate(high_scoring_episodes[:10]):
        print(f"{ep['id']:>12} | {ep['mean_score']:>12.6f} | {ep['num_samples']:>12}")
    
    # Show bottom 10 lowest scoring episodes
    low_scoring_episodes.sort(key=lambda x: x['mean_score'])
    print(f"\n{'='*70}")
    print(f"TOP 10 LOWEST SCORING EPISODES (will be filtered out)")
    print(f"{'='*70}")
    print(f"{'Episode ID':>12} | {'Mean Score':>12} | {'Num Samples':>12}")
    print(f"{'-'*70}")
    for i, ep in enumerate(low_scoring_episodes[:10]):
        print(f"{ep['id']:>12} | {ep['mean_score']:>12.6f} | {ep['num_samples']:>12}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"By filtering to episodes with mean_score > {overall_mean_score:.6f}:")
    print(f"  - We keep {len(high_scoring_episodes)}/{total_episodes} episodes ({len(high_scoring_episodes)/total_episodes*100:.2f}%)")
    print(f"  - We keep {high_scoring_samples}/{total_samples} samples ({high_scoring_samples/total_samples*100:.2f}%)")
    print(f"  - Average score improves from {overall_mean_score:.6f} to {sum(kept_scores)/len(kept_scores):.6f}")
    print(f"  - This represents a {((sum(kept_scores)/len(kept_scores) - overall_mean_score) / overall_mean_score * 100):.2f}% improvement")
    print(f"{'='*70}\n")
    
    return {
        'total_episodes': total_episodes,
        'kept_episodes': len(high_scoring_episodes),
        'filtered_episodes': len(low_scoring_episodes),
        'kept_samples': high_scoring_samples,
        'filtered_samples': low_scoring_samples,
        'overall_mean': overall_mean_score,
        'kept_mean': sum(kept_scores)/len(kept_scores) if kept_scores else 0
    }

if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "/root/openvla/episode_scores_full_results.json"
    analyze_episode_filtering(json_path)

