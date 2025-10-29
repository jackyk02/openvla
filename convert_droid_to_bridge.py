#!/usr/bin/env python3
"""
Convert DROID actions to Bridge scale actions using token-based conversion.

This script:
1. Loads the extracted DROID dataset (droid_extracted.json)
2. Converts all DROID actions to Bridge scale via tokens
3. Adds gripper dimension (set to 0.0 as placeholder)
4. Saves the converted dataset with metadata
"""

import json
import numpy as np
from tqdm import tqdm
import argparse

# DROID & BRIDGE normalization constants (from token2action.py)
DROID_Q01 = np.array([
    -0.7776297926902771,
    -0.5803514122962952,
    -0.5795090794563293,
    -0.6464047729969025,
    -0.7041108310222626,
    -0.8895104378461838,
])

DROID_Q99 = np.array([
    0.7597932070493698,
    0.5726242214441299,
    0.7351000607013702,
    0.6705610305070877,
    0.6464948207139969,
    0.8897542208433151,
])

BRIDGE_Q01 = np.array([
    -0.02872725307941437,
    -0.04170349963009357,
    -0.026093858778476715,
    -0.08092105075716972,
    -0.09288699507713317,
    -0.20718276381492615,
    0.0
])

BRIDGE_Q99 = np.array([
    0.028309678435325586,
    0.040855254605412394,
    0.040161586627364146,
    0.08192047759890528,
    0.07792850524187081,
    0.20382574498653397,
    1.0
])

# Padding value used in the dataset
ACTION_PADDING_VALUE = -5.0


class TokenActionConverter:
    """Convert actions to/from token representation for cross-dataset normalization."""
    
    def __init__(self, q01, q99, n_action_bins=256, vocab_size=32000):
        """
        Initialize converter with normalization statistics.
        
        Args:
            q01: 1st percentile values for each action dimension
            q99: 99th percentile values for each action dimension
            n_action_bins: Number of discretization bins (default: 256)
            vocab_size: Vocabulary size for token mapping (default: 32000)
        """
        self.q01 = np.array(q01)
        self.q99 = np.array(q99)
        self.vocab_size = vocab_size
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def action_to_token(self, actions):
        """
        Convert actions to tokens.
        
        Normalization: norm = 2 * (action - q01) / (q99 - q01) - 1
        Discretization: Find nearest bin center
        Token mapping: vocab_size - bin_index - 1
        """
        norm = 2 * (actions - self.q01) / (self.q99 - self.q01) - 1
        disc = np.array([np.abs(self.bin_centers - v).argmin() for v in norm])
        return self.vocab_size - disc - 1

    def token_to_action(self, tokens):
        """
        Convert tokens back to actions.
        
        Token unmapping: bin_index = vocab_size - token - 1
        Denormalization: action = 0.5 * (norm + 1) * (q99 - q01) + q01
        """
        disc = self.vocab_size - np.array(tokens)
        disc = np.clip(disc - 1, 0, len(self.bin_centers) - 1)
        norm = self.bin_centers[disc]
        return 0.5 * (norm + 1) * (self.q99 - self.q01) + self.q01


def convert_action_history(droid_actions, droid_converter, bridge_converter, 
                          preserve_padding=True):
    """
    Convert a DROID action history to Bridge scale.
    
    Args:
        droid_actions: Array of shape [T, 7] with DROID actions (6D cartesian + 1D gripper)
        droid_converter: TokenActionConverter for DROID (6D)
        bridge_converter: TokenActionConverter for Bridge (6D)
        preserve_padding: If True, keep padding values as ACTION_PADDING_VALUE
        
    Returns:
        Array of shape [T, 7] with Bridge-scale actions (6D + binarized gripper)
    """
    droid_actions = np.array(droid_actions)
    T, action_dim = droid_actions.shape
    
    # Check if this is all padding
    if preserve_padding and np.all(droid_actions == ACTION_PADDING_VALUE):
        # Return padded 7D actions
        return np.full((T, 7), ACTION_PADDING_VALUE)
    
    bridge_actions = []
    
    for t in range(T):
        action = droid_actions[t]
        
        # Check if this timestep is padding
        if preserve_padding and np.all(action == ACTION_PADDING_VALUE):
            bridge_actions.append(np.full(7, ACTION_PADDING_VALUE))
            continue
        
        # Extract DROID cartesian velocity (first 6 dims) and gripper (last dim)
        droid_cartesian = action[:6] if action_dim >= 6 else action
        droid_gripper = action[6] if action_dim >= 7 else 0.0
        
        # Step 1: DROID cartesian → tokens
        tokens = droid_converter.action_to_token(droid_cartesian)
        
        # Step 2: tokens → Bridge cartesian actions (6D)
        bridge_action_6d = bridge_converter.token_to_action(tokens)
        
        # Step 3: Binarize gripper based on 0.5 threshold
        # < 0.5 → 0.0 (open), >= 0.5 → 1.0 (closed)
        bridge_gripper = 1.0 if droid_gripper >= 0.5 else 0.0
        
        # Step 4: Combine Bridge cartesian + binarized gripper
        bridge_action_7d = np.append(bridge_action_6d, bridge_gripper)
        
        bridge_actions.append(bridge_action_7d)
    
    return np.array(bridge_actions)


def convert_droid_dataset(input_path, output_path):
    """
    Convert entire DROID dataset to Bridge scale.
    
    The gripper dimension is binarized based on 0.5 threshold:
    - gripper < 0.5 → 0.0 (open)
    - gripper >= 0.5 → 1.0 (closed)
    
    Args:
        input_path: Path to input JSON file (DROID scale)
        output_path: Path to output JSON file (Bridge scale)
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize converters
    print("Initializing converters...")
    droid_converter = TokenActionConverter(DROID_Q01, DROID_Q99)
    bridge_converter = TokenActionConverter(BRIDGE_Q01[:6], BRIDGE_Q99[:6])
    
    # Get action histories
    action_histories = dataset['action_histories']
    print(f"Converting {len(action_histories)} action histories...")
    
    # Convert each action history
    converted_histories = {}
    stats = {
        'total': len(action_histories),
        'converted': 0,
        'all_padding': 0,
        'min_values': None,
        'max_values': None,
    }
    
    all_non_padding_actions = []
    
    for action_id, actions in tqdm(action_histories.items(), desc="Converting actions"):
        # Convert to numpy array
        droid_actions = np.array(actions)
        
        # Convert to Bridge scale (with automatic gripper binarization)
        bridge_actions = convert_action_history(
            droid_actions, 
            droid_converter, 
            bridge_converter,
            preserve_padding=True
        )
        
        # Store converted actions
        converted_histories[action_id] = bridge_actions.tolist()
        
        # Collect statistics
        if np.all(bridge_actions == ACTION_PADDING_VALUE):
            stats['all_padding'] += 1
        else:
            stats['converted'] += 1
            # Collect non-padding actions for statistics
            non_padding_mask = ~(bridge_actions == ACTION_PADDING_VALUE).all(axis=1)
            if non_padding_mask.any():
                all_non_padding_actions.append(bridge_actions[non_padding_mask])
    
    # Calculate statistics on converted actions
    if all_non_padding_actions:
        all_actions = np.vstack(all_non_padding_actions)
        stats['min_values'] = all_actions.min(axis=0).tolist()
        stats['max_values'] = all_actions.max(axis=0).tolist()
        stats['mean_values'] = all_actions.mean(axis=0).tolist()
        stats['std_values'] = all_actions.std(axis=0).tolist()
    
    print(f"\nConversion statistics:")
    print(f"  Total action histories: {stats['total']}")
    print(f"  Successfully converted: {stats['converted']}")
    print(f"  All-padding histories: {stats['all_padding']}")
    
    if stats['min_values'] is not None:
        print(f"\nConverted action statistics (7D: 6D position + gripper):")
        print(f"  Min values: {[f'{v:.6f}' for v in stats['min_values']]}")
        print(f"  Max values: {[f'{v:.6f}' for v in stats['max_values']]}")
        print(f"  Mean values: {[f'{v:.6f}' for v in stats['mean_values']]}")
        print(f"  Std values: {[f'{v:.6f}' for v in stats['std_values']]}")
    
    # Update dataset
    dataset['action_histories'] = converted_histories
    
    # Update metadata
    original_metadata = dataset.get('_metadata', {})
    dataset['_metadata'] = {
        **original_metadata,
        'action_dim': 7,  # 6D cartesian + 1D gripper
        'conversion': {
            'method': 'token_based',
            'source_scale': 'droid',
            'target_scale': 'bridge',
            'gripper_binarization': 'threshold_0.5',
            'gripper_note': 'Binarized from DROID gripper: <0.5→0.0 (open), ≥0.5→1.0 (closed)',
            'droid_q01': DROID_Q01.tolist(),
            'droid_q99': DROID_Q99.tolist(),
            'bridge_q01': BRIDGE_Q01.tolist(),
            'bridge_q99': BRIDGE_Q99.tolist(),
            'n_action_bins': 256,
            'vocab_size': 32000,
        },
        'action_statistics': stats,
    }
    
    # Save converted dataset
    print(f"\nSaving converted dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done!")
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION: First Action History")
    print("="*80)
    first_action_id = list(converted_histories.keys())[0]
    first_actions = np.array(converted_histories[first_action_id])
    
    print(f"\nAction history ID: {first_action_id}")
    print(f"Shape: {first_actions.shape}")
    print(f"Expected: [window_size, 7] where 7 = 6D position + 1D gripper\n")
    
    # Show first few actions
    for t in range(min(5, len(first_actions))):
        action = first_actions[t]
        is_padding = np.all(action == ACTION_PADDING_VALUE)
        status = "[PADDING]" if is_padding else ""
        print(f"t={t}: [{', '.join([f'{a:8.5f}' for a in action])}] {status}")
    
    if len(first_actions) > 5:
        print("...")
        for t in range(max(5, len(first_actions) - 2), len(first_actions)):
            action = first_actions[t]
            is_padding = np.all(action == ACTION_PADDING_VALUE)
            status = "[PADDING]" if is_padding else ""
            print(f"t={t}: [{', '.join([f'{a:8.5f}' for a in action])}] {status}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert DROID actions to Bridge scale using token-based conversion. '
                    'Gripper values are binarized: <0.5→0.0 (open), ≥0.5→1.0 (closed)'
    )
    parser.add_argument('--input', type=str, default='droid_extracted.json',
                        help='Input JSON file with DROID-scale actions (7D: 6D cartesian + 1D gripper)')
    parser.add_argument('--output', type=str, default='droid_bridge_converted.json',
                        help='Output JSON file with Bridge-scale actions (7D: 6D cartesian + 1D binarized gripper)')
    
    args = parser.parse_args()
    
    convert_droid_dataset(
        input_path=args.input,
        output_path=args.output
    )

