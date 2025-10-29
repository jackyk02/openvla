import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import re
import hashlib
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
import warnings

# Define padding value consistently
ACTION_PADDING_VALUE = -5.0

def normalize_instruction(instr):
    """
    Normalize instruction by removing trailing punctuation and converting to lowercase.
    Returns None if instruction is empty after normalization.
    """
    if not instr or not isinstance(instr, str):
        return None
    
    instr = instr.strip().lower()
    # Remove any trailing punctuation (., !, ?)
    instr = re.sub(r'[.?!]+$', '', instr).strip()
    
    # Return None if instruction is empty after normalization
    if not instr:
        return None
    
    return instr

def hash_action_history(action_hist):
    """
    Create a hash for an action history sequence for deduplication
    """
    # Convert to bytes for hashing
    action_bytes = np.array(action_hist).tobytes()
    return hashlib.md5(action_bytes).hexdigest()

def get_or_create_action_history_id(action_hist, action_histories, action_history_hash_to_id, next_action_id):
    """
    Get existing action history ID or create new one if not exists
    """
    action_hash = hash_action_history(action_hist)
    
    if action_hash in action_history_hash_to_id:
        return action_history_hash_to_id[action_hash], next_action_id
    else:
        # Create new entry
        action_id = f"action_{next_action_id}"
        action_histories[action_id] = action_hist.tolist()
        action_history_hash_to_id[action_hash] = action_id
        return action_id, next_action_id + 1

def get_or_create_instruction_id(instruction, instructions, instruction_to_id, next_instruction_id):
    """
    Get existing instruction ID or create new one if not exists.
    Returns None for instruction_id if instruction is empty or None.
    """
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        return None, next_instruction_id
        
    if instruction in instruction_to_id:
        return instruction_to_id[instruction], next_instruction_id
    else:
        # Create new entry
        instruction_id = f"instr_{next_instruction_id}"
        instructions[instruction_id] = instruction
        instruction_to_id[instruction] = instruction_id
        return instruction_id, next_instruction_id + 1

def load_instruction_rephrases(json_path):
    """
    Load instruction rephrases from JSON file
    
    Args:
        json_path (str): Path to the instruction mapping JSON file
        
    Returns:
        dict: Dictionary mapping original instructions to their rephrases
    """
    rephrases_dict = {}
    
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    
    # The JSON file is structured as a dictionary with numbered keys
    # Each entry has "original" and "rephrases" fields
    if isinstance(all_rephrases, dict):
        for key, entry in all_rephrases.items():
            if isinstance(entry, dict) and "original" in entry and "rephrases" in entry:
                orig = normalize_instruction(entry["original"])
                
                # Skip entries with empty original instructions
                if orig is None:
                    print(f"Warning: Empty original instruction in rephrases entry {key}, skipping...")
                    continue
                
                # Filter out empty rephrases
                rephs = []
                for r in entry["rephrases"]:
                    if r and isinstance(r, str) and r.strip():
                        rephs.append(r.strip())
                
                if rephs:  # Only add if there are valid rephrases
                    rephrases_dict[orig] = rephs
                else:
                    print(f"Warning: No valid rephrases found for instruction '{orig}', skipping...")
                    
    elif isinstance(all_rephrases, list):
        # Fallback for list format
        for entry in all_rephrases:
            if "original" in entry and "rephrases" in entry:
                orig = normalize_instruction(entry["original"])
                
                # Skip entries with empty original instructions
                if orig is None:
                    print(f"Warning: Empty original instruction in rephrases list entry, skipping...")
                    continue
                
                # Filter out empty rephrases
                rephs = []
                for r in entry["rephrases"]:
                    if r and isinstance(r, str) and r.strip():
                        rephs.append(r.strip())
                
                if rephs:  # Only add if there are valid rephrases
                    rephrases_dict[orig] = rephs
                else:
                    print(f"Warning: No valid rephrases found for instruction '{orig}', skipping...")
    
    return rephrases_dict

def extract_droid_dataset(builder_dir, episode_ids, output_path, history_length=10, 
                         rephrases_json_path=None, max_episodes=None, images_folder=None, 
                         window_before=6, window_after=3, use_exterior_camera=True, image_size=(256, 256)):
    """
    Extract DROID dataset with action history, exterior_image_1_left camera images, and language instructions.
    Uses a centered window around each timestep (default: t-6 to t+3, with t as current).
    Window includes: [t-6, t-5, t-4, t-3, t-2, t-1, t, t+1, t+2, t+3] = 10 timesteps total.
    Padding with ACTION_PADDING_VALUE (-5.0) when needed.
    All images are resized to the specified image_size (default: 256x256).
    
    Args:
        builder_dir (str): Path to the DROID dataset directory
        episode_ids (list): List of episode IDs to process
        output_path (str): Path to save the extracted dataset (JSON file)
        history_length (int): DEPRECATED - use window_before and window_after instead
        rephrases_json_path (str): Path to the JSON file containing instruction rephrases
        max_episodes (int): Maximum number of episodes to process (for debugging)
        images_folder (str): Path to folder where exterior camera images will be saved as JPG files
        window_before (int): Number of timesteps before current timestep (default: 6, i.e., t-6 to t-1)
        window_after (int): Number of timesteps after current timestep (default: 3, i.e., t+1 to t+3; current t is included)
        use_exterior_camera (bool): If True, use exterior_image_1_left camera (default: True)
        image_size (tuple): Target size for resizing images (width, height), default: (256, 256)
    """
    
    # Create images folder if not specified
    if images_folder is None:
        images_folder = os.path.splitext(output_path)[0] + "_images"
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created images folder: {images_folder}")
    
    # Initialize dataset builder
    print(f"Loading DROID dataset from {builder_dir}...")
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    
    # Load instruction rephrases if provided
    rephrases_dict = None
    if rephrases_json_path is not None:
        print(f"Loading instruction rephrases from {rephrases_json_path}...")
        rephrases_dict = load_instruction_rephrases(rephrases_json_path)
        print(f"Loaded {len(rephrases_dict)} instruction rephrases")
    
    # Memory-optimized data structures
    action_histories = {}  # {hash_id: action_history_list}
    instructions = {}      # {instruction_id: instruction_text}
    samples = []          # [{action_history_id, image_file, instruction_id, episode_id, timestep}]
    
    # Lookup tables for deduplication
    action_history_hash_to_id = {}  # {hash: id}
    instruction_to_id = {}          # {instruction: id}
    
    # Legacy structure for processing
    raw_episode_data_by_instruction = defaultdict(list)
    action_dim = None
    total_episodes_processed = 0
    image_counter = 0  # Global counter for image naming
    next_action_id = 0
    next_instruction_id = 0
    
    # Determine episode range
    if episode_ids is None:
        # Get dataset info to determine total number of episodes
        info = builder.info
        # DROID uses 'train' split by default
        split_name = 'train' if 'train' in info.splits else list(info.splits.keys())[0]
        total_episodes = info.splits[split_name].num_examples
        if max_episodes:
            total_episodes = min(total_episodes, max_episodes)
        episode_ids = list(range(total_episodes))
    else:
        if max_episodes:
            episode_ids = episode_ids[:max_episodes]
    
    print(f"Processing {len(episode_ids)} episodes...")
    
    # Process episodes
    for episode_id in tqdm(episode_ids, desc="Processing episodes"):
        try:
            # Load single episode - DROID uses 'train' split
            split_name = 'train' if 'train' in builder.info.splits else list(builder.info.splits.keys())[0]
            ds = builder.as_dataset(split=f"{split_name}[{episode_id}:{episode_id + 1}]")
            episode = next(iter(ds))
            
            # Extract language instruction from episode
            # DROID stores language instruction at episode level, not step level
            if "language_instruction" in episode:
                language_instruction = episode["language_instruction"].numpy().decode()
            elif "steps" in episode and len(list(episode["steps"])) > 0:
                # Fallback: try to get from first step
                steps = list(episode["steps"])
                if "language_instruction" in steps[0]:
                    language_instruction = steps[0]["language_instruction"].numpy().decode()
                else:
                    print(f"Warning: No language instruction found in episode {episode_id}, skipping...")
                    continue
            else:
                print(f"Warning: Could not extract language instruction from episode {episode_id}, skipping...")
                continue
            
            original_instruction = normalize_instruction(language_instruction)
            
            # Skip episodes with empty or invalid instructions
            if original_instruction is None:
                print(f"Warning: Empty instruction in episode {episode_id}, skipping...")
                continue
            
            # Extract actions and observations for all steps
            steps = list(episode["steps"])
            if not steps:
                continue
            
            actions_list = []
            observations_list = []
            
            for step in steps:
                # Extract action from DROID format
                # DROID actions: cartesian_velocity (6D: translation + rotation) + gripper
                if "action" in step:
                    action = step["action"].numpy()
                elif "action_dict" in step:
                    # DROID stores actions in action_dict
                    action_dict = step["action_dict"]
                    cart_vel = action_dict["cartesian_velocity"].numpy()
                    gripper = action_dict["gripper_position"].numpy()
                    # Invert gripper (DROID convention: 1-gripper_position)
                    action = np.concatenate([cart_vel, 1 - gripper])
                else:
                    print(f"Warning: No action found in episode {episode_id}, step {len(actions_list)}, skipping episode...")
                    break
                
                observation = step["observation"]
                
                # Extract exterior_image_1_left camera image
                exterior_image = None
                if use_exterior_camera:
                    # Use exterior_image_1_left camera
                    if "exterior_image_1_left" in observation:
                        exterior_image = observation["exterior_image_1_left"].numpy()
                    else:
                        print(f"Warning: exterior_image_1_left not found in episode {episode_id}, skipping...")
                        break
                else:
                    # Fallback: try other cameras
                    for img_key in ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left", "wrist_image"]:
                        if img_key in observation:
                            exterior_image = observation[img_key].numpy()
                            break
                
                if exterior_image is None:
                    print(f"Warning: No camera image found in episode {episode_id}, skipping...")
                    break
                
                # Process and save image as JPG
                if exterior_image.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if exterior_image.max() <= 1.0:
                        exterior_image = (exterior_image * 255).astype(np.uint8)
                    else:
                        exterior_image = exterior_image.astype(np.uint8)
                
                # Handle RGBD (4 channels) by taking only RGB
                if len(exterior_image.shape) == 3 and exterior_image.shape[-1] == 4:
                    exterior_image = exterior_image[..., :3]
                elif len(exterior_image.shape) == 2:
                    # Grayscale - convert to RGB
                    exterior_image = np.stack([exterior_image] * 3, axis=-1)
                
                # Convert to PIL Image and resize to target size
                pil_image = Image.fromarray(exterior_image)
                pil_image = pil_image.resize(image_size, Image.LANCZOS)
                
                # Save resized image as JPG
                image_filename = f"{image_counter}.jpg"
                image_path = os.path.join(images_folder, image_filename)
                pil_image.save(image_path, "JPEG", quality=95)
                    
                actions_list.append(action)
                observations_list.append({"exterior_image_file": image_filename})
                image_counter += 1
            
            if len(actions_list) == 0:
                continue
                
            # Convert to numpy arrays
            actions = np.array(actions_list)
            
            # Determine action dimension
            if action_dim is None:
                action_dim = actions.shape[1]
                if action_dim <= 0:
                    raise ValueError("Invalid action dimension")
            elif actions.shape[1] != action_dim:
                print(f"Warning: Inconsistent action dim in episode {episode_id}. Expected {action_dim}, got {actions.shape[1]}. Skipping episode.")
                continue
            
            # Store raw episode data
            T = len(actions_list)
            raw_episode_data_by_instruction[original_instruction].append({
                'actions': actions.tolist(),  # Convert to list for JSON serialization
                'observations': observations_list,
                'len': T,
                'episode_id': episode_id
            })
            
            total_episodes_processed += 1
            
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid episode.")
    if total_episodes_processed == 0:
        raise ValueError("No valid episodes found in the specified dataset.")
    
    print(f"\nProcessed {total_episodes_processed} episodes successfully.")
    print(f"Action dimension: {action_dim}")
    print(f"Saved {image_counter} images (resized to {image_size[0]}x{image_size[1]}) to {images_folder}")
    
    # Generate normalized dataset with deduplicated action histories
    print("Generating memory-optimized samples with deduplicated action histories...")
    print(f"Using centered window: t-{window_before} to t+{window_after} (total window size: {window_before + 1 + window_after})")
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32)
    total_window_size = window_before + 1 + window_after
    
    # First pass: create all samples with original instructions
    for instruction in tqdm(raw_episode_data_by_instruction.keys(), desc="Processing Original Instructions"):
        norm_instruction = normalize_instruction(instruction)
        
        # Skip instructions that become empty after normalization
        if norm_instruction is None:
            print(f"Warning: Instruction '{instruction}' became empty after normalization, skipping...")
            continue
        
        # Get or create instruction ID
        instruction_id, next_instruction_id = get_or_create_instruction_id(
            norm_instruction, instructions, instruction_to_id, next_instruction_id
        )
        
        # Double-check that instruction_id is valid
        if instruction_id is None:
            print(f"Warning: Could not create instruction ID for '{norm_instruction}', skipping...")
            continue
        
        for episode_data in raw_episode_data_by_instruction[instruction]:
            actions = np.array(episode_data['actions'])  # Convert back to numpy for processing
            observations = episode_data['observations']
            T = episode_data['len']
            episode_id = episode_data['episode_id']
            
            for t in range(T):
                # Generate centered window: [t-window_before, ..., t, ..., t+window_after]
                start_idx = t - window_before
                end_idx = t + window_after + 1  # +1 because range is exclusive
                
                # Calculate padding needed at start and end
                padding_before = max(0, -start_idx)  # How many steps before 0
                padding_after = max(0, end_idx - T)   # How many steps after T-1
                
                # Adjust indices to valid range
                actual_start = max(0, start_idx)
                actual_end = min(T, end_idx)
                
                # Get actual actions in the window
                actual_actions = actions[actual_start:actual_end]
                
                # Add padding if needed
                if padding_before > 0 or padding_after > 0:
                    parts = []
                    if padding_before > 0:
                        parts.append(np.repeat(padding_array_template, padding_before, axis=0))
                    parts.append(actual_actions)
                    if padding_after > 0:
                        parts.append(np.repeat(padding_array_template, padding_after, axis=0))
                    action_hist = np.concatenate(parts, axis=0).astype(actions.dtype)
                else:
                    action_hist = actual_actions
                
                # Skip all-padded samples
                if np.all(action_hist == ACTION_PADDING_VALUE):
                    continue
                
                # Final check on window size
                if action_hist.shape[0] != total_window_size:
                    warnings.warn(f"Generated window size mismatch for '{instruction}' t={t}. Expected {total_window_size}, got {action_hist.shape[0]}. Skipping.")
                    continue
                
                # Get or create action history ID (deduplication happens here)
                action_history_id, next_action_id = get_or_create_action_history_id(
                    action_hist, action_histories, action_history_hash_to_id, next_action_id
                )
                
                # Create normalized sample
                sample = {
                    'action_history_id': action_history_id,
                    'exterior_image_file': observations[t]['exterior_image_file'],
                    'instruction_id': instruction_id,
                    'episode_id': episode_id,
                    'timestep': t
                }
                
                samples.append(sample)
    
    # Second pass: add rephrased instructions (reusing action histories and images)
    if rephrases_dict is not None:
        print("Adding rephrased instructions...")
        original_samples_count = len(samples)
        
        for orig_instruction, rephrases in tqdm(rephrases_dict.items(), desc="Processing Rephrases"):
            if orig_instruction in instruction_to_id:
                original_instruction_id = instruction_to_id[orig_instruction]
                
                # Find all samples with this original instruction
                original_samples = [s for s in samples if s['instruction_id'] == original_instruction_id]
                
                for reph in rephrases:
                    norm_reph = normalize_instruction(reph)
                    
                    # Skip empty rephrases
                    if norm_reph is None:
                        print(f"Warning: Rephrase '{reph}' became empty after normalization, skipping...")
                        continue
                    
                    # Get or create rephrase instruction ID
                    reph_instruction_id, next_instruction_id = get_or_create_instruction_id(
                        norm_reph, instructions, instruction_to_id, next_instruction_id
                    )
                    
                    # Skip if instruction ID creation failed
                    if reph_instruction_id is None:
                        print(f"Warning: Could not create instruction ID for rephrase '{norm_reph}', skipping...")
                        continue
                    
                    # Create new samples with rephrased instruction (reusing action_history_id and image)
                    for original_sample in original_samples:
                        rephrase_sample = {
                            'action_history_id': original_sample['action_history_id'],  # Reuse action history
                            'exterior_image_file': original_sample['exterior_image_file'],  # Reuse image
                            'instruction_id': reph_instruction_id,  # New instruction
                            'episode_id': original_sample['episode_id'],
                            'timestep': original_sample['timestep']
                        }
                        samples.append(rephrase_sample)
        
        print(f"Added {len(samples) - original_samples_count} rephrased samples")
    
    # Create optimized final dataset structure
    total_instructions = len(instructions)
    total_samples = len(samples)
    total_unique_action_histories = len(action_histories)
    
    print(f"\nMemory optimization results:")
    print(f"Total unique action histories: {total_unique_action_histories}")
    print(f"Total samples: {total_samples}")
    print(f"Compression ratio: {total_samples / total_unique_action_histories:.2f}x samples per unique action history")
    
    # Validation: Check for empty instructions
    empty_instruction_count = 0
    for instr_id, instr_text in instructions.items():
        if not instr_text or not instr_text.strip():
            empty_instruction_count += 1
    
    if empty_instruction_count > 0:
        print(f"Warning: Found {empty_instruction_count} empty instructions that were kept in the dataset")
    else:
        print("✓ All instructions are non-empty")
    
    # Build the final optimized dataset
    final_dataset = {
        'action_histories': action_histories,
        'instructions': instructions,
        'samples': samples,
        '_metadata': {
            'action_dim': action_dim,
            'window_before': window_before,
            'window_after': window_after,
            'total_window_size': total_window_size,
            'history_length': history_length,  # Keep for backward compatibility
            'total_images': image_counter,
            'images_folder': images_folder,
            'total_instructions': total_instructions,
            'total_samples': total_samples,
            'total_unique_action_histories': total_unique_action_histories,
            'compression_ratio': total_samples / total_unique_action_histories if total_unique_action_histories > 0 else 0,
            'format_version': '2.1_centered_window_droid',
            'padding_value': ACTION_PADDING_VALUE,
            'camera_type': 'exterior_image_1_left' if use_exterior_camera else 'mixed',
            'image_size': image_size
        }
    }
    
    print(f"\nDataset extraction complete!")
    print(f"Window configuration: t-{window_before} to t+{window_after} (total size: {total_window_size})")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"Camera type: {'exterior_image_1_left' if use_exterior_camera else 'Mixed cameras'}")
    print(f"Total instructions: {total_instructions}")
    print(f"Total samples: {total_samples}")
    print(f"Total unique action histories: {total_unique_action_histories}")
    print(f"Memory efficiency: {total_samples / total_unique_action_histories:.2f}x reuse of action histories")
    
    # Calculate instruction usage statistics
    instruction_usage = defaultdict(int)
    for sample in samples:
        instruction_usage[sample['instruction_id']] += 1
    
    usage_counts = list(instruction_usage.values())
    if usage_counts:
        print(f"Average samples per instruction: {np.mean(usage_counts):.2f}")
        print(f"Min samples per instruction: {np.min(usage_counts)}")
        print(f"Max samples per instruction: {np.max(usage_counts)}")
    
    # Save dataset as JSON
    print(f"\nSaving optimized dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print("Done!")
    
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract DROID dataset with action histories, wrist camera images, and language instructions.')
    parser.add_argument('--builder_dir', type=str, default='/root/tfrecords/droid_100/1.0.0',
                        help='Path to the DROID dataset directory')
    parser.add_argument('--episode_ids', nargs='+', type=int, default=None,
                        help='Specific episode IDs to process (if not specified, processes all)')
    parser.add_argument('--output_path', type=str, default='droid_dataset_extracted.json',
                        help='Path to save the extracted dataset (JSON file)')
    parser.add_argument('--history_length', type=int, default=10,
                        help='DEPRECATED - use --window_before and --window_after instead')
    parser.add_argument('--window_before', type=int, default=6,
                        help='Number of timesteps before current timestep to include in window (default: 6, i.e., t-6 to t-1)')
    parser.add_argument('--window_after', type=int, default=3,
                        help='Number of timesteps after current timestep to include in window (default: 3, i.e., t+1 to t+3; current t is included as first future step)')
    parser.add_argument('--rephrases_json', type=str, 
                        default=None,
                        help='Path to the JSON file containing instruction rephrases')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to process (for debugging/testing)')
    parser.add_argument('--images_folder', type=str, default=None,
                        help='Path to folder where exterior camera images will be saved as JPG files (default: output_path_images)')
    parser.add_argument('--use_exterior_camera', action='store_true', default=True,
                        help='Use exterior_image_1_left camera (default: True)')
    parser.add_argument('--use_all_cameras', action='store_true',
                        help='Use all available cameras (overrides --use_exterior_camera)')
    parser.add_argument('--image_width', type=int, default=256,
                        help='Target image width for resizing (default: 256)')
    parser.add_argument('--image_height', type=int, default=256,
                        help='Target image height for resizing (default: 256)')
    
    args = parser.parse_args()
    
    # Handle camera selection flags
    use_exterior = args.use_exterior_camera and not args.use_all_cameras
    
    # Image size
    image_size = (args.image_width, args.image_height)
    
    # Extract dataset
    final_dataset = extract_droid_dataset(
        builder_dir=args.builder_dir,
        episode_ids=args.episode_ids,
        output_path=args.output_path,
        history_length=args.history_length,
        rephrases_json_path=args.rephrases_json,
        max_episodes=args.max_episodes,
        images_folder=args.images_folder,
        window_before=args.window_before,
        window_after=args.window_after,
        use_exterior_camera=use_exterior,
        image_size=image_size
    )

