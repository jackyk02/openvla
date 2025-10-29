#!/usr/bin/env python3
"""
Create videos for episodes with instructions overlaid on top.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm


def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        # Use getbbox instead of getsize for PIL >= 10.0.0
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def add_text_to_frame(frame, text, font_scale=1.0, thickness=2, bg_color=(0, 0, 0), 
                      text_color=(255, 255, 255), padding=10):
    """
    Add text with background to the top of a frame using PIL for better text rendering
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a nice font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                  size=int(24 * font_scale))
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
                                      size=int(24 * font_scale))
        except:
            font = ImageFont.load_default()
    
    # Wrap text to fit frame width
    img_width = pil_img.width
    max_text_width = img_width - (padding * 4)
    text_lines = wrap_text(text, font, max_text_width)
    
    # Calculate text height
    line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1] + 5
    total_text_height = line_height * len(text_lines)
    
    # Create background rectangle
    bg_height = total_text_height + (padding * 2)
    draw.rectangle([(0, 0), (img_width, bg_height)], fill=bg_color)
    
    # Draw each line of text
    y_position = padding
    for line in text_lines:
        # Center the text horizontally
        bbox = font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        x_position = (img_width - text_width) // 2
        
        # Draw text with slight offset for shadow effect
        draw.text((x_position + 2, y_position + 2), line, font=font, fill=(0, 0, 0))
        draw.text((x_position, y_position), line, font=font, fill=text_color)
        
        y_position += line_height
    
    # Convert back to BGR for OpenCV
    frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return frame_with_text


def create_episode_video(episode_id, samples, dataset_dict, images_folder, 
                        output_path, fps=10, font_scale=1.0):
    """
    Create a video for a single episode with instruction overlay
    
    Args:
        episode_id: Episode identifier
        samples: List of sample dictionaries for this episode
        dataset_dict: Full dataset dictionary with action_histories and instructions
        images_folder: Path to folder containing images
        output_path: Path to save output video
        fps: Frames per second for output video
        font_scale: Scale factor for text size
    """
    # Get instruction for this episode
    if not samples:
        print(f"No samples for episode {episode_id}")
        return False
    
    instruction_id = samples[0].get('instruction_id')
    if not instruction_id or instruction_id not in dataset_dict['instructions']:
        print(f"No instruction found for episode {episode_id}")
        return False
    
    instruction_text = dataset_dict['instructions'][instruction_id]
    
    print(f"\nCreating video for Episode {episode_id}")
    print(f"  Instruction: \"{instruction_text}\"")
    print(f"  Samples: {len(samples)}")
    
    # Sort samples by timestep
    sorted_samples = sorted(samples, key=lambda x: x.get('timestep', 0))
    
    # Collect frames
    frames = []
    valid_samples = 0
    
    for sample in sorted_samples:
        # Get image file (DROID uses exterior_image_file)
        image_file = sample.get('exterior_image_file')
        if not image_file:
            continue
        
        image_path = os.path.join(images_folder, image_file)
        if not os.path.exists(image_path):
            continue
        
        # Load image
        try:
            img = cv2.imread(image_path)
            if img is not None:
                frames.append(img)
                valid_samples += 1
        except Exception as e:
            print(f"  Warning: Could not load image {image_file}: {e}")
            continue
    
    if len(frames) == 0:
        print(f"  Error: No valid frames for episode {episode_id}")
        return False
    
    print(f"  Valid frames: {len(frames)}")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  Error: Could not create video writer")
        return False
    
    # Write frames with instruction overlay
    for frame in tqdm(frames, desc=f"  Writing frames", leave=False):
        frame_with_text = add_text_to_frame(
            frame, 
            instruction_text,
            font_scale=font_scale,
            thickness=2,
            bg_color=(0, 0, 0),
            text_color=(255, 255, 255),
            padding=10
        )
        out.write(frame_with_text)
    
    out.release()
    
    # Verify video was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ Video saved: {output_path} ({file_size:.2f} MB)")
        return True
    else:
        print(f"  Error: Video file not created")
        return False


def create_videos_for_episodes(scores_path, dataset_path, images_folder, 
                               output_dir, num_episodes=5, mode='bottom', 
                               fps=10, font_scale=1.0):
    """
    Create videos for top or bottom N episodes by score
    
    Args:
        scores_path: Path to episode scores JSON
        dataset_path: Path to dataset JSON
        images_folder: Path to folder containing images
        output_dir: Directory to save videos
        num_episodes: Number of episodes to process
        mode: 'top' for highest scores, 'bottom' for lowest scores
        fps: Frames per second
        font_scale: Text size scale factor
    """
    print("="*80)
    print(f"Creating Videos for {mode.title()} {num_episodes} Episodes")
    print("="*80)
    
    # Load scores
    print(f"\nLoading episode scores from: {scores_path}")
    with open(scores_path, 'r') as f:
        scores_data = json.load(f)
    
    episode_scores = scores_data['episode_scores']
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Sort episodes by mean score
    sorted_episodes = sorted(
        episode_scores.items(), 
        key=lambda x: x[1]['mean_score'],
        reverse=(mode == 'top')
    )
    
    # Select top/bottom N episodes
    selected_episodes = sorted_episodes[:num_episodes]
    
    print(f"\n{mode.title()} {num_episodes} Episodes by Mean Score:")
    for episode_id, scores in selected_episodes:
        print(f"  Episode {episode_id}: {scores['mean_score']:.4f} (n={scores['num_samples']})")
    
    # Group dataset samples by episode
    samples_by_episode = {}
    for sample in dataset_dict['samples']:
        episode_id = str(sample.get('episode_id'))
        if episode_id not in samples_by_episode:
            samples_by_episode[episode_id] = []
        samples_by_episode[episode_id].append(sample)
    
    # Create videos
    print(f"\nCreating videos...")
    success_count = 0
    
    for episode_id, scores in selected_episodes:
        episode_id_str = str(episode_id)
        
        if episode_id_str not in samples_by_episode:
            print(f"\nWarning: No samples found for episode {episode_id}")
            continue
        
        samples = samples_by_episode[episode_id_str]
        
        # Create output filename
        mean_score = scores['mean_score']
        output_filename = f"episode_{episode_id}_score_{mean_score:.4f}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create video
        success = create_episode_video(
            episode_id=episode_id,
            samples=samples,
            dataset_dict=dataset_dict,
            images_folder=images_folder,
            output_path=output_path,
            fps=fps,
            font_scale=font_scale
        )
        
        if success:
            success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Videos created: {success_count}/{num_episodes}")
    print(f"Output directory: {output_dir}")
    
    if success_count > 0:
        print("\nCreated videos:")
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.mp4'):
                filepath = os.path.join(output_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filename} ({file_size:.2f} MB)")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create videos for episodes with instruction overlay'
    )
    
    parser.add_argument('--scores', type=str, default='droid_episode_scores.json',
                       help='Path to episode scores JSON')
    parser.add_argument('--dataset', type=str, default='droid_bridge_converted.json',
                       help='Path to dataset JSON')
    parser.add_argument('--images', type=str, default='droid_extracted_images',
                       help='Path to folder containing images')
    parser.add_argument('--output_dir', type=str, default='droid_episode_videos',
                       help='Directory to save videos')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to process (default: 5)')
    parser.add_argument('--mode', type=str, default='bottom', choices=['top', 'bottom'],
                       help='Select top or bottom episodes (default: bottom)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for videos (default: 10)')
    parser.add_argument('--font_scale', type=float, default=1.0,
                       help='Font size scale factor (default: 1.0)')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.scores):
        print(f"Error: Scores file not found: {args.scores}")
        print("\nPlease compute episode scores first:")
        print("  ./compute_droid_scores_example.sh")
        exit(1)
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        exit(1)
    
    if not os.path.exists(args.images):
        print(f"Error: Images folder not found: {args.images}")
        exit(1)
    
    # Create videos
    create_videos_for_episodes(
        scores_path=args.scores,
        dataset_path=args.dataset,
        images_folder=args.images,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        mode=args.mode,
        fps=args.fps,
        font_scale=args.font_scale
    )

