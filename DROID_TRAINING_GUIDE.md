# Guide: Training OpenVLA on DROID Dataset Only

This guide provides step-by-step instructions for training OpenVLA exclusively on the DROID (Distributed Robot Interaction Dataset) from the Open X-Embodiment collection.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Download DROID Dataset](#download-droid-dataset)
3. [Setup OpenVLA Environment](#setup-openvla-environment)
4. [Training Options](#training-options)
5. [Monitor Training](#monitor-training)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **Training from Scratch**: 8x GPUs (recommended: A100 80GB)
- **Fine-tuning with LoRA**: 1x GPU with at least 27GB VRAM
- **Storage**: ~1.7TB for full DROID dataset (or ~2GB for droid_100 subset)

### Software Requirements
- Python 3.10+
- PyTorch 2.2.0
- CUDA-compatible GPU
- Google Cloud SDK (for downloading dataset)

---

## Download DROID Dataset

### Step 1: Install Google Cloud SDK

If you don't have `gsutil` installed:

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize (optional, but recommended)
gcloud init
```

### Step 2: Download DROID Dataset

Choose one of the following options:

#### Option A: Full DROID Dataset (~1.7TB)
```bash
# Set your dataset directory
export DATASET_DIR=/path/to/your/datasets

# Download full DROID dataset in RLDS format
gsutil -m cp -r gs://gresearch/robotics/droid ${DATASET_DIR}/droid
```

#### Option B: DROID-100 Subset (~2GB - for debugging/testing)
```bash
# Set your dataset directory
export DATASET_DIR=/path/to/your/datasets

# Download smaller subset
gsutil -m cp -r gs://gresearch/robotics/droid_100 ${DATASET_DIR}/droid_100
```

**Note**: The download may take several hours for the full dataset depending on your internet connection.

---

## Setup OpenVLA Environment

### Step 1: Clone and Install OpenVLA

```bash
# Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone OpenVLA repository
git clone https://github.com/openvla/openvla.git
cd openvla

# Install OpenVLA
pip install -e .

# Install Flash Attention 2 (optional but recommended for training)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### Step 2: Verify Dataset Structure

After downloading, your dataset directory should look like:

```
/path/to/datasets/
└── droid/
    ├── 1.0.0/
    │   ├── dataset_info.json
    │   └── [trajectory files...]
    └── ...
```

---

## Training Options

I've configured OpenVLA with support for DROID-only training. You have two main options:

### Option 1: Full Training from Scratch (Recommended for Research)

Train a full VLA model on DROID dataset from a pre-trained VLM backbone.

**Requirements**: 8 GPUs with ~80GB VRAM each

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-droid" \
  --data_root_dir /path/to/datasets \
  --run_root_dir /path/to/logs_and_checkpoints \
  --wandb_project "openvla-droid" \
  --wandb_entity "your-wandb-entity" \
  --save_interval 5000
```

**Configuration Details** (from `prismatic/conf/vla.py`):
- Base VLM: `prism-dinosiglip-224px+7b` (DINO + SigLIP vision backbone + Llama-2 7B)
- Expected world size: 8 GPUs
- Global batch size: 256
- Per-device batch size: 32
- Learning rate: 2e-5
- Epochs: 1000 (or specify `--max_steps`)
- Shuffle buffer: 500K samples

**Key Arguments**:
- `--vla.type`: Specifies the training configuration (use `"prism-dinosiglip-224px+mx-droid"`)
- `--data_root_dir`: Path to your datasets directory (parent of `droid/`)
- `--run_root_dir`: Where to save logs and checkpoints
- `--image_aug`: Set to `True` for data augmentation (recommended)
- `--save_interval`: Save checkpoint every N gradient steps

### Option 2: Fine-tuning Pre-trained OpenVLA via LoRA

Fine-tune the pre-trained OpenVLA model on DROID dataset using LoRA (Low-Rank Adaptation).

**Requirements**: 1 GPU with ~72GB VRAM (or less with smaller batch sizes)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /path/to/datasets \
  --dataset_name droid \
  --run_root_dir /path/to/logs_and_checkpoints \
  --adapter_tmp_dir /path/to/adapter_weights \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla-droid-lora" \
  --wandb_entity "your-wandb-entity" \
  --save_steps 2500
```

**Note**: If you have less GPU memory, reduce `--batch_size` and increase `--grad_accumulation_steps` proportionally to maintain effective batch size:
- 48GB GPU: `--batch_size 8 --grad_accumulation_steps 2`
- 32GB GPU: `--batch_size 4 --grad_accumulation_steps 4`

### Option 3: Fine-tuning on DROID Subset (droid_wipe)

If you want to fine-tune on a specific task subset:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /path/to/datasets \
  --dataset_name droid_wipe \
  --run_root_dir /path/to/logs_and_checkpoints \
  --adapter_tmp_dir /path/to/adapter_weights \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla-droid-wipe" \
  --wandb_entity "your-wandb-entity" \
  --save_steps 1000
```

---

## Monitor Training

### Weights & Biases (Recommended)

Set up W&B for experiment tracking:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Use `--wandb_project` and `--wandb_entity` flags in training commands

Monitor:
- Training loss
- Action token accuracy
- Learning rate schedule
- GPU utilization

### Local Logs

Checkpoints and logs are saved in `--run_root_dir`:
```
/path/to/logs_and_checkpoints/
├── prism-dinosiglip-224px+mx-droid+nX+bY+...
│   ├── checkpoints/
│   │   ├── step-005000-epoch-XX-loss=X.XXXX.pt
│   │   └── ...
│   ├── config.yaml
│   └── logs/
```

---

## DROID Dataset Information

### Dataset Overview
- **Size**: ~1.7TB (RLDS format)
- **Source**: Open X-Embodiment collection
- **Content**: Multi-task robot manipulation demonstrations
- **Robot Platforms**: Various Franka robots and other manipulators
- **Action Space**: End-effector velocity control (6 DoF + gripper)

### Dataset Configuration
The DROID dataset in OpenVLA uses:
- **Primary Camera**: `exterior_image_1_left`
- **Secondary Camera**: `exterior_image_2_left` (randomly swapped during training)
- **Wrist Camera**: `wrist_image_left`
- **State Observations**: Cartesian position (7D: xyz + quaternion) + gripper state
- **Actions**: Cartesian velocity (6D: linear + angular) + gripper command (1D)
- **State Encoding**: Position + Quaternion
- **Action Encoding**: End-effector position control

### Data Transforms
The dataset uses `droid_baseact_transform` which:
1. Extracts cartesian velocity commands (translation + rotation)
2. Inverts gripper position (1 - gripper_position)
3. Randomly swaps exterior camera images for augmentation
4. Concatenates proprioceptive state (position + gripper)
5. Filters out zero-action trajectories during training

---

## Troubleshooting

### Issue: "Dataset not found" Error

**Solution**: Ensure the dataset path is correct. The structure should be:
```
${data_root_dir}/droid/1.0.0/dataset_info.json
```

### Issue: Out of Memory (OOM) Error

**Solutions**:
1. Reduce batch size: `--per_device_batch_size 16` (or lower)
2. Enable gradient checkpointing (should be enabled by default)
3. Use smaller model or LoRA fine-tuning instead of full training
4. Reduce shuffle buffer size in config

### Issue: Slow Download Speed

**Solutions**:
1. Use `-m` flag with gsutil for parallel downloads (already included)
2. Increase number of parallel processes: `gsutil -m -o "GSUtil:parallel_process_count=16" cp -r ...`
3. Download during off-peak hours
4. Consider downloading subset first for testing

### Issue: Training Loss Not Decreasing

**Checks**:
1. Verify dataset is loading correctly (check logs for dataset statistics)
2. Ensure data transforms are applied (check action shapes in logs)
3. Monitor action token accuracy (should be > 0.1 initially)
4. Try adjusting learning rate
5. Enable image augmentation if not already: `--image_aug True`

### Issue: "tensorflow-datasets" Version Error

**Solution**: Downgrade tensorflow-datasets:
```bash
pip install tensorflow-datasets==4.9.3
```

### Issue: "AttributeError: 'DLataset' object has no attribute 'traj_map'"

**Solution**: Upgrade dlimp:
```bash
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
```

---

## Extracting DROID Dataset for Custom Processing

If you need to extract and preprocess the DROID dataset into a memory-efficient JSON format (e.g., for custom training pipelines or data analysis), we provide an extraction script.

### Using the Extraction Script

The `augment_droid_dataset.py` script extracts:
- **Exterior camera images** (exterior_image_1_left, resized to 256x256, saved as JPG files)
- **Action histories** with temporal windows (deduplicated for memory efficiency)
- **Language instructions** (with optional rephrasing support)
- **Episode and timestep metadata**

#### Quick Example

```bash
# Extract first 10 episodes for testing
python augment_droid_dataset.py \
  --builder_dir /root/tfrecords/droid_100/1.0.0 \
  --output_path droid_100_extracted.json \
  --images_folder droid_100_images \
  --max_episodes 10 \
  --use_exterior_camera
```

Or use the provided example script:
```bash
./extract_droid_example.sh
```

#### Full Extraction

```bash
# Extract all episodes from droid_100
python augment_droid_dataset.py \
  --builder_dir /root/tfrecords/droid_100/1.0.0 \
  --output_path droid_100_full.json \
  --images_folder droid_100_images_full \
  --window_before 6 \
  --window_after 3 \
  --use_exterior_camera
```

#### With Instruction Rephrases

If you have a JSON file with instruction rephrases:
```bash
python augment_droid_dataset.py \
  --builder_dir /root/tfrecords/droid_100/1.0.0 \
  --output_path droid_augmented.json \
  --rephrases_json instruction_rephrases.json \
  --use_exterior_camera
```

### Script Arguments

- `--builder_dir`: Path to DROID dataset directory (e.g., `/root/tfrecords/droid_100/1.0.0`)
- `--output_path`: Output JSON file path
- `--images_folder`: Folder to save extracted images (default: `<output_path>_images`)
- `--max_episodes`: Limit number of episodes (for testing)
- `--window_before`: Timesteps before current (default: 6)
- `--window_after`: Timesteps after current (default: 3)
- `--use_exterior_camera`: Use exterior_image_1_left camera (default: True)
- `--use_all_cameras`: Use all available cameras (overrides exterior camera)
- `--image_width`: Target image width for resizing (default: 256)
- `--image_height`: Target image height for resizing (default: 256)
- `--rephrases_json`: Optional JSON file with instruction rephrases

### Output Format

The script produces:
1. **JSON file** with structure:
   - `action_histories`: Deduplicated action sequences
   - `instructions`: Unique instruction texts
   - `samples`: List of training samples linking actions, images, and instructions
   - `_metadata`: Dataset statistics and configuration

2. **Images folder** containing all wrist camera images as JPG files

### Example Output Structure

```json
{
  "action_histories": {
    "action_0": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    "action_1": [[0.5, 0.6, ...], [0.7, 0.8, ...], ...]
  },
  "instructions": {
    "instr_0": "pick up the red block",
    "instr_1": "place the object in the bowl"
  },
  "samples": [
    {
      "action_history_id": "action_0",
      "exterior_image_file": "0.jpg",
      "instruction_id": "instr_0",
      "episode_id": 0,
      "timestep": 5
    }
  ],
  "_metadata": {
    "action_dim": 7,
    "window_before": 6,
    "window_after": 3,
    "total_samples": 1000,
    "camera_type": "exterior_image_1_left",
    "image_size": [256, 256]
  }
}
```

---

## Converting DROID Actions to Bridge Scale

If you plan to fine-tune a Bridge-trained OpenVLA model with DROID data, you may want to convert the actions to Bridge scale.

### Why Convert?

- **Bridge models** are trained on Bridge V2 dataset with specific action scales
- **DROID actions** use different velocity magnitudes
- **Token-based conversion** maps actions through the same discretization used during training

### Quick Conversion

```bash
# Convert extracted DROID dataset to Bridge scale
./convert_droid_example.sh
```

Or manually:
```bash
python convert_droid_to_bridge.py \
  --input droid_extracted.json \
  --output droid_bridge_converted.json
```

### What Gets Converted

- **Cartesian velocities** (6D): Mapped from DROID scale to Bridge scale via tokens
- **Gripper** (1D): Binarized based on 0.5 threshold
  - Original < 0.5 → 0.0 (open)
  - Original ≥ 0.5 → 1.0 (closed)

### Output

The converted dataset has:
- Same structure as input (action_histories, instructions, samples)
- Actions in Bridge scale (7D: 6D cartesian + 1D binary gripper)
- Metadata with conversion parameters and statistics

See `ACTION_CONVERSION_README.md` for detailed documentation on the conversion method.

---

## Additional Resources

- **DROID Dataset Paper**: [https://droid-dataset.github.io](https://droid-dataset.github.io)
- **OpenVLA Paper**: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- **OpenVLA GitHub**: [https://github.com/openvla/openvla](https://github.com/openvla/openvla)
- **Open X-Embodiment**: [https://robotics-transformer-x.github.io/](https://robotics-transformer-x.github.io/)

---

## Example: Complete Training Workflow

Here's a complete example from scratch:

```bash
# 1. Setup environment
conda create -n openvla python=3.10 -y
conda activate openvla
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 2. Clone and install OpenVLA
cd ~
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# 3. Download DROID dataset
export DATASET_DIR=/data/robot_datasets
mkdir -p ${DATASET_DIR}
gsutil -m cp -r gs://gresearch/robotics/droid ${DATASET_DIR}/droid

# 4. Setup W&B
pip install wandb
wandb login

# 5. Start training (LoRA fine-tuning example)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ${DATASET_DIR} \
  --dataset_name droid \
  --run_root_dir ./experiments/droid_lora \
  --adapter_tmp_dir ./experiments/droid_lora/adapters \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla-droid" \
  --wandb_entity "my-entity" \
  --save_steps 2500
```

---

## Citation

If you use DROID dataset or OpenVLA in your research, please cite:

```bibtex
@article{khazatsky2024droid,
  title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
  author={Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and Balakrishna, Ashwin and Dasari, Sudeep and Karamcheti, Siddharth and Nasiriany, Soroush and Srirama, Mohan Kumar and Chen, Lawrence Yunliang and Ellis, Kevin and Finnegan, Peter and Grannen, Jennifer and Hu, Moo Jin and Itkina, Masha and Jain, Vidhi and Karamouzas, Ioannis and Kim, Minjune and Li, Yuchen and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2403.12945},
  year={2024}
}

@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

