#!/bin/bash
# Script for full training of OpenVLA on DROID dataset from scratch
# Requires 8 GPUs with ~80GB VRAM each

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Dataset paths
DATA_ROOT_DIR="/path/to/datasets"  # Parent directory containing droid/ folder

# Output paths
RUN_ROOT_DIR="./runs/droid_full_$(date +%Y%m%d_%H%M%S)"

# Training configuration
VLA_TYPE="prism-dinosiglip-224px+mx-droid"  # Training config for DROID

# Image augmentation
IMAGE_AUG=True                     # Enable image augmentation (recommended)

# Hardware configuration
NUM_NODES=1                        # Number of nodes
NUM_GPUS=8                         # GPUs per node (8 recommended)

# Logging
WANDB_PROJECT="openvla-droid-full" # Weights & Biases project name
WANDB_ENTITY=""                    # Your W&B entity/username (leave empty to use default)
SAVE_INTERVAL=5000                 # Save checkpoint every N gradient steps

# Optional: Resume from checkpoint
PRETRAINED_CHECKPOINT=""           # Path to checkpoint to resume from (leave empty for fresh start)
IS_RESUME=False                    # Set to True if resuming training

# ============================================================================
# VALIDATION
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OpenVLA DROID Full Training Script ===${NC}"
echo ""

# Check if DATA_ROOT_DIR exists
if [ ! -d "${DATA_ROOT_DIR}" ]; then
    echo -e "${RED}Error: DATA_ROOT_DIR does not exist: ${DATA_ROOT_DIR}${NC}"
    echo "Please set DATA_ROOT_DIR to the parent directory containing your droid/ folder"
    exit 1
fi

# Check if dataset exists
DATASET_PATH="${DATA_ROOT_DIR}/droid"
if [ ! -d "${DATASET_PATH}" ]; then
    echo -e "${RED}Error: DROID dataset not found: ${DATASET_PATH}${NC}"
    echo ""
    echo "To download DROID dataset, run:"
    echo -e "  ${YELLOW}./download_droid.sh full ${DATA_ROOT_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset found: ${DATASET_PATH}${NC}"

# Check GPU count
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
    echo -e "${YELLOW}Warning: Requested ${NUM_GPUS} GPUs, but only ${AVAILABLE_GPUS} available${NC}"
    echo "Recommended: 8 GPUs with ~80GB VRAM each"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 0
    fi
fi

# Create output directory
mkdir -p "${RUN_ROOT_DIR}"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo ""
echo -e "${BLUE}Training Configuration:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "VLA Type:           ${YELLOW}${VLA_TYPE}${NC}"
echo -e "Dataset:            ${YELLOW}droid${NC} (${DATASET_PATH})"
echo -e "Output directory:   ${YELLOW}${RUN_ROOT_DIR}${NC}"
echo ""
echo -e "Image augmentation: ${YELLOW}${IMAGE_AUG}${NC}"
echo -e "Number of nodes:    ${YELLOW}${NUM_NODES}${NC}"
echo -e "GPUs per node:      ${YELLOW}${NUM_GPUS}${NC}"
echo -e "Save every:         ${YELLOW}${SAVE_INTERVAL}${NC} steps"
echo ""
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    echo -e "Resume from:        ${YELLOW}${PRETRAINED_CHECKPOINT}${NC}"
    echo -e "Is resume:          ${YELLOW}${IS_RESUME}${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}Training Details (from config):${NC}"
echo "  - Base VLM: prism-dinosiglip-224px+7b"
echo "  - Global batch size: 256"
echo "  - Per-device batch size: 32"
echo "  - Learning rate: 2e-5"
echo "  - Epochs: 1000 (or specify max_steps)"
echo "  - Shuffle buffer: 500K samples"
echo ""

# Warning message
echo -e "${RED}WARNING: Full training requires significant computational resources:${NC}"
echo "  - 8 GPUs with ~80GB VRAM each"
echo "  - Training may take several days/weeks"
echo "  - Ensure sufficient storage for checkpoints"
echo ""

# Ask for confirmation
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

TRAIN_CMD="torchrun --standalone --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS} vla-scripts/train.py"
TRAIN_CMD="${TRAIN_CMD} --vla.type \"${VLA_TYPE}\""
TRAIN_CMD="${TRAIN_CMD} --data_root_dir \"${DATA_ROOT_DIR}\""
TRAIN_CMD="${TRAIN_CMD} --run_root_dir \"${RUN_ROOT_DIR}\""
TRAIN_CMD="${TRAIN_CMD} --image_aug ${IMAGE_AUG}"
TRAIN_CMD="${TRAIN_CMD} --save_interval ${SAVE_INTERVAL}"

# Add pretrained checkpoint if specified
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained_checkpoint \"${PRETRAINED_CHECKPOINT}\""
    TRAIN_CMD="${TRAIN_CMD} --is_resume ${IS_RESUME}"
fi

# Add W&B arguments if configured
if [ -n "${WANDB_PROJECT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --wandb_project ${WANDB_PROJECT}"
fi
if [ -n "${WANDB_ENTITY}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --wandb_entity ${WANDB_ENTITY}"
fi

# ============================================================================
# RUN TRAINING
# ============================================================================

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""
echo -e "${YELLOW}Command:${NC}"
echo "${TRAIN_CMD}" | sed 's/ --/\n  --/g'
echo ""

# Save command to file
echo "${TRAIN_CMD}" > "${RUN_ROOT_DIR}/train_command.txt"
echo -e "${GREEN}✓ Command saved to: ${RUN_ROOT_DIR}/train_command.txt${NC}"
echo ""

# Log start time
START_TIME=$(date)
echo "Training started at: ${START_TIME}" | tee "${RUN_ROOT_DIR}/training.log"
echo "" | tee -a "${RUN_ROOT_DIR}/training.log"

# Execute training
eval ${TRAIN_CMD} 2>&1 | tee -a "${RUN_ROOT_DIR}/training.log"

# Log end time
END_TIME=$(date)
echo "" | tee -a "${RUN_ROOT_DIR}/training.log"
echo "Training started at:  ${START_TIME}" | tee -a "${RUN_ROOT_DIR}/training.log"
echo "Training finished at: ${END_TIME}" | tee -a "${RUN_ROOT_DIR}/training.log"

echo ""
echo -e "${GREEN}=== Training Complete ===${NC}"
echo -e "Checkpoints saved in: ${YELLOW}${RUN_ROOT_DIR}${NC}"
echo -e "Full log available at: ${YELLOW}${RUN_ROOT_DIR}/training.log${NC}"

