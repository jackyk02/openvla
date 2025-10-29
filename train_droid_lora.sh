#!/bin/bash
# Quick-start script for fine-tuning OpenVLA on DROID dataset using LoRA
# This script provides a template - adjust parameters based on your setup

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Dataset paths
DATA_ROOT_DIR="/path/to/datasets"  # Parent directory containing droid/ folder
DATASET_NAME="droid"               # Options: droid, droid_100, droid_wipe

# Output paths
RUN_ROOT_DIR="./runs/droid_lora_$(date +%Y%m%d_%H%M%S)"
ADAPTER_TMP_DIR="${RUN_ROOT_DIR}/adapters"

# Model configuration
VLA_PATH="openvla/openvla-7b"      # Pre-trained OpenVLA model
LORA_RANK=32                       # LoRA rank (higher = more parameters, better fit)

# Training hyperparameters
BATCH_SIZE=16                      # Batch size per GPU (reduce if OOM)
GRAD_ACCUMULATION_STEPS=1          # Gradient accumulation steps
LEARNING_RATE=5e-4                 # Learning rate
IMAGE_AUG=True                     # Enable image augmentation

# Hardware configuration
NUM_GPUS=1                         # Number of GPUs to use

# Logging
WANDB_PROJECT="openvla-droid"      # Weights & Biases project name
WANDB_ENTITY=""                    # Your W&B entity/username (leave empty to use default)
SAVE_STEPS=2500                    # Save checkpoint every N steps

# ============================================================================
# VALIDATION
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OpenVLA DROID Training Script ===${NC}"
echo ""

# Check if DATA_ROOT_DIR exists
if [ ! -d "${DATA_ROOT_DIR}" ]; then
    echo -e "${RED}Error: DATA_ROOT_DIR does not exist: ${DATA_ROOT_DIR}${NC}"
    echo "Please set DATA_ROOT_DIR to the parent directory containing your droid/ folder"
    exit 1
fi

# Check if dataset exists
DATASET_PATH="${DATA_ROOT_DIR}/${DATASET_NAME}"
if [ ! -d "${DATASET_PATH}" ]; then
    echo -e "${RED}Error: Dataset not found: ${DATASET_PATH}${NC}"
    echo ""
    echo "Available datasets in ${DATA_ROOT_DIR}:"
    ls -d ${DATA_ROOT_DIR}/*/ 2>/dev/null || echo "  (none found)"
    echo ""
    echo "To download DROID dataset, run:"
    echo -e "  ${YELLOW}./download_droid.sh full ${DATA_ROOT_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset found: ${DATASET_PATH}${NC}"

# Create output directories
mkdir -p "${RUN_ROOT_DIR}"
mkdir -p "${ADAPTER_TMP_DIR}"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo ""
echo -e "${BLUE}Training Configuration:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Model:              ${YELLOW}${VLA_PATH}${NC}"
echo -e "Dataset:            ${YELLOW}${DATASET_NAME}${NC} (${DATASET_PATH})"
echo -e "Output directory:   ${YELLOW}${RUN_ROOT_DIR}${NC}"
echo ""
echo -e "LoRA rank:          ${YELLOW}${LORA_RANK}${NC}"
echo -e "Batch size:         ${YELLOW}${BATCH_SIZE}${NC}"
echo -e "Grad accumulation:  ${YELLOW}${GRAD_ACCUMULATION_STEPS}${NC}"
echo -e "Learning rate:      ${YELLOW}${LEARNING_RATE}${NC}"
echo -e "Image augmentation: ${YELLOW}${IMAGE_AUG}${NC}"
echo ""
echo -e "Number of GPUs:     ${YELLOW}${NUM_GPUS}${NC}"
echo -e "Save every:         ${YELLOW}${SAVE_STEPS}${NC} steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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

TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} vla-scripts/finetune.py"
TRAIN_CMD="${TRAIN_CMD} --vla_path \"${VLA_PATH}\""
TRAIN_CMD="${TRAIN_CMD} --data_root_dir \"${DATA_ROOT_DIR}\""
TRAIN_CMD="${TRAIN_CMD} --dataset_name ${DATASET_NAME}"
TRAIN_CMD="${TRAIN_CMD} --run_root_dir \"${RUN_ROOT_DIR}\""
TRAIN_CMD="${TRAIN_CMD} --adapter_tmp_dir \"${ADAPTER_TMP_DIR}\""
TRAIN_CMD="${TRAIN_CMD} --lora_rank ${LORA_RANK}"
TRAIN_CMD="${TRAIN_CMD} --batch_size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --learning_rate ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --image_aug ${IMAGE_AUG}"
TRAIN_CMD="${TRAIN_CMD} --save_steps ${SAVE_STEPS}"

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

