#!/bin/bash
# Script to download DROID dataset for OpenVLA training
# Usage: ./download_droid.sh [full|subset] [destination_directory]

set -e

# Default values
DATASET_TYPE="${1:-full}"  # Options: full, subset
DEST_DIR="${2:-./datasets}"  # Default to ./datasets

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DROID Dataset Download Script ===${NC}"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}Error: gsutil is not installed${NC}"
    echo ""
    echo "Please install Google Cloud SDK:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
    echo ""
    exit 1
fi

# Create destination directory
mkdir -p "${DEST_DIR}"
echo -e "Destination directory: ${YELLOW}${DEST_DIR}${NC}"
echo ""

# Download based on type
if [ "$DATASET_TYPE" == "full" ]; then
    echo -e "${GREEN}Downloading full DROID dataset (~1.7TB)...${NC}"
    echo -e "${YELLOW}This may take several hours depending on your internet connection.${NC}"
    echo ""
    
    # Create destination directory first
    mkdir -p "${DEST_DIR}/droid"
    gsutil -m cp -r gs://gresearch/robotics/droid/* "${DEST_DIR}/droid/"
    
    echo ""
    echo -e "${GREEN}✓ Full DROID dataset downloaded successfully!${NC}"
    echo -e "Location: ${YELLOW}${DEST_DIR}/droid${NC}"
    
elif [ "$DATASET_TYPE" == "subset" ]; then
    echo -e "${GREEN}Downloading DROID-100 subset (~2GB)...${NC}"
    echo -e "${YELLOW}This is a smaller dataset for testing/debugging.${NC}"
    echo ""
    
    # Create destination directory first
    mkdir -p "${DEST_DIR}/droid_100"
    gsutil -m cp -r gs://gresearch/robotics/droid_100/* "${DEST_DIR}/droid_100/"
    
    echo ""
    echo -e "${GREEN}✓ DROID-100 subset downloaded successfully!${NC}"
    echo -e "Location: ${YELLOW}${DEST_DIR}/droid_100${NC}"
    
else
    echo -e "${RED}Error: Invalid dataset type '${DATASET_TYPE}'${NC}"
    echo "Usage: $0 [full|subset] [destination_directory]"
    exit 1
fi

# Verify download
echo ""
echo -e "${GREEN}Verifying download...${NC}"

if [ "$DATASET_TYPE" == "full" ]; then
    DATASET_PATH="${DEST_DIR}/droid"
else
    DATASET_PATH="${DEST_DIR}/droid_100"
fi

if [ -d "${DATASET_PATH}" ]; then
    echo -e "${GREEN}✓ Dataset directory exists${NC}"
    
    # Check for dataset_info.json
    if find "${DATASET_PATH}" -name "dataset_info.json" -type f | grep -q .; then
        echo -e "${GREEN}✓ Dataset metadata found${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: dataset_info.json not found${NC}"
    fi
    
    # Show disk usage
    DATASET_SIZE=$(du -sh "${DATASET_PATH}" | cut -f1)
    echo -e "Dataset size: ${YELLOW}${DATASET_SIZE}${NC}"
else
    echo -e "${RED}✗ Error: Dataset directory not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Download Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Set up OpenVLA environment (if not already done)"
echo "2. Run training with:"
if [ "$DATASET_TYPE" == "full" ]; then
    echo -e "   ${YELLOW}torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\${NC}"
    echo -e "   ${YELLOW}  --vla_path \"openvla/openvla-7b\" \\${NC}"
    echo -e "   ${YELLOW}  --data_root_dir ${DEST_DIR} \\${NC}"
    echo -e "   ${YELLOW}  --dataset_name droid \\${NC}"
    echo -e "   ${YELLOW}  --run_root_dir ./runs/droid \\${NC}"
    echo -e "   ${YELLOW}  --adapter_tmp_dir ./adapters/droid \\${NC}"
    echo -e "   ${YELLOW}  --lora_rank 32 --batch_size 16 --learning_rate 5e-4${NC}"
else
    echo -e "   ${YELLOW}(Similar command, but with dataset_name droid_100)${NC}"
fi
echo ""
echo -e "For more details, see: ${YELLOW}DROID_TRAINING_GUIDE.md${NC}"

