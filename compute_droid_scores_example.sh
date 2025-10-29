#!/bin/bash
# Example script to compute VLA-CLIP scores for DROID episodes

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Compute VLA-CLIP Scores for DROID Episodes ===${NC}"
echo ""

# Configuration
# UPDATE THIS PATH to your VLA-CLIP ensemble checkpoint:
MERGED_CHECKPOINT="/root/vla-clip/bridge_verifier/downloads/ensemble_789_trainable_only.pt"

# These paths should work if you ran the extraction and conversion scripts:
DROID_DATASET="droid_bridge_converted.json"
IMAGES_FOLDER="droid_extracted_images"
OUTPUT_JSON="droid_episode_scores.json"

# Set MAX_EPISODES to limit processing (for testing)
# Set to empty string ("") to process all episodes
MAX_EPISODES=100

echo -e "${BLUE}Configuration:${NC}"
echo "  Checkpoint: ${MERGED_CHECKPOINT}"
echo "  Dataset: ${DROID_DATASET}"
echo "  Images: ${IMAGES_FOLDER}"
echo "  Output: ${OUTPUT_JSON}"
if [ -n "$MAX_EPISODES" ]; then
    echo "  Max episodes: ${MAX_EPISODES} (testing mode)"
else
    echo "  Max episodes: All"
fi
echo ""

# Check if checkpoint exists
if [ ! -f "${MERGED_CHECKPOINT}" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${MERGED_CHECKPOINT}${NC}"
    echo ""
    echo "Please update MERGED_CHECKPOINT path in this script to point to your"
    echo "VLA-CLIP ensemble checkpoint (merged_checkpoint.pt)"
    exit 1
fi

# Check if dataset exists
if [ ! -f "${DROID_DATASET}" ]; then
    echo -e "${RED}Error: Dataset not found: ${DROID_DATASET}${NC}"
    echo ""
    echo "Please convert DROID to Bridge scale first:"
    echo "  ./convert_droid_example.sh"
    exit 1
fi

# Check if images folder exists
if [ ! -d "${IMAGES_FOLDER}" ]; then
    echo -e "${RED}Error: Images folder not found: ${IMAGES_FOLDER}${NC}"
    echo ""
    echo "Please extract DROID dataset first:"
    echo "  ./extract_droid_example.sh"
    exit 1
fi

echo -e "${GREEN}Starting score computation...${NC}"
echo ""

# Build command
CMD="python /root/openvla/compute_droid_episode_scores.py \
    --merged_checkpoint \"${MERGED_CHECKPOINT}\" \
    --droid_dataset \"${DROID_DATASET}\" \
    --images_folder \"${IMAGES_FOLDER}\" \
    --output \"${OUTPUT_JSON}\""

# Add max_episodes if set
if [ -n "$MAX_EPISODES" ]; then
    CMD="${CMD} --max_episodes ${MAX_EPISODES}"
fi

# Run computation
eval ${CMD}

echo ""
echo -e "${GREEN}=== Score Computation Complete ===${NC}"
echo ""
echo "Output file: ${YELLOW}${OUTPUT_JSON}${NC}"
echo ""
echo "The results contain:"
echo "  - Per-episode mean VLA-CLIP scores"
echo "  - Overall statistics across all episodes"
echo "  - Individual sample scores for each episode"
echo ""
echo "To process all episodes (not just ${MAX_EPISODES}):"
echo "  Edit this script and set MAX_EPISODES=\"\""

