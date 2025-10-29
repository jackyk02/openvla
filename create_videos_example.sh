#!/bin/bash
# Create videos for bottom 5 episodes

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Create Videos for Bottom 5 Episodes ===${NC}"
echo ""

# Configuration
SCORES_FILE="droid_episode_scores.json"
DATASET_FILE="droid_bridge_converted.json"
IMAGES_FOLDER="droid_extracted_images"
OUTPUT_DIR="droid_bottom_5_videos"
NUM_EPISODES=5
FPS=10

echo -e "${BLUE}Configuration:${NC}"
echo "  Scores file: ${SCORES_FILE}"
echo "  Dataset: ${DATASET_FILE}"
echo "  Images folder: ${IMAGES_FOLDER}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of episodes: ${NUM_EPISODES}"
echo "  FPS: ${FPS}"
echo ""

# Check files exist
if [ ! -f "${SCORES_FILE}" ]; then
    echo -e "${RED}Error: Scores file not found: ${SCORES_FILE}${NC}"
    echo ""
    echo "Please compute episode scores first:"
    echo "  ./compute_droid_scores_example.sh"
    exit 1
fi

if [ ! -f "${DATASET_FILE}" ]; then
    echo -e "${RED}Error: Dataset not found: ${DATASET_FILE}${NC}"
    exit 1
fi

if [ ! -d "${IMAGES_FOLDER}" ]; then
    echo -e "${RED}Error: Images folder not found: ${IMAGES_FOLDER}${NC}"
    exit 1
fi

echo -e "${GREEN}Creating videos...${NC}"
echo ""

# Run video creation
python /root/openvla/create_episode_videos.py \
    --scores "${SCORES_FILE}" \
    --dataset "${DATASET_FILE}" \
    --images "${IMAGES_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_episodes ${NUM_EPISODES} \
    --mode bottom \
    --fps ${FPS}

echo ""
echo -e "${GREEN}=== Video Creation Complete ===${NC}"
echo ""
echo "Videos saved in: ${YELLOW}${OUTPUT_DIR}/${NC}"
echo ""
echo "To view the videos:"
echo "  ls -lh ${OUTPUT_DIR}/"
echo ""
echo "To create videos for top 5 episodes instead:"
echo "  python create_episode_videos.py --mode top"

