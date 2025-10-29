#!/bin/bash
# Example script to extract and augment DROID dataset
# This creates a memory-efficient JSON dataset with deduplicated action histories

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DROID Dataset Extraction Example ===${NC}"
echo ""

# Configuration
DROID_DIR="/root/tfrecords/droid_100/1.0.0"  # Path to DROID dataset
OUTPUT_JSON="droid_100_extracted.json"        # Output JSON file
IMAGES_FOLDER="droid_100_images"              # Output images folder
MAX_EPISODES=10                                # Limit for testing (set to None for all)

echo -e "${BLUE}Configuration:${NC}"
echo "  Dataset: ${DROID_DIR}"
echo "  Output JSON: ${OUTPUT_JSON}"
echo "  Images folder: ${IMAGES_FOLDER}"
echo "  Max episodes: ${MAX_EPISODES}"
echo "  Image size: 256x256 (resized)"
echo ""

# Check if dataset exists
if [ ! -d "${DROID_DIR}" ]; then
    echo -e "${RED}Error: DROID dataset not found at ${DROID_DIR}${NC}"
    echo ""
    echo "Please download DROID dataset first:"
    echo "  cd /root/openvla"
    echo "  ./download_droid.sh subset /root/tfrecords"
    exit 1
fi

echo -e "${GREEN}Starting extraction...${NC}"
echo ""

# Run extraction
python /root/openvla/augment_droid_dataset.py \
    --builder_dir "${DROID_DIR}" \
    --output_path "${OUTPUT_JSON}" \
    --images_folder "${IMAGES_FOLDER}" \
    --max_episodes ${MAX_EPISODES} \
    --window_before 6 \
    --window_after 3 \
    --use_exterior_camera

echo ""
echo -e "${GREEN}=== Extraction Complete ===${NC}"
echo ""
echo "Output files:"
echo "  JSON dataset: ${YELLOW}${OUTPUT_JSON}${NC}"
echo "  Images: ${YELLOW}${IMAGES_FOLDER}/${NC}"
echo ""
echo "To extract with rephrased instructions (if available):"
echo "  python augment_droid_dataset.py \\"
echo "    --builder_dir ${DROID_DIR} \\"
echo "    --output_path ${OUTPUT_JSON} \\"
echo "    --rephrases_json your_rephrases.json"
echo ""
echo "To extract all episodes (not just ${MAX_EPISODES}):"
echo "  python augment_droid_dataset.py \\"
echo "    --builder_dir ${DROID_DIR} \\"
echo "    --output_path ${OUTPUT_JSON} \\"
echo "    --use_exterior_camera"
echo ""
echo "To change image size (default 256x256):"
echo "  python augment_droid_dataset.py \\"
echo "    --builder_dir ${DROID_DIR} \\"
echo "    --output_path ${OUTPUT_JSON} \\"
echo "    --image_width 512 --image_height 512"

