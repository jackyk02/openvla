#!/bin/bash
# Example script to convert DROID actions to Bridge scale using token-based conversion

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DROID to Bridge Action Conversion ===${NC}"
echo ""

# Configuration
INPUT_JSON="droid_extracted.json"
OUTPUT_JSON="droid_bridge_converted.json"

echo -e "${BLUE}Configuration:${NC}"
echo "  Input:  ${INPUT_JSON}"
echo "  Output: ${OUTPUT_JSON}"
echo "  Gripper: Binarized from DROID (<0.5→0.0 open, ≥0.5→1.0 closed)"
echo ""

# Check if input file exists
if [ ! -f "${INPUT_JSON}" ]; then
    echo -e "${RED}Error: Input file not found: ${INPUT_JSON}${NC}"
    echo ""
    echo "Please extract DROID dataset first:"
    echo "  ./extract_droid_example.sh"
    exit 1
fi

echo -e "${GREEN}Starting conversion...${NC}"
echo ""

# Run conversion
python /root/openvla/convert_droid_to_bridge.py \
    --input "${INPUT_JSON}" \
    --output "${OUTPUT_JSON}"

echo ""
echo -e "${GREEN}=== Conversion Complete ===${NC}"
echo ""
echo "Output file: ${YELLOW}${OUTPUT_JSON}${NC}"
echo ""
echo "The converted dataset contains:"
echo "  - Action dimension: 7 (6D Bridge-scale position + 1D binarized gripper)"
echo "  - Gripper: Binarized from DROID values (<0.5→0.0, ≥0.5→1.0)"
echo "  - Conversion method: Token-based (DROID → tokens → Bridge)"
echo ""
echo "Next steps:"
echo "1. Use this file for training on Bridge-scale models"
echo "2. Gripper values are binarized from original DROID gripper states"
echo "3. See conversion metadata in '_metadata.conversion' for details"

