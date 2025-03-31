#!/bin/bash
set -e
# Resolve script directory reliably
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_NAME="fomo25-baseline-container"
DEF_FILE="Apptainer.def"

# Parse arguments
if [ $# -ge 1 ]; then
    CONTAINER_NAME="$1"
fi
if [ $# -ge 2 ]; then
    DEF_FILE="$2"
fi

# Construct full paths - fixed to match actual directory structure
DEF_PATH="${SCRIPT_DIR}/src/${DEF_FILE}"
OUTPUT_PATH="${SCRIPT_DIR}/apptainer_images/${CONTAINER_NAME}.sif"

# Validate definition file exists
if [ ! -f "$DEF_PATH" ]; then
    echo "Error: Apptainer definition file not found at ${DEF_PATH}"
    exit 1
fi

# Ensure output directory exists
mkdir -p "${SCRIPT_DIR}/apptainer_images"

# Build container
echo "Building Apptainer container: ${CONTAINER_NAME}.sif"
apptainer build --fakeroot --force "$OUTPUT_PATH" "$DEF_PATH"
echo "Container built successfully: ${OUTPUT_PATH}"