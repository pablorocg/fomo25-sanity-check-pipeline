#!/bin/bash
# This script builds an Apptainer container with sudo privileges

# Exit on error
set -e

# Get script directory
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

# Construct full paths
DEF_PATH="${SCRIPT_DIR}/src/${DEF_FILE}"
OUTPUT_PATH="${SCRIPT_DIR}/apptainer_images/${CONTAINER_NAME}.sif"

# Validate definition file exists
if [ ! -f "$DEF_PATH" ]; then
    echo "Error: Apptainer definition file not found at ${DEF_PATH}"
    exit 1
fi

# Ensure output directory exists
mkdir -p "${SCRIPT_DIR}/apptainer_images"

# Build container with sudo
echo "Building Apptainer container with sudo privileges: ${CONTAINER_NAME}.sif"
sudo apptainer build "$OUTPUT_PATH" "$DEF_PATH"

# Fix permissions so the container is usable by the current user
echo "Setting permissions on container..."
sudo chown $(whoami) "$OUTPUT_PATH"

echo "Container built successfully: ${OUTPUT_PATH}"
echo "File size: $(du -h "$OUTPUT_PATH" | cut -f1)"

# Verify container works
echo "Verifying container..."
apptainer exec "$OUTPUT_PATH" echo "Container verification successful"

echo "Build complete! The container image is ready to be transferred to other systems."