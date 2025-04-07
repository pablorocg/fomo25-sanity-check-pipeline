#!/usr/bin/env bash
# Simplified script to build a container for the FOMO25 Challenge

# Exit on error
set -e

# Colors for messages
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error codes
ERROR_DOCKER_NOT_FOUND=1
ERROR_DOCKERFILE_NOT_FOUND=2
ERROR_BUILD_FAILED=3
ERROR_TEST_FAILED=4
ERROR_CONVERT_FAILED=5

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKERFILE_PATH="${SCRIPT_DIR}/src/Dockerfile"
IMAGES_DIR="${SCRIPT_DIR}/apptainer-images"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
IMAGE_NAME="fomo25sanitycheckpipeline"
IMAGE_TAG="latest"
LOG_FILE="${SCRIPT_DIR}/build_$(date +"%Y%m%d-%H%M%S").log"

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    
    case $level in
        "INFO")
            color=$BLUE
            ;;
        "SUCCESS")
            color=$GREEN
            ;;
        "WARNING")
            color=$YELLOW
            ;;
        "ERROR")
            color=$RED
            ;;
    esac
    
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] ${message}${NC}" | tee -a "$LOG_FILE"
}

# Show help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --name NAME        Container name (default: fomo25sanitycheckpipeline)"
    echo "  --tag TAG          Container tag (default: latest)"
    echo "  --build-only       Only build the Docker image"
    echo "  --convert-only     Only convert the Docker image to Apptainer"
    echo "  --test-only        Only test the existing Docker image"
    echo "  --help             Show this help"
}

# Function to parse arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --convert-only)
                CONVERT_ONLY=true
                shift
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_message "ERROR" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Function to check environment requirements
check_requirements() {
    log_message "INFO" "Checking requirements..."
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        log_message "ERROR" "Docker is not installed. Please install Docker first."
        log_message "ERROR" "On Ubuntu/Debian: sudo apt-get install docker.io"
        log_message "ERROR" "On CentOS/RHEL: sudo yum install docker"
        log_message "ERROR" "On macOS: Install Docker Desktop from https://www.docker.com/products/docker-desktop"
        exit $ERROR_DOCKER_NOT_FOUND
    else
        local docker_version=$(docker --version | cut -d ' ' -f3 | tr -d ',')
        log_message "INFO" "Found Docker version $docker_version"
    fi
    
    # Check if Dockerfile exists
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        log_message "ERROR" "Dockerfile not found at $DOCKERFILE_PATH"
        log_message "ERROR" "Make sure you have a Dockerfile in the src/ directory"
        log_message "ERROR" "The full path should be: $DOCKERFILE_PATH"
        exit $ERROR_DOCKERFILE_NOT_FOUND
    fi
    
    # Check for source directory
    if [ ! -d "${SCRIPT_DIR}/src" ]; then
        log_message "ERROR" "Could not find 'src' directory. Please create this directory with your code."
        log_message "ERROR" "The source directory should be at: ${SCRIPT_DIR}/src"
        log_message "ERROR" "This directory should contain your code and Dockerfile"
        exit $ERROR_SRC_DIR_NOT_FOUND
    fi
}

# Function to prepare directories
prepare_directories() {
    log_message "INFO" "Preparing directories..."
    
    # Create required directories if they don't exist
    mkdir -p "${SCRIPT_DIR}/test/input" 2>/dev/null || true
    mkdir -p "${SCRIPT_DIR}/test/output" 2>/dev/null || true
    mkdir -p "$IMAGES_DIR" 2>/dev/null || true
    
    # Ensure proper permissions for test directories
    chmod -R 755 "${SCRIPT_DIR}/test/input" 2>/dev/null || true
    chmod -R 777 "${SCRIPT_DIR}/test/output" 2>/dev/null || true
    
    if [ ! -d "${SCRIPT_DIR}/test/input" ]; then
        log_message "ERROR" "Failed to create input directory: ${SCRIPT_DIR}/test/input"
        log_message "ERROR" "Check if you have write permissions in the parent directory"
        exit 1
    fi
    
    if [ ! -d "${SCRIPT_DIR}/test/output" ]; then
        log_message "ERROR" "Failed to create output directory: ${SCRIPT_DIR}/test/output"
        log_message "ERROR" "Check if you have write permissions in the parent directory"
        exit 1
    fi
}

# Function to build Docker image
build_docker_image() {
    if [ "$TEST_ONLY" = true ] || [ "$CONVERT_ONLY" = true ]; then
        return 0
    fi
    
    local full_tag="${IMAGE_NAME}:${IMAGE_TAG}"
    log_message "INFO" "Building Docker image: $full_tag"
    
    if docker build -f $DOCKERFILE_PATH --tag $full_tag $SCRIPT_DIR 2>&1 | tee -a "$LOG_FILE"; then
        log_message "SUCCESS" "Docker image built successfully: $full_tag"
    else
        log_message "ERROR" "Docker build failed."
        log_message "ERROR" "Common issues:"
        log_message "ERROR" "1. In your Dockerfile, make sure to use correct paths. The build context is the root directory."
        log_message "ERROR" "2. For paths in COPY commands, use 'src/file' instead of just 'file'."
        log_message "ERROR" "3. Make sure all required files are present in the src directory."
        log_message "ERROR" "4. Check for syntax errors in your Dockerfile."
        log_message "ERROR" "5. Ensure you have internet connectivity for pulling base images."
        log_message "ERROR" "Check the full build log at: $LOG_FILE"
        exit $ERROR_BUILD_FAILED
    fi
}

# Function to test Docker image
test_docker_image() {
    if [ "$BUILD_ONLY" = true ] || [ "$CONVERT_ONLY" = true ]; then
        return 0
    fi
    
    local full_tag="${IMAGE_NAME}:${IMAGE_TAG}"
    log_message "INFO" "Testing Docker image: $full_tag"
    
    # Clean output directory and ensure proper permissions
    log_message "INFO" "Preparing test environment..."
    rm -rf "${SCRIPT_DIR}/test/output"/* 2>/dev/null || true
    mkdir -p "${SCRIPT_DIR}/test/output" 2>/dev/null || true
    chmod -R 777 "${SCRIPT_DIR}/test/output" 2>/dev/null || true
    
    # Check if input directory has test data
    if [ ! "$(ls -A "${SCRIPT_DIR}/test/input" 2>/dev/null)" ]; then
        log_message "WARNING" "Input directory is empty. No test data found."
        log_message "WARNING" "Place test data in: ${SCRIPT_DIR}/test/input"
        log_message "WARNING" "Continuing with empty input directory..."
    fi
    
    # Run container using the specified command
    log_message "INFO" "Running inference in container..."
    
    # Using the command provided in the previous message
    if docker run --rm --gpus all \
        --volume $(pwd)/test/input:/input:ro \
        --volume $(pwd)/test/output:/output \
        --user $(id -u):$(id -g) \
        -it \
        $full_tag 2>&1 | tee -a "$LOG_FILE"; then
        
        log_message "SUCCESS" "Inference test completed successfully"
        # Check if any output files were generated
        local output_files=$(find "$OUTPUT_DIR" -type f | wc -l)
        if [[ $output_files -gt 0 ]]; then
            log_message "SUCCESS" "Found $output_files output files"
        else
            log_message "WARNING" "No output files generated. This might be due to:"
            log_message "WARNING" "1. Empty input directory or missing test data"
            log_message "WARNING" "2. Issues with your model's inference code"
            log_message "WARNING" "3. Incorrect input/output paths in your code (should use /input and /output)"
        fi
    else
        log_message "ERROR" "Inference test failed. Possible issues:"
        log_message "ERROR" "1. GPU issues - Make sure your host has NVIDIA GPUs with proper drivers"
        log_message "ERROR" "2. NVIDIA Docker runtime - Check that nvidia-docker is installed"
        log_message "ERROR" "3. Code errors - Check your inference code inside the container"
        log_message "ERROR" "4. Missing dependencies - Ensure all requirements are installed in the Dockerfile"
        log_message "ERROR" "5. Permission issues - The container may not have write access to the output directory"
        log_message "ERROR" "For more details, check the log at: $LOG_FILE"
        exit $ERROR_TEST_FAILED
    fi
}

# Function to convert Docker image to Apptainer/Singularity
convert_to_apptainer() {
    if [ "$BUILD_ONLY" = true ] || [ "$TEST_ONLY" = true ]; then
        return 0
    fi
    
    local full_tag="${IMAGE_NAME}:${IMAGE_TAG}"
    log_message "INFO" "Converting to Apptainer/Singularity container..."
    
    # Detect which command to use
    local container_cmd=""
    if command -v apptainer &> /dev/null; then
        container_cmd="apptainer"
    elif command -v singularity &> /dev/null; then
        container_cmd="singularity"
    fi
    
    # If no Apptainer/Singularity is installed, just save Docker image
    if [ -z "$container_cmd" ]; then
        log_message "INFO" "Neither apptainer nor singularity is installed. Saving Docker image only."
        
        # Save Docker image as file
        local output_tar="${IMAGES_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
        log_message "INFO" "Saving Docker image as $output_tar..."
        
        if docker save "$full_tag" | gzip -c > "$output_tar"; then
            log_message "SUCCESS" "Image saved as: $output_tar"
            log_message "INFO" "To convert to Apptainer/Singularity later, run:"
            log_message "INFO" "  sudo apptainer build ${IMAGE_NAME}-${IMAGE_TAG}.sif docker-archive://${output_tar}"
        else
            log_message "ERROR" "Failed to save Docker image. Possible issues:"
            log_message "ERROR" "1. Disk space - Ensure you have enough free disk space"
            log_message "ERROR" "2. Permissions - Make sure you have write access to ${IMAGES_DIR}"
            log_message "ERROR" "3. Docker image - Check if the image exists: docker images | grep ${IMAGE_NAME}"
            exit $ERROR_CONVERT_FAILED
        fi
    else
        # First save the Docker image to a tar file
        local output_tar="${SCRIPT_DIR}/${IMAGE_NAME}-temp.tar"
        log_message "INFO" "Saving Docker image as $output_tar..."
        
        if ! docker save "$full_tag" -o "$output_tar"; then
            log_message "ERROR" "Failed to save Docker image. Possible issues:"
            log_message "ERROR" "1. Disk space - Ensure you have enough free disk space"
            log_message "ERROR" "2. Permissions - Make sure you have write access to ${SCRIPT_DIR}"
            log_message "ERROR" "3. Docker image - Check if the image exists: docker images | grep ${IMAGE_NAME}"
            exit $ERROR_CONVERT_FAILED
        fi
        
        # Create Apptainer container
        local output_sif="${IMAGES_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.sif"
        
        log_message "INFO" "Using $container_cmd for conversion..."
        
        if sudo "$container_cmd" build "$output_sif" "docker-archive://${output_tar}" 2>&1 | tee -a "$LOG_FILE"; then
            log_message "SUCCESS" "$container_cmd container created: $output_sif"
        else
            log_message "ERROR" "Failed to convert Docker image to $container_cmd format. Possible issues:"
            log_message "ERROR" "1. Sudo permissions - Make sure you have sudo privileges"
            log_message "ERROR" "2. $container_cmd version - Some older versions have compatibility issues"
            log_message "ERROR" "3. Disk space - Ensure you have enough free disk space"
            log_message "ERROR" "4. Container conflicts - Try with a clean build environment"
            exit $ERROR_CONVERT_FAILED
        fi
        
        # Cleanup
        log_message "INFO" "Cleaning up temporary files..."
        rm -f "$output_tar"
    fi
}

# Function to print final summary
print_summary() {
    log_message "INFO" "=============================================="
    log_message "SUCCESS" "Process completed successfully"
    log_message "INFO" "=============================================="
    
    # Show final instructions
    log_message "INFO" "To submit to the challenge:"
    
    local output_sif="${IMAGES_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.sif"
    local output_tar="${IMAGES_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
    
    if [ -f "$output_sif" ]; then
        log_message "INFO" "- Use the file: $output_sif"
    elif [ -f "$output_tar" ]; then
        log_message "INFO" "- First convert: $output_tar to .sif format"
        log_message "INFO" "- Command: sudo apptainer build ${IMAGE_NAME}-${IMAGE_TAG}.sif docker-archive://${output_tar}"
    else
        log_message "INFO" "- Run again with --convert-only to create the .sif file"
    fi
    
    log_message "INFO" "Build log file: $LOG_FILE"
}

# Main function to execute the script
main() {
    # Initialize log file
    echo "# FOMO25 Build Log - $(date)" > "$LOG_FILE"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Print header
    log_message "INFO" "=============================================="
    log_message "INFO" "FOMO25 Challenge - Container Builder"
    log_message "INFO" "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    log_message "INFO" "=============================================="
    
    # Run steps
    check_requirements
    prepare_directories
    build_docker_image
    test_docker_image
    convert_to_apptainer
    print_summary
    
    exit 0
}

# Trap for cleaning up on interruption
trap 'log_message "ERROR" "Process interrupted"; exit 1' INT TERM

# Execute main function with all provided arguments
main "$@"