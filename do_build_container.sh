#!/bin/bash
# Build script for FOMO25 Apptainer/Singularity containers

# Color codes for messages
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default values
CONTAINER_NAME="fomo25-container"
DEF_FILE="${SCRIPT_DIR}/Apptainer.def"
IMAGES_DIR="${SCRIPT_DIR}/apptainer-images"
CONFIG_FILE="${SCRIPT_DIR}/config.yml"
CONTAINER_CMD="apptainer"
SOURCE_DIR="${SCRIPT_DIR}/src"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# Print message with color
msg() {
  local color=$1
  local text=$2
  local emoji=$3
  echo -e "${color}${emoji} ${text}${NC}"
}

# Parse YAML configuration
parse_yaml() {
    local yaml_file=$1
    
    [ -r $yaml_file ] || return 1
    
    msg "$BLUE" "Reading configuration from $yaml_file" "📄"
    
    # Read container settings
    if grep -q "container:" "$yaml_file"; then
        # Container name
        local container_name=$(grep -A 10 "container:" "$yaml_file" | grep "name:" | head -n1 | sed -e "s/.*name: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$container_name" ]; then
            CONTAINER_NAME="$container_name"
            msg "$BLUE" "Using container name: $CONTAINER_NAME" "📦"
        fi
        
        # Container command
        local cmd=$(grep -A 10 "container:" "$yaml_file" | grep "command:" | head -n1 | sed -e "s/.*command: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$cmd" ]; then
            CONTAINER_CMD="$cmd"
            msg "$BLUE" "Using container command: $CONTAINER_CMD" "🔧"
        fi
    fi
    
    # Read directory settings
    if grep -q "directories:" "$yaml_file"; then
        # Input directory
        local input_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "input:" | head -n1 | sed -e "s/.*input: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$input_dir" ]; then
            INPUT_DIR="${SCRIPT_DIR}/${input_dir}"
            msg "$BLUE" "Using input directory: $INPUT_DIR" "📂"
        fi
        
        # Output directory
        local output_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "output:" | head -n1 | sed -e "s/.*output: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$output_dir" ]; then
            OUTPUT_DIR="${SCRIPT_DIR}/${output_dir}"
            msg "$BLUE" "Using output directory: $OUTPUT_DIR" "📂"
        fi
        
        # Containers directory
        local containers_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "containers:" | head -n1 | sed -e "s/.*containers: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$containers_dir" ]; then
            IMAGES_DIR="${SCRIPT_DIR}/${containers_dir}"
            msg "$BLUE" "Using containers directory: $IMAGES_DIR" "📂"
        fi
    fi
    
    # Read build settings
    if grep -q "build:" "$yaml_file"; then
        # Source directory
        local source_dir=$(grep -A 10 "build:" "$yaml_file" | grep "source:" | head -n1 | sed -e "s/.*source: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$source_dir" ]; then
            SOURCE_DIR="${SCRIPT_DIR}/${source_dir}"
            msg "$BLUE" "Using source directory: $SOURCE_DIR" "📂"
        fi
        
        # Definition file
        local def_file=$(grep -A 10 "build:" "$yaml_file" | grep "definition:" | head -n1 | sed -e "s/.*definition: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$def_file" ]; then
            DEF_FILE="${SCRIPT_DIR}/${def_file}"
            msg "$BLUE" "Using definition file: $DEF_FILE" "📄"
        fi
    fi
    
    # Set container path
    CONTAINER_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
}

# Show help
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name NAME      Container name (default: fomo25-container)"
  echo "  -d, --def FILE       Definition file (default: ./Apptainer.def)"
  echo "  -o, --output DIR     Output directory for containers (default: ./apptainer-images)"
  echo "  -c, --config FILE    Config file path (default: ./config.yml)"
  echo "  --cmd COMMAND        Custom Apptainer/Singularity command (default: apptainer)"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  $0 -n custom-model                  # Build with custom name"
  echo "  $0 -d custom.def -n my-model        # Build with custom definition"
  echo "  $0 -c my-config.yml                 # Use custom config file"
}

# Parse arguments
parse_args() {
  # First check if config file is specified
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "-c" || "${!i}" == "--config" ]]; then
      next=$((i+1))
      if [[ $next -le $# ]]; then
        CONFIG_FILE="${!next}"
      fi
    fi
  done
  
  # Parse config file first (if exists)
  if [ -f "$CONFIG_FILE" ]; then
    parse_yaml "$CONFIG_FILE"
  fi
  
  # Then parse command-line arguments (override config)
  while [[ $# -gt 0 ]]; do
    case $1 in
      -n|--name)
        CONTAINER_NAME="$2"
        shift 2
        ;;
      -d|--def)
        DEF_FILE="$2"
        shift 2
        ;;
      -o|--output)
        IMAGES_DIR="$2"
        shift 2
        ;;
      -c|--config)
        # Already handled above
        shift 2
        ;;
      --cmd)
        CONTAINER_CMD="$2"
        shift 2
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        msg "$RED" "Unknown option: $1" "⚠️"
        show_help
        exit 1
        ;;
    esac
  done
  
  # Update container path
  CONTAINER_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
}

# Create required directories
create_directories() {
  mkdir -p "$IMAGES_DIR"
  mkdir -p "$INPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  
  msg "$BLUE" "Directories created/verified" "📂"
}

# Check environment
check_env() {
  # Check for specified container command
  if command -v "$CONTAINER_CMD" &>/dev/null; then
    msg "$GREEN" "Using $CONTAINER_CMD" "✅"
  else
    msg "$YELLOW" "Command '$CONTAINER_CMD' not found, checking alternatives..." "⚠️"
    
    # Fallback to alternatives
    if command -v apptainer &>/dev/null; then
      CONTAINER_CMD="apptainer"
      msg "$GREEN" "Using Apptainer instead" "✅"
    elif command -v singularity &>/dev/null; then
      CONTAINER_CMD="singularity"
      msg "$GREEN" "Using Singularity instead" "✅"
    else
      msg "$RED" "Neither Apptainer nor Singularity found. Please install one of them." "❌"
      exit 1
    fi
  fi
  
  # Check definition file exists
  if [ ! -f "$DEF_FILE" ]; then
    msg "$RED" "Definition file not found: $DEF_FILE" "❌"
    exit 1
  fi
  
  # Check source directory exists
  if [ ! -d "$SOURCE_DIR" ]; then
    msg "$YELLOW" "Source directory not found: $SOURCE_DIR. Creating it..." "⚠️"
    mkdir -p "$SOURCE_DIR"
  fi
}

# Build container
build_container() {
  msg "$BLUE" "Building container: $CONTAINER_PATH" "🔨"
  
  # Remove existing container if it exists
  if [ -f "$CONTAINER_PATH" ]; then
    msg "$YELLOW" "Removing existing container: $CONTAINER_PATH" "🗑️"
    rm -f "$CONTAINER_PATH"
  fi
  
  # Build with proper error handling
  if "$CONTAINER_CMD" build "$CONTAINER_PATH" "$DEF_FILE"; then
    if [ -f "$CONTAINER_PATH" ]; then
      msg "$GREEN" "Container built successfully" "✅"
      msg "$BLUE" "Container location: $CONTAINER_PATH" "📦"
      return 0
    else
      msg "$RED" "Build failed: Container file not created" "❌"
      return 1
    fi
  else
    msg "$RED" "Build failed" "❌"
    return 1
  fi
}

# Main function
main() {
  msg "$BLUE" "Starting FOMO25 container build" "🚀"
  
  # Parse arguments
  parse_args "$@"
  
  # Create directories
  create_directories
  
  # Check environment
  check_env
  
  # Build container
  build_container
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    msg "$GREEN" "Build completed successfully" "✅"
  else
    msg "$RED" "Build failed with errors" "❌"
  fi
  
  return $exit_code
}

# Trap errors
trap 'msg "$RED" "An error occurred. Exiting..." "❌"; exit 1' ERR

# Execute main function
main "$@"