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
CONTAINER_CMD_PATH=""

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
    
    # Read the config file for container path
    if grep -q "container:" "$yaml_file"; then
        # Container path
        local container_path=$(grep -A 5 "container:" "$yaml_file" | grep "path:" | head -n1 | sed -e "s/.*path: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$container_path" ]; then
            if [[ "$container_path" == ./* ]]; then
                # Relative path
                CONTAINER_PATH="${SCRIPT_DIR}/${container_path#./}"
                
                # Extract directory from path
                IMAGES_DIR=$(dirname "$CONTAINER_PATH")
                
                # Extract name from path
                if [[ "$container_path" == *.sif ]]; then
                    CONTAINER_NAME=$(basename "$container_path" .sif)
                fi
            else
                # Absolute path
                CONTAINER_PATH="$container_path"
                
                # Extract directory from path
                IMAGES_DIR=$(dirname "$CONTAINER_PATH")
                
                # Extract name from path
                if [[ "$container_path" == *.sif ]]; then
                    CONTAINER_NAME=$(basename "$container_path" .sif)
                fi
            fi
            
            msg "$BLUE" "Using container path: $CONTAINER_PATH" "📦"
        fi
        
        # Command path
        local cmd_path=$(grep -A 5 "container:" "$yaml_file" | grep "command:" | head -n1 | sed -e "s/.*command: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$cmd_path" ]; then
            CONTAINER_CMD_PATH="$cmd_path"
            msg "$BLUE" "Using custom command: $CONTAINER_CMD_PATH" "🔧"
        fi
    fi
}

# Show help
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name NAME      Container name (default: fomo25-container)"
  echo "  -d, --def FILE       Definition file (default: ./Apptainer.def)"
  echo "  -o, --output DIR     Output directory for containers (default: ./apptainer-images)"
  echo "  -c, --config FILE    Config file path (default: ./config.yml)"
  echo "  --cmd PATH           Custom Apptainer/Singularity command path"
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
        CONTAINER_CMD_PATH="$2"
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
  
  # Set container path if not already set by config
  if [ -z "$CONTAINER_PATH" ]; then
    CONTAINER_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
  fi
}

# Check environment
check_env() {
  # Check for Apptainer/Singularity
  CONTAINER_CMD=""
  
  if [ ! -z "$CONTAINER_CMD_PATH" ]; then
    # Custom command path specified
    if [ -x "$CONTAINER_CMD_PATH" ]; then
      CONTAINER_CMD="$CONTAINER_CMD_PATH"
      msg "$GREEN" "Using custom container command: $CONTAINER_CMD" "✅"
    elif command -v "$CONTAINER_CMD_PATH" &>/dev/null; then
      CONTAINER_CMD="$CONTAINER_CMD_PATH"
      msg "$GREEN" "Found custom container command: $CONTAINER_CMD" "✅"
    else
      msg "$RED" "Custom container command not found: $CONTAINER_CMD_PATH" "❌"
      exit 1
    fi
  elif command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
    msg "$GREEN" "Using Apptainer" "✅"
  elif command -v singularity &>/dev/null; then
    CONTAINER_CMD="singularity"
    msg "$GREEN" "Using Singularity" "✅"
  else
    msg "$RED" "Neither Apptainer nor Singularity found. Please install one of them." "❌"
    exit 1
  fi
  
  # Check definition file exists
  if [ ! -f "$DEF_FILE" ]; then
    msg "$RED" "Definition file not found: $DEF_FILE" "❌"
    exit 1
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p "$IMAGES_DIR"
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