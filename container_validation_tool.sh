#!/bin/bash
# FOMO25 Unified Container Build and Validation Script

set -e  # Exit on error

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
CONFIG_FILE="${SCRIPT_DIR}/container_config.yml"
CONTAINER_CMD_PATH=""
USE_GPU=true
USE_SUDO=false
ONLY_BUILD=false
ONLY_VALIDATE=false
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
RESULT_FILE="${SCRIPT_DIR}/validation_result.json"

# Print message with color
msg() {
  local color=$1
  local text=$2
  local emoji=$3
  echo -e "${color}${emoji} ${text}${NC}"
}

# Error handling function
handle_error() {
  local exit_code=$1
  local error_message=$2
  
  if [ $exit_code -ne 0 ]; then
    msg "$RED" "$error_message" "❌"
    
    # Additional troubleshooting guidance
    case "$error_message" in
      *"command not found"*)
        msg "$YELLOW" "Ensure Apptainer/Singularity is installed and in your PATH" "💡"
        msg "$YELLOW" "Installation guide: https://apptainer.org/docs/admin/main/installation.html" "💡"
        ;;
      *"definition file"*)
        msg "$YELLOW" "Check that your Apptainer.def file exists and is correctly formatted" "💡"
        msg "$YELLOW" "Reference the example in submission-guide.md" "💡"
        ;;
      *"permission denied"*)
        if $USE_SUDO; then
          msg "$YELLOW" "Try running with --no-sudo option if you have user namespace enabled" "💡"
        else
          msg "$YELLOW" "Try running with sudo or enable user namespaces" "💡"
          msg "$YELLOW" "See: https://apptainer.org/docs/admin/main/user_namespace.html" "💡"
        fi
        ;;
      *"Container not found"*)
        msg "$YELLOW" "Run build first or specify the correct container path" "💡"
        ;;
      *"No space left on device"*)
        msg "$YELLOW" "Free up disk space or set APPTAINER_TMPDIR to a different partition" "💡"
        msg "$YELLOW" "Example: export APPTAINER_TMPDIR=/path/with/space" "💡"
        ;;
    esac
    
    exit $exit_code
  fi
}

# Parse YAML configuration
parse_yaml() {
    local yaml_file=$1
    
    [ -r $yaml_file ] || return 1
    
    msg "$BLUE" "Reading configuration from $yaml_file" "📄"
    
    # Read the config file for container configuration
    if grep -q "container:" "$yaml_file"; then
        # Container name
        local container_name=$(grep -A 10 "container:" "$yaml_file" | grep "name:" | head -n1 | sed -e "s/.*name: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$container_name" ]; then
            CONTAINER_NAME="$container_name"
            msg "$BLUE" "Using container name: $CONTAINER_NAME" "📦"
        fi
        
        # Container path
        local container_path=$(grep -A 10 "container:" "$yaml_file" | grep "path:" | head -n1 | sed -e "s/.*path: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$container_path" ]; then
            if [[ "$container_path" == ./* ]]; then
                # Relative path
                CONTAINER_PATH="${SCRIPT_DIR}/${container_path#./}"
            else
                # Absolute path
                CONTAINER_PATH="$container_path"
            fi
            
            # Extract directory from path
            IMAGES_DIR=$(dirname "$CONTAINER_PATH")
            
            # Extract name from path if it has .sif extension
            if [[ "$container_path" == *.sif ]]; then
                CONTAINER_NAME=$(basename "$container_path" .sif)
            fi
            
            msg "$BLUE" "Using container path: $CONTAINER_PATH" "📦"
        fi
        
        # Command path
        local cmd_path=$(grep -A 10 "container:" "$yaml_file" | grep "command:" | head -n1 | sed -e "s/.*command: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$cmd_path" ]; then
            CONTAINER_CMD_PATH="$cmd_path"
            msg "$BLUE" "Using custom container command: $CONTAINER_CMD_PATH" "🔧"
        fi
    fi
    
    # Read directory settings
    if grep -q "directories:" "$yaml_file"; then
        # Input directory
        local input_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "input:" | head -n1 | sed -e "s/.*input: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$input_dir" ]; then
            if [[ "$input_dir" == /* ]]; then
                # Absolute path
                INPUT_DIR="$input_dir"
            else
                # Relative path
                INPUT_DIR="${SCRIPT_DIR}/${input_dir}"
            fi
            msg "$BLUE" "Input directory: $INPUT_DIR" "📂"
        fi
        
        # Output directory
        local output_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "output:" | head -n1 | sed -e "s/.*output: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$output_dir" ]; then
            if [[ "$output_dir" == /* ]]; then
                # Absolute path
                OUTPUT_DIR="$output_dir"
            else
                # Relative path
                OUTPUT_DIR="${SCRIPT_DIR}/${output_dir}"
            fi
            msg "$BLUE" "Output directory: $OUTPUT_DIR" "📁"
        fi
        
        # Containers directory
        local containers_dir=$(grep -A 10 "directories:" "$yaml_file" | grep "containers:" | head -n1 | sed -e "s/.*containers: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$containers_dir" ]; then
            if [[ "$containers_dir" == /* ]]; then
                # Absolute path for container
                IMAGES_DIR="$containers_dir"
            else
                # Relative path for container
                IMAGES_DIR="${SCRIPT_DIR}/${containers_dir}"
            fi
            msg "$BLUE" "Containers directory: $IMAGES_DIR" "📁"
        fi
    fi
    
    # Read build settings
    if grep -q "build:" "$yaml_file"; then
        # Definition file
        local def_file=$(grep -A 10 "build:" "$yaml_file" | grep "definition:" | head -n1 | sed -e "s/.*definition: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$def_file" ]; then
            if [[ "$def_file" == /* ]]; then
                # Absolute path
                DEF_FILE="$def_file"
            else
                # Relative path
                DEF_FILE="${SCRIPT_DIR}/${def_file}"
            fi
            msg "$BLUE" "Using definition file: $DEF_FILE" "📄"
        fi
    fi
    
    # Read validation settings
    if grep -q "validate:" "$yaml_file"; then
        # GPU setting
        local gpu_setting=$(grep -A 10 "validate:" "$yaml_file" | grep "gpu:" | head -n1 | sed -e "s/.*gpu: *//;s/['\"]//g" | tr -d ' ')
        if [ "$gpu_setting" == "false" ]; then
            USE_GPU=false
            msg "$BLUE" "GPU support: disabled" "🔄"
        else
            USE_GPU=true
            msg "$BLUE" "GPU support: enabled" "🔄"
        fi
        
        # Result file
        local result_file=$(grep -A 10 "validate:" "$yaml_file" | grep "result_file:" | head -n1 | sed -e "s/.*result_file: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$result_file" ]; then
            if [[ "$result_file" == /* ]]; then
                # Absolute path
                RESULT_FILE="$result_file"
            else
                # Relative path
                RESULT_FILE="${SCRIPT_DIR}/${result_file}"
            fi
            msg "$BLUE" "Using result file: $RESULT_FILE" "📊"
        fi
        
        # Sudo option
        local sudo_setting=$(grep -A 10 "validate:" "$yaml_file" | grep "use_sudo:" | head -n1 | sed -e "s/.*use_sudo: *//;s/['\"]//g" | tr -d ' ')
        if [ "$sudo_setting" == "false" ]; then
            USE_SUDO=false
            msg "$BLUE" "Sudo: disabled" "🔑"
        fi
    fi
    
    # Set container path if not already done
    if [ -z "${CONTAINER_PATH:-}" ]; then
        CONTAINER_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
    fi
}

# Show help
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name NAME      Container name (default: fomo25-container)"
  echo "  -d, --def FILE       Definition file (default: ./Apptainer.def)"
  echo "  -o, --output DIR     Output directory for containers (default: ./apptainer-images)"
  echo "  -c, --config FILE    Config file path (default: ./container_config.yml)"
  echo "  -i, --input DIR      Input directory for validation (default: ./test/input)"
  echo "  -r, --result DIR     Results directory for validation (default: ./test/output)"
  echo "  --result-file FILE   Output JSON file for validation results"
  echo "  --cmd PATH           Custom Apptainer/Singularity command path"
  echo "  --no-gpu             Disable GPU support for validation"
  echo "  --no-sudo            Build container without sudo (requires user namespaces)"
  echo "  --build-only         Only build the container, don't validate"
  echo "  --validate-only      Only validate the container, don't build"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  $0 -n custom-model                  # Build and validate with custom name"
  echo "  $0 -d custom.def -n my-model        # Build and validate with custom definition"
  echo "  $0 -c my-config.yml                 # Use custom config file"
  echo "  $0 --build-only                     # Only build the container"
  echo "  $0 --validate-only -n my-container  # Only validate existing container"
  echo "  $0 --no-sudo                        # Build without sudo (user namespace mode)"
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
        # Update container path only if not explicitly set
        if [ -z "${CONTAINER_PATH:-}" ]; then
            CONTAINER_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
        fi
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
      -i|--input)
        INPUT_DIR="$2"
        shift 2
        ;;
      -r|--result)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      -c|--config)
        # Already handled above
        shift 2
        ;;
      --result-file)
        RESULT_FILE="$2"
        shift 2
        ;;
      --cmd)
        CONTAINER_CMD_PATH="$2"
        shift 2
        ;;
      --no-gpu)
        USE_GPU=false
        shift
        ;;
      --no-sudo)
        USE_SUDO=false
        shift
        ;;
      --build-only)
        ONLY_BUILD=true
        shift
        ;;
      --validate-only)
        ONLY_VALIDATE=true
        shift
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
  
  # Validate incompatible options
  if $ONLY_BUILD && $ONLY_VALIDATE; then
    msg "$RED" "Cannot use both --build-only and --validate-only at the same time" "⚠️"
    exit 1
  fi
  
  # Set container path if not already done
  if [ -z "${CONTAINER_PATH:-}" ]; then
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
      handle_error 1 "Custom container command not found: $CONTAINER_CMD_PATH"
    fi
  elif command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
    msg "$GREEN" "Using Apptainer" "✅"
  elif command -v singularity &>/dev/null; then
    CONTAINER_CMD="singularity"
    msg "$GREEN" "Using Singularity" "✅"
  else
    msg "$RED" "Neither Apptainer nor Singularity found. Please install one of them." "❌"
    handle_error 1 "Neither Apptainer nor Singularity found"
  fi
  
  # Check definition file exists if building
  if ! $ONLY_VALIDATE; then
    if [ ! -f "$DEF_FILE" ]; then
      msg "$RED" "Definition file not found: $DEF_FILE" "❌"
      handle_error 1 "Definition file not found: $DEF_FILE"
    fi
  fi
  
  # Check container exists if validating
  if $ONLY_VALIDATE || ! $ONLY_BUILD; then
    if [ ! -f "$CONTAINER_PATH" ] && $ONLY_VALIDATE; then
      msg "$RED" "Container not found: $CONTAINER_PATH" "❌"
      handle_error 1 "Container not found: $CONTAINER_PATH"
    fi
  fi
  
  # Create directories
  mkdir -p "$IMAGES_DIR"
  if ! $ONLY_BUILD; then
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
  fi
}

# Check for GPU capabilities
check_gpu() {
  if $USE_GPU; then
    msg "$BLUE" "Checking for GPU support..." "🔍"
    
    # Multiple checks for better detection
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
      msg "$GREEN" "NVIDIA GPU detected with nvidia-smi" "✅"
      return 0
    elif [ -d /dev/dri ] && [ -n "$(ls -A /dev/dri 2>/dev/null)" ]; then
      msg "$GREEN" "GPU devices detected in /dev/dri" "✅"
      return 0
    elif [ -c /dev/nvidia0 ]; then
      msg "$GREEN" "NVIDIA GPU device found at /dev/nvidia0" "✅"
      return 0
    else
      msg "$YELLOW" "No GPU detected, will use CPU mode" "⚠️"
      USE_GPU=false
      return 1
    fi
  else
    msg "$BLUE" "GPU support disabled, using CPU mode" "🔄"
    return 1
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
  
  # Create directory if it doesn't exist
  mkdir -p "$(dirname "$CONTAINER_PATH")"
  
  # Build command
  BUILD_CMD=""
  if $USE_SUDO; then
    BUILD_CMD="sudo $CONTAINER_CMD build"
  else
    BUILD_CMD="$CONTAINER_CMD build --fakeroot"
  fi
  
  # Build with proper error handling
  msg "$BLUE" "Running: $BUILD_CMD $CONTAINER_PATH $DEF_FILE" "📋"
  if $BUILD_CMD "$CONTAINER_PATH" "$DEF_FILE"; then
    if [ -f "$CONTAINER_PATH" ]; then
      msg "$GREEN" "Container built successfully" "✅"
      msg "$BLUE" "Container location: $CONTAINER_PATH" "📦"
      return 0
    else
      msg "$RED" "Build failed: Container file not created" "❌"
      handle_error 1 "Container file not created after build"
    fi
  else
    msg "$RED" "Build failed" "❌"
    handle_error 1 "Container build command failed"
  fi
}

# Test container structure
test_container_structure() {
  msg "$BLUE" "Testing container structure: $CONTAINER_PATH" "🧪"
  
  # Basic structure test
  if ! "$CONTAINER_CMD" test "$CONTAINER_PATH"; then
    msg "$RED" "Container self-test failed" "❌"
    handle_error 1 "Container self-test failed"
  fi
  
  # Check for basic functionality
  msg "$BLUE" "Testing if container runs basic commands..." "🔍"
  if "$CONTAINER_CMD" exec "$CONTAINER_PATH" echo "Container is working" >/dev/null 2>&1; then
    msg "$GREEN" "Container can run basic commands" "✅"
  else
    msg "$RED" "Container failed to run basic command" "❌"
    handle_error 1 "Container failed to run basic command"
  fi
  
  # Check container structure
  msg "$BLUE" "Checking container file structure..." "🔍"
  if "$CONTAINER_CMD" exec "$CONTAINER_PATH" test -f /app/predict.py; then
    msg "$GREEN" "Found prediction script: /app/predict.py" "✅"
  else
    msg "$RED" "Required file /app/predict.py not found in container" "❌"
    handle_error 1 "Required file /app/predict.py not found in container"
  fi
  
  # Check predict.py is executable
  if "$CONTAINER_CMD" exec "$CONTAINER_PATH" test -x /app/predict.py; then
    msg "$GREEN" "predict.py is executable" "✅"
  else
    msg "$YELLOW" "Warning: predict.py is not executable" "⚠️"
    
    # Try to fix it
    msg "$BLUE" "Attempting to make predict.py executable..." "🔧"
    if "$CONTAINER_CMD" exec "$CONTAINER_PATH" chmod +x /app/predict.py 2>/dev/null; then
      msg "$GREEN" "Fixed: predict.py is now executable" "✅"
    else
      msg "$YELLOW" "Could not make predict.py executable, validation may fail" "⚠️"
    fi
  fi
}

# Check GPU support in container
test_container_gpu() {
  if $USE_GPU; then
    msg "$BLUE" "Checking container GPU support..." "🔍"
    
    # Use GPU flag for container execution
    GPU_FLAG="--nv"
    
    # Try to check GPU visibility
    if "$CONTAINER_CMD" exec $GPU_FLAG "$CONTAINER_PATH" nvidia-smi &>/dev/null; then
      msg "$GREEN" "GPU support verified with nvidia-smi" "✅"
      return 0
    else
      # Try alternative method
      local gpu_env=$("$CONTAINER_CMD" exec $GPU_FLAG "$CONTAINER_PATH" bash -c '
        python3 -c "import os; print(\"GPU_AVAILABLE=\" + str(\"NVIDIA_VISIBLE_DEVICES\" in os.environ))"
      ' 2>/dev/null)
      
      if [[ "$gpu_env" == *"GPU_AVAILABLE=True"* ]]; then
        msg "$GREEN" "GPU support verified through environment variables" "✅"
        return 0
      else
        # Try to check for CUDA
        local cuda_check=$("$CONTAINER_CMD" exec $GPU_FLAG "$CONTAINER_PATH" bash -c '
          python3 -c "
import sys
try:
    import torch
    print(\"CUDA_AVAILABLE=\" + str(torch.cuda.is_available()))
except (ImportError, ModuleNotFoundError):
    try:
        import tensorflow as tf
        print(\"GPU_AVAILABLE=\" + str(len(tf.config.list_physical_devices(\"GPU\")) > 0))
    except (ImportError, ModuleNotFoundError):
        print(\"NO_FRAMEWORK=True\")
"
        ' 2>/dev/null)
        
        if [[ "$cuda_check" == *"CUDA_AVAILABLE=True"* || "$cuda_check" == *"GPU_AVAILABLE=True"* ]]; then
          msg "$GREEN" "GPU support verified through ML framework" "✅"
          return 0
        else
          msg "$YELLOW" "GPU detected on host but not visible in container" "⚠️"
          
          # Recommendations
          if [[ "$cuda_check" == *"NO_FRAMEWORK=True"* ]]; then
            msg "$YELLOW" "Neither PyTorch nor TensorFlow found in container" "⚠️"
            msg "$YELLOW" "Ensure your container has GPU-compatible ML frameworks installed" "💡"
          else
            msg "$YELLOW" "Container may lack GPU drivers or has compatibility issues" "⚠️"
            msg "$YELLOW" "Ensure container has compatible CUDA/driver versions" "💡"
          fi
          
          msg "$YELLOW" "Will try CPU mode for validation" "⚠️"
          USE_GPU=false
          return 1
        fi
      fi
    fi
  else
    msg "$BLUE" "GPU support disabled, using CPU mode" "🔄"
    return 1
  fi
}

# Generate test data if needed
generate_test_data() {
  if [ ! "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
    msg "$YELLOW" "Input directory is empty, generating synthetic test data" "🧪"
    if [ -f "${SCRIPT_DIR}/validation/test_data_generator.py" ]; then
      python3 "${SCRIPT_DIR}/validation/test_data_generator.py" "$INPUT_DIR" || {
        msg "$RED" "Failed to generate test data" "❌"
        handle_error 1 "Failed to generate test data"
      }
      msg "$GREEN" "Test data generated successfully" "✅"
    else
      msg "$RED" "Test data generator script not found" "❌"
      handle_error 1 "Test data generator script not found: ${SCRIPT_DIR}/validation/test_data_generator.py"
    fi
  else
    msg "$GREEN" "Using existing test data in $INPUT_DIR" "✅"
  fi
}

# Compute metrics (added from first script)
compute_metrics() {
  msg "$BLUE" "Computing metrics" "📊"
  if [ -f "${SCRIPT_DIR}/validation/compute_metrics.py" ]; then
    python3 "${SCRIPT_DIR}/validation/compute_metrics.py" "$OUTPUT_DIR" "$INPUT_DIR" || {
      msg "$RED" "Failed to compute metrics" "❌"
      errors+=("Failed to compute metrics")
      return 1
    }
    msg "$GREEN" "Metrics computed successfully" "✅"
    
    # Check if metrics were generated
    if [ -f "${OUTPUT_DIR}/results/metrics_results.csv" ]; then
      msg "$GREEN" "Metrics saved to ${OUTPUT_DIR}/results/metrics_results.csv" "📈"
    else
      msg "$YELLOW" "Metrics computation completed but CSV not found" "⚠️"
    fi
    
    return 0
  else
    msg "$RED" "Metrics script not found: ${SCRIPT_DIR}/validation/compute_metrics.py" "❌"
    errors+=("Metrics script not found")
    return 1
  fi
}

# Run inference
run_inference() {
  # Generate test data if needed
  generate_test_data
  
  # Check if input directory has data
  local input_files=$(find "$INPUT_DIR" -type f -name "*.nii*" | wc -l)
  if [ "$input_files" -eq 0 ]; then
    msg "$RED" "No NIfTI files found in input directory: $INPUT_DIR" "❌"
    handle_error 1 "No NIfTI files found in input directory"
  fi
  
  # Clean output directory
  rm -rf "${OUTPUT_DIR:?}"/* 2>/dev/null || true
  mkdir -p "$OUTPUT_DIR/results"
  chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
  
  # GPU flag
  GPU_FLAG=""
  if $USE_GPU; then
    GPU_FLAG="--nv"
  fi
  
  # Create results directory in output
  mkdir -p "$OUTPUT_DIR/results"
  
  # Create instance name from container name
  local INSTANCE_NAME="${CONTAINER_NAME}_instance"
  
  # Check if instance already exists and stop it if needed
  if "$CONTAINER_CMD" instance list | grep -q "$INSTANCE_NAME"; then
    msg "$YELLOW" "Instance $INSTANCE_NAME already exists, stopping it first" "⚠️"
    "$CONTAINER_CMD" instance stop "$INSTANCE_NAME" || true
  fi
  
  # Run container in detached mode
  msg "$BLUE" "Starting container instance: $INSTANCE_NAME" "🚀"
  "$CONTAINER_CMD" instance start $GPU_FLAG \
    --bind "$INPUT_DIR:/input:ro" \
    --bind "$OUTPUT_DIR:/output" \
    "$CONTAINER_PATH" "$INSTANCE_NAME" || {
    msg "$RED" "Failed to start container instance" "❌"
    handle_error 1 "Failed to start container instance"
  }
  msg "$GREEN" "Container instance started successfully" "✅"
  
  # Run inference on each file
  msg "$BLUE" "Processing $input_files NIfTI files from $INPUT_DIR" "🧠"
  local count=0
  local success=0
  local failed_files=()
  
  for input_file in "$INPUT_DIR"/*.nii*; do
    if [ -f "$input_file" ]; then
      count=$((count+1))
      filename=$(basename "$input_file")
      output_filename="${filename%.nii*}_pred.nii.gz"
      
      msg "$BLUE" "[$count/$input_files] Processing $filename..." "⏳"
      
      # Run inference with performance monitoring
      start_time=$(date +%s.%N)
      
      "$CONTAINER_CMD" exec instance://"$INSTANCE_NAME" \
        python /app/predict.py --input "/input/$filename" --output "/output/$output_filename"
      
      exit_code=$?
      end_time=$(date +%s.%N)
      duration=$(echo "$end_time - $start_time" | bc)
      
      if [ $exit_code -eq 0 ] && [ -f "$OUTPUT_DIR/$output_filename" ]; then
        success=$((success+1))
        msg "$GREEN" "Successfully processed $filename in ${duration}s" "✅"
      else
        msg "$RED" "Failed to process $filename (exit code: $exit_code)" "❌"
        failed_files+=("$filename")
      fi
    fi
  done
  
  # Stop the instance
  msg "$BLUE" "Stopping container instance" "🛑"
  "$CONTAINER_CMD" instance stop "$INSTANCE_NAME" || {
    msg "$YELLOW" "Warning: Failed to stop container instance" "⚠️"
  }
  
  # Check if any output was generated
  if [ $success -eq 0 ]; then
    msg "$RED" "No files were processed successfully" "❌"
    handle_error 1 "No files were processed successfully"
  fi
  
  msg "$BLUE" "Inference complete: $success/$count files processed successfully" "🏁"
  
  # Save performance data
  echo "{\"files_processed\": $count, \"files_succeeded\": $success, \"processing_time\": $duration}" \
    > "$OUTPUT_DIR/results/performance.json"
  
  # Always compute metrics (using the function from first script)
  compute_metrics
  
  # Return status based on success
  if [ $success -gt 0 ]; then
    return 0
  else
    return 1
  fi
}

# Save validation results
save_validation_results() {
  local status=$1
  local container_exists=$2
  local can_run_basic_commands=$3
  local required_files_present=$4
  local gpu_supported=$5
  local inference_succeeded=$6
  
  msg "$BLUE" "Saving validation results to $RESULT_FILE" "💾"
  
  # Create errors and warnings arrays
  errors=()
  warnings=()
  
  # Add errors based on test results
  if [ "$container_exists" == "false" ]; then
    errors+=("Container not found at $CONTAINER_PATH")
  fi
  
  if [ "$can_run_basic_commands" == "false" ]; then
    errors+=("Container cannot run basic commands")
  fi
  
  if [ "$required_files_present" == "false" ]; then
    errors+=("Required file /app/predict.py not found in container")
  fi
  
  if [ "$inference_succeeded" == "false" ]; then
    errors+=("Inference failed to produce valid outputs")
  fi
  
  # Add warnings for GPU if relevant
  if $USE_GPU && [ "$gpu_supported" == "false" ]; then
    warnings+=("GPU requested but not supported by container, used CPU instead")
  fi
  
  # Create JSON structure
  cat > "$RESULT_FILE" << EOF
{
  "status": "$status",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "container": {
    "name": "$CONTAINER_NAME",
    "path": "$CONTAINER_PATH",
    "build_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "size_bytes": $(du -b "$CONTAINER_PATH" 2>/dev/null | cut -f1 || echo "null")
  },
  "checks": {
    "container_exists": $container_exists,
    "can_run_basic_commands": $can_run_basic_commands,
    "required_files_present": $required_files_present,
    "gpu_supported": $gpu_supported,
    "inference_succeeded": $inference_succeeded
  },
  "errors": [
EOF

  # Add errors
  for ((i=0; i<${#errors[@]}; i++)); do
    cat >> "$RESULT_FILE" << EOF
    "${errors[$i]}"$(if [[ $i -lt $((${#errors[@]}-1)) ]]; then echo ","; fi)
EOF
  done

  # Add warnings section
  cat >> "$RESULT_FILE" << EOF
  ],
  "warnings": [
EOF

  # Add warnings
  for ((i=0; i<${#warnings[@]}; i++)); do
    cat >> "$RESULT_FILE" << EOF
    "${warnings[$i]}"$(if [[ $i -lt $((${#warnings[@]}-1)) ]]; then echo ","; fi)
EOF
  done

  # Add performance data if available
  if [ -f "$OUTPUT_DIR/results/performance.json" ]; then
    cat >> "$RESULT_FILE" << EOF
  ],
  "performance": $(cat "$OUTPUT_DIR/results/performance.json")
}
EOF
  else
    # Close JSON if no performance data
    cat >> "$RESULT_FILE" << EOF
  ]
}
EOF
  fi
}

# Run complete validation
validate_container() {
  msg "$BLUE" "Starting container validation" "🚀"
  
  # Initialize status variables
  local container_exists=false
  local can_run_basic_commands=false
  local required_files_present=false
  local gpu_supported=false
  local inference_succeeded=false
  
  # Check if container exists
  if [ -f "$CONTAINER_PATH" ]; then
    container_exists=true
    msg "$GREEN" "Container exists at $CONTAINER_PATH" "✅"
    
    # Test container structure
    test_container_structure
    can_run_basic_commands=true
    required_files_present=true
    
    # Test GPU support
    if $USE_GPU; then
      test_container_gpu && gpu_supported=true
    fi
    
    # Run inference
    if run_inference; then
      inference_succeeded=true
      msg "$GREEN" "Inference succeeded" "✅"
    else
      msg "$RED" "Inference failed" "❌"
    fi
    
    # Save validation results
    local status="PASSED"
    if [ "$inference_succeeded" == "false" ]; then
      status="FAILED"
    fi
    
    save_validation_results "$status" "$container_exists" "$can_run_basic_commands" \
      "$required_files_present" "$gpu_supported" "$inference_succeeded"
    
    if [ "$status" == "PASSED" ]; then
      msg "$GREEN" "Validation completed successfully" "🎉"
      return 0
    else
      msg "$RED" "Validation failed" "❌"
      return 1
    fi
  else
    msg "$RED" "Container not found at $CONTAINER_PATH" "❌"
    save_validation_results "FAILED" "false" "false" "false" "false" "false"
    return 1
  fi
}

# Main function
main() {
  msg "$BLUE" "FOMO25 Challenge Container Tool" "🚀"
  
  # Parse arguments
  parse_args "$@"
  
  # Check environment
  check_env
  
  # Check for GPU support
  if ! $ONLY_BUILD; then
    check_gpu
  fi
  
  # Perform requested operations
  if $ONLY_BUILD; then
    build_container
    return $?
  elif $ONLY_VALIDATE; then
    validate_container
    return $?
  else
    # Both build and validate
    build_container && validate_container
    return $?
  fi
}

# Trap for cleanup
cleanup() {
  # Stop any running instances
  if [ ! -z "$CONTAINER_CMD" ]; then
    "$CONTAINER_CMD" instance list 2>/dev/null | grep -q "${CONTAINER_NAME}_instance" && \
      "$CONTAINER_CMD" instance stop "${CONTAINER_NAME}_instance" 2>/dev/null || true
  fi
  
  msg "$BLUE" "Cleanup complete" "🧹"
}

# Set trap
trap cleanup EXIT INT TERM

# Run the main function
main "$@"