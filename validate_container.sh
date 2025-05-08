#!/bin/bash
# FOMO25 Container Validation Tool

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
CONTAINER_PATH="${SCRIPT_DIR}/apptainer-images/${CONTAINER_NAME}.sif"
# Removed OPERATION variable
USE_GPU=true
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
CONFIG_FILE="${SCRIPT_DIR}/container_config.yml"
RESULT_FILE="${SCRIPT_DIR}/validation_result.json"
CONTAINER_CMD="apptainer" # Default command
CONTAINER_CMD_PATH="" # Custom command path

# Validation status variables
container_exists=false
container_runtime_available=false
can_run_basic_commands=false
required_files_present=false
gpu_supported=false
inference_ran=false
outputs_generated=false
errors=()
warnings=()

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
        # Container name
        local container_name=$(grep -A 5 "container:" "$yaml_file" | grep "name:" | head -n1 | sed -e "s/.*name: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$container_name" ]; then
            CONTAINER_NAME="$container_name"
            CONTAINER_PATH="${SCRIPT_DIR}/apptainer-images/${CONTAINER_NAME}.sif"
            msg "$BLUE" "Using container name: $CONTAINER_NAME" "📦"
        fi
        
        # Command path
        local cmd_path=$(grep -A 5 "container:" "$yaml_file" | grep "command:" | head -n1 | sed -e "s/.*command: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$cmd_path" ]; then
            CONTAINER_CMD="$cmd_path"
            msg "$BLUE" "Using container command: $CONTAINER_CMD" "🔧"
        fi
    fi
    
    # GPU setting from validate section
    if grep -q "validate:" "$yaml_file"; then
        local gpu_setting=$(grep -A 10 "validate:" "$yaml_file" | grep "gpu:" | head -n1 | sed -e "s/.*gpu: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$gpu_setting" ]; then
            if [ "$gpu_setting" == "false" ]; then
                USE_GPU=false
                msg "$BLUE" "GPU support: disabled" "🔄"
            else
                USE_GPU=true
                msg "$BLUE" "GPU support: enabled" "🔄"
            fi
        fi
        
        # Result file
        local result_file=$(grep -A 10 "validate:" "$yaml_file" | grep "result_file:" | head -n1 | sed -e "s/.*result_file: *//;s/['\"]//g" | tr -d ' ')
        if [ ! -z "$result_file" ]; then
            RESULT_FILE="${SCRIPT_DIR}/${result_file}"
            msg "$BLUE" "Using result file: $RESULT_FILE" "📊"
        fi
    fi
    
    # Directories
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
                CONTAINER_PATH="${containers_dir}/${CONTAINER_NAME}.sif"
            else
                # Relative path for container
                CONTAINER_PATH="${SCRIPT_DIR}/${containers_dir}/${CONTAINER_NAME}.sif"
            fi
            msg "$BLUE" "Container path: $CONTAINER_PATH" "📁"
        fi
    fi
}

# Show help
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name NAME      Container name (default: fomo25-container)"
  echo "  -p, --path PATH      Container path (overrides name)"
  echo "  -i, --input DIR      Input directory (default: ./test/input)"
  echo "  -o, --output DIR     Output directory (default: ./test/output)"
  echo "  -c, --config FILE    Config file path (default: ./container_config.yml)"
  echo "  --no-gpu             Disable GPU support"
  echo "  --result FILE        Specify output JSON file for results"
  echo "  --cmd PATH           Custom Apptainer/Singularity command path"
  echo "  -h, --help           Show this help"
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
        CONTAINER_PATH="${SCRIPT_DIR}/apptainer-images/${CONTAINER_NAME}.sif"
        shift 2
        ;;
      -p|--path)
        CONTAINER_PATH="$2"
        shift 2
        ;;
      -i|--input)
        INPUT_DIR="$2"
        shift 2
        ;;
      -o|--output)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      -c|--config)
        # Already handled above
        shift 2
        ;;
      --no-gpu)
        USE_GPU=false
        shift
        ;;
      --result)
        RESULT_FILE="$2"
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
}

# Check environment
check_env() {
  # Check for Apptainer/Singularity
  if [ ! -z "$CONTAINER_CMD_PATH" ]; then
    # Custom command path specified
    if [ -x "$CONTAINER_CMD_PATH" ]; then
      CONTAINER_CMD="$CONTAINER_CMD_PATH"
      msg "$GREEN" "Using custom container command: $CONTAINER_CMD" "✅"
      container_runtime_available=true
    elif command -v "$CONTAINER_CMD_PATH" &>/dev/null; then
      CONTAINER_CMD="$CONTAINER_CMD_PATH"
      msg "$GREEN" "Found custom container command: $CONTAINER_CMD" "✅"
      container_runtime_available=true
    else
      msg "$RED" "Custom container command not found: $CONTAINER_CMD_PATH" "❌"
      errors+=("Custom container command not found: $CONTAINER_CMD_PATH")
      container_runtime_available=false
      exit 1
    fi
  elif command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
    msg "$GREEN" "Using Apptainer" "✅"
    container_runtime_available=true
  else
    msg "$RED" "Apptainer was not found. Please install one of them." "❌"
    errors+=("Neither apptainer nor singularity found")
    container_runtime_available=false
    exit 1
  fi
  
  # Check container exists
  if [ ! -f "$CONTAINER_PATH" ]; then
    msg "$RED" "Container not found: $CONTAINER_PATH" "❌"
    msg "$BLUE" "Run $CONTAINER_CMD build first to create the container" "💡"
    errors+=("Container not found: $CONTAINER_PATH")
    exit 1
  else
    msg "$GREEN" "Container found: $CONTAINER_PATH" "✅"
    container_exists=true
  fi
  
  # Create directories
  mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
  msg "$BLUE" "Using input directory: $INPUT_DIR" "📂"
  msg "$BLUE" "Using output directory: $OUTPUT_DIR" "📁"
}

# Test container
test_container() {
  msg "$BLUE" "Testing container: $CONTAINER_PATH" "🧪"
  
  # Basic structure test
  if ! "$CONTAINER_CMD" test "$CONTAINER_PATH"; then
    msg "$RED" "Container self-test failed" "❌"
    errors+=("Container self-test failed")
    return 1
  fi
  
  # Check for basic functionality
  msg "$BLUE" "Testing if container runs basic commands..." "🔍"
  if "$CONTAINER_CMD" exec "$CONTAINER_PATH" echo "Container is working" >/dev/null 2>&1; then
    msg "$GREEN" "Container can run basic commands" "✅"
    can_run_basic_commands=true
  else
    msg "$RED" "Container failed to run basic command" "❌"
    errors+=("Container failed to run basic command")
    return 1
  fi
  
  # Check container structure
  msg "$BLUE" "Checking container file structure..." "🔍"
  if "$CONTAINER_CMD" exec "$CONTAINER_PATH" test -f /app/predict.py; then
    msg "$GREEN" "Found prediction script: /app/predict.py" "✅"
    required_files_present=true
  else
    msg "$RED" "Required file /app/predict.py not found in container" "❌"
    errors+=("Required file /app/predict.py not found in container")
    return 1
  fi
  
  # Check GPU support if enabled
  if $USE_GPU; then
    msg "$BLUE" "Checking GPU support..." "🔍"
    if "$CONTAINER_CMD" exec --nv "$CONTAINER_PATH" nvidia-smi &>/dev/null; then
      msg "$GREEN" "GPU support verified with nvidia-smi" "✅"
      gpu_supported=true
    else
      # Try alternative method
      local gpu_env=$("$CONTAINER_CMD" exec --nv "$CONTAINER_PATH" bash -c 'python -c "import os; print(\"GPU_AVAILABLE=\" + str(\"NVIDIA_VISIBLE_DEVICES\" in os.environ))"' 2>/dev/null)
      
      if [[ "$gpu_env" == *"GPU_AVAILABLE=True"* ]]; then
        msg "$GREEN" "GPU support verified through environment variables" "✅"
        gpu_supported=true
      else
        msg "$YELLOW" "GPU not available, continuing with CPU" "⚠️"
        warnings+=("Container cannot access GPU, continuing with CPU")
        gpu_supported=false
      fi
    fi
  fi
  
  msg "$GREEN" "Container tests passed" "✅"
  return 0
}

# Generate test data if needed
generate_test_data() {
  if [ ! "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
    msg "$YELLOW" "Input directory is empty, generating synthetic test data" "🧪"
    if [ -f "${SCRIPT_DIR}/validation/test_data_generator.py" ]; then
      python3 "${SCRIPT_DIR}/validation/test_data_generator.py" "$INPUT_DIR" || {
        msg "$RED" "Failed to generate test data" "❌"
        errors+=("Failed to generate test data")
        return 1
      }
      msg "$GREEN" "Test data generated successfully" "✅"
    else
      msg "$RED" "Test data generator script not found" "❌"
      errors+=("Test data generator script not found")
      return 1
    fi
  fi
  return 0
}

# Check if outputs were generated
check_outputs() {
  msg "$BLUE" "Checking for generated outputs..." "🔍"
  
  local output_files=$(find "$OUTPUT_DIR" -type f | wc -l)
  
  if [[ $output_files -gt 0 ]]; then
    msg "$GREEN" "Found $output_files output files" "✅"
    outputs_generated=true
    return 0
  else
    msg "$RED" "No output files generated" "❌"
    errors+=("No output files generated")
    outputs_generated=false
    return 1
  fi
}

# Compute metrics
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

## Run inference file by file
run_inference() {
  # Generate test data if needed
  generate_test_data || return 1
  
  # Check if input directory has data
  local input_files=$(find "$INPUT_DIR" -type f -name "*.nii*" | wc -l)
  if [ "$input_files" -eq 0 ]; then
    msg "$RED" "No NIfTI files found in input directory: $INPUT_DIR" "❌"
    errors+=("No NIfTI files found in input directory")
    return 1
  fi
  
  # Clean output directory
  rm -rf "${OUTPUT_DIR:?}"/* 2>/dev/null || true
  chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
  
  # GPU flag
  GPU_FLAG=""
  if $USE_GPU; then
    GPU_FLAG="--nv"
  fi
  
  # Create instance name from container name
  local INSTANCE_NAME="${CONTAINER_NAME}_instance"
  
  # Run container in detached mode
  msg "$BLUE" "Starting container instance: $INSTANCE_NAME" "🚀"
  "$CONTAINER_CMD" instance start $GPU_FLAG \
    --bind "$INPUT_DIR:/input:ro" \
    --bind "$OUTPUT_DIR:/output" \
    "$CONTAINER_PATH" "$INSTANCE_NAME" || {
    msg "$RED" "Failed to start container instance" "❌"
    errors+=("Failed to start container instance")
    return 1
  }
  msg "$GREEN" "Container instance started successfully" "✅"
  
  # Run inference on each file
  msg "$BLUE" "Processing $input_files NIfTI files from $INPUT_DIR" "🧠"
  local count=0
  local success=0
  
  for input_file in "$INPUT_DIR"/*.nii*; do
    if [ -f "$input_file" ]; then
      count=$((count+1))
      filename=$(basename "$input_file")
      
      msg "$BLUE" "[$count/$input_files] Processing $filename..." "⏳"
      
      # Create output directory if needed
      mkdir -p "$OUTPUT_DIR"
      
      # Run inference on this file using the instance
      "$CONTAINER_CMD" exec instance://"$INSTANCE_NAME" \
        python /app/predict.py --input "/input/$filename" --output "/output/$filename"
      
      if [ $? -eq 0 ]; then
        success=$((success+1))
        msg "$GREEN" "Successfully processed $filename" "✅"
      else
        msg "$RED" "Failed to process $filename" "❌"
        errors+=("Failed to process $filename")
      fi
    fi
  done
  
  # Stop the instance
  msg "$BLUE" "Stopping container instance" "🛑"
  "$CONTAINER_CMD" instance stop "$INSTANCE_NAME" || {
    msg "$YELLOW" "Warning: Failed to stop container instance" "⚠️"
    warnings+=("Failed to stop container instance")
  }
  
  msg "$BLUE" "Inference complete: $success/$count files processed successfully" "🏁"
  
  if [ $success -gt 0 ]; then
    inference_ran=true
    check_outputs
    # Always compute metrics
    compute_metrics
    return 0
  else
    msg "$RED" "No files were processed successfully" "❌"
    errors+=("No files were processed successfully")
    inference_ran=false
    return 1
  fi
}

# Save validation results to JSON
save_results() {
  local status="PASSED"
  
  if [[ ${#errors[@]} -gt 0 ]]; then
    status="FAILED"
  fi
  
  # Create JSON structure
  cat > "$RESULT_FILE" << EOF
{
  "status": "$status",
  "checks": {
    "container_exists": $container_exists,
    "container_runtime_available": $container_runtime_available,
    "can_run_basic_commands": $can_run_basic_commands,
    "required_files_present": $required_files_present,
    "gpu_supported": $gpu_supported,
    "inference_ran": $inference_ran,
    "outputs_generated": $outputs_generated
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

  # Close JSON
  cat >> "$RESULT_FILE" << EOF
  ]
}
EOF

  msg "$BLUE" "Results saved to $RESULT_FILE" "💾"
}

# Run full validation
run_validation() {
  msg "$BLUE" "Starting complete validation process" "🚀"
  
  # Test the container
  test_container || {
    save_results
    return 1
  }
  
  # Run inference
  run_inference || {
    save_results
    return 1
  }
  
  # Save validation results
  save_results
  
  # Final status
  if [[ ${#errors[@]} -eq 0 ]]; then
    msg "$GREEN" "Validation complete - ALL CHECKS PASSED" "🎉"
    return 0
  else
    msg "$YELLOW" "Validation complete - ${#errors[@]} CHECKS FAILED" "⚠️"
    for error in "${errors[@]}"; do
      msg "$RED" "ERROR: $error" "❌"
    done
    return 1
  fi
}

# Main function
main() {
  msg "$BLUE" "FOMO25 Container Validation Tool" "🚀"
  
  # Parse arguments and config
  parse_args "$@"
  
  # Check environment
  check_env
  
  # Always run validation (operation parameter removed)
  run_validation
  
  exit $?
}

# Trap errors
trap 'msg "$RED" "An error occurred. Exiting..." "❌"; save_results; exit 1' ERR

# Execute main function
main "$@"