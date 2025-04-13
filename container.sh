#!/bin/bash
# All-in-one container management script for neuroimaging containers
# Handles building, testing, and running containers with minimal configuration

# Color codes for messages
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Variables with defaults
CONTAINER_NAME="fomo25-container"
OPERATION="build"
USE_GPU=true
IMAGES_DIR="${SCRIPT_DIR}/apptainer-images"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
DEF_FILE="${SCRIPT_DIR}/Apptainer.def"
GENERATE_DATA=true
COMPUTE_METRICS=true
RESULT_FILE="${SCRIPT_DIR}/validation_result.json"

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

# Show help
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name NAME      Container name (default: neuro-container)"
  echo "  -b, --build          Build the container"
  echo "  -t, --test           Test an existing container"
  echo "  -r, --run            Run inference on a container"
  echo "  -v, --validate       Run full validation (test+run+metrics)"
  echo "  -i, --input DIR      Input directory (default: ./test/input)"
  echo "  -o, --output DIR     Output directory (default: ./test/output)"
  echo "  -d, --def FILE       Definition file (default: ./Apptainer.def)"
  echo "  --no-gpu             Disable GPU support"
  echo "  --no-generate        Skip test data generation"
  echo "  --no-metrics         Skip metrics computation"
  echo "  --result FILE        Specify output JSON file for results"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  $0 -b -n mycontainer          # Build a container"
  echo "  $0 -t -n mycontainer          # Test a container"
  echo "  $0 -r -n mycontainer -i data  # Run inference with custom input directory"
  echo "  $0 -v -n mycontainer          # Run full validation suite"
  # All the process (Build, Test, Run, Metrics)
  echo "  $0 -v -n mycontainer -i data  # Run full validation with custom input directory"
}

# Parse arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -n|--name)
        CONTAINER_NAME="$2"
        shift 2
        ;;
      -b|--build)
        OPERATION="build"
        shift
        ;;
      -t|--test)
        OPERATION="test"
        shift
        ;;
      -r|--run)
        OPERATION="run"
        shift
        ;;
      -v|--validate)
        OPERATION="validate"
        shift
        ;;
      -i|--input)
        INPUT_DIR="$2"
        shift 2
        ;;
      -o|--output)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      -d|--def)
        DEF_FILE="$2"
        shift 2
        ;;
      --no-gpu)
        USE_GPU=false
        shift
        ;;
      --no-generate)
        GENERATE_DATA=false
        shift
        ;;
      --no-metrics)
        COMPUTE_METRICS=false
        shift
        ;;
      --result)
        RESULT_FILE="$2"
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

  # Create output directory if it doesn't exist
  mkdir -p "$IMAGES_DIR"
  
  # Set image path
  IMAGE_PATH="${IMAGES_DIR}/${CONTAINER_NAME}.sif"
}

# Check environment
check_env() {
  # Check for Apptainer/Singularity
  CONTAINER_CMD=""
  if command -v apptainer &>/dev/null; then
    CONTAINER_CMD="apptainer"
    msg "$BLUE" "Using Apptainer" "🐋"
    container_runtime_available=true
  elif command -v singularity &>/dev/null; then
    CONTAINER_CMD="singularity"
    msg "$BLUE" "Using Singularity" "🐋"
    container_runtime_available=true
    warnings+=("Using singularity instead of apptainer")
  else
    msg "$RED" "Neither Apptainer nor Singularity found. Please install one of them." "❌"
    errors+=("Neither apptainer nor singularity found")
    container_runtime_available=false
    exit 1
  fi
  
  # Check definition file exists if building
  if [ "$OPERATION" = "build" ] && [ ! -f "$DEF_FILE" ]; then
    msg "$RED" "Definition file not found: $DEF_FILE" "❌"
    errors+=("Definition file not found: $DEF_FILE")
    exit 1
  fi
  
  # Check container exists if testing/running/validating
  if [ "$OPERATION" != "build" ]; then
    if [ -f "$IMAGE_PATH" ]; then
      msg "$GREEN" "Container found: $IMAGE_PATH" "✅"
      container_exists=true
    else
      msg "$RED" "Container not found: $IMAGE_PATH" "❌"
      msg "$BLUE" "Build it first with: $0 -b -n $CONTAINER_NAME" "💡"
      errors+=("Container not found: $IMAGE_PATH")
      exit 1
    fi
  fi
  
  # Create directories
  mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
}

# Build container
build_container() {
  msg "$BLUE" "Building container: $IMAGE_PATH" "🔨"
  
  # Build with proper error handling
  if sudo "$CONTAINER_CMD" build "$IMAGE_PATH" "$DEF_FILE"; then
    if [ -f "$IMAGE_PATH" ]; then
      msg "$GREEN" "Container built successfully" "✅"
      return 0
    else
      msg "$RED" "Build failed: Container file not created" "❌"
      errors+=("Build failed: Container file not created")
      return 1
    fi
  else
    msg "$RED" "Build failed" "❌"
    errors+=("Container build failed")
    return 1
  fi
}

# Test container
test_container() {
  msg "$BLUE" "Testing container: $IMAGE_PATH" "🧪"
  
  # Basic structure test
  if ! "$CONTAINER_CMD" test "$IMAGE_PATH"; then
    msg "$RED" "Container self-test failed" "❌"
    errors+=("Container self-test failed")
    return 1
  fi
  
  # Check for basic functionality
  msg "$BLUE" "Testing if container runs basic commands..." "🔍"
  if "$CONTAINER_CMD" exec "$IMAGE_PATH" echo "Container is working" >/dev/null 2>&1; then
    msg "$GREEN" "Container can run basic commands" "✅"
    can_run_basic_commands=true
  else
    msg "$RED" "Container failed to run basic command" "❌"
    errors+=("Container failed to run basic command")
    return 1
  fi
  
  # Check container structure
  msg "$BLUE" "Checking container file structure..." "🔍"
  if "$CONTAINER_CMD" exec "$IMAGE_PATH" test -f /app/predict.py; then
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
    if "$CONTAINER_CMD" exec --nv "$IMAGE_PATH" nvidia-smi &>/dev/null; then
      msg "$GREEN" "GPU support verified with nvidia-smi" "✅"
      gpu_supported=true
    else
      # Try alternative method
      local gpu_env=$("$CONTAINER_CMD" exec --nv "$IMAGE_PATH" bash -c 'python -c "import os; print(\"GPU_AVAILABLE=\" + str(\"NVIDIA_VISIBLE_DEVICES\" in os.environ))"' 2>/dev/null)
      
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

# Monitor memory and time
monitor_performance() {
  local start_time=$SECONDS
  local max_memory=0
  local pid=$1
  
  # Monitor memory usage
  while kill -0 $pid 2>/dev/null; do
    local current_memory=$(ps -o rss= -p $pid 2>/dev/null || echo "0")
    if [[ $current_memory -gt $max_memory ]]; then
      max_memory=$current_memory
    fi
    sleep 0.1
  done
  
  local elapsed_time=$((SECONDS - start_time))
  
  msg "$BLUE" "Performance metrics:" "📊"
  msg "$BLUE" "  Memory usage: $((max_memory / 1024)) MB" "💾"
  msg "$BLUE" "  Execution time: ${elapsed_time} seconds" "⏱️"
  
  return 0
}

# Generate test data if needed
generate_test_data() {
  if $GENERATE_DATA && [ ! "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
    msg "$YELLOW" "Generating synthetic test data" "🧪"
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
    msg "$GREEN" "Found $output_files output files" "📄"
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
  if ! $COMPUTE_METRICS; then
    return 0
  fi
  
  msg "$BLUE" "Computing metrics" "📊"
  if [ -f "${SCRIPT_DIR}/validation/compute_metrics.py" ]; then
    python3 "${SCRIPT_DIR}/validation/compute_metrics.py" "$OUTPUT_DIR" "$INPUT_DIR" || {
      msg "$RED" "Failed to compute metrics" "❌"
      errors+=("Failed to compute metrics")
      return 1
    }
    msg "$GREEN" "Metrics computed successfully" "✅"
  else
    msg "$YELLOW" "Metrics script not found, skipping metrics computation" "⚠️"
    warnings+=("Metrics script not found, skipping metrics computation")
  fi
  return 0
}

# Run inference
run_inference() {
  # Generate test data if needed
  generate_test_data || return 1
  
  # Check if input directory has data
  if [ ! "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
    msg "$YELLOW" "Input directory is empty: $INPUT_DIR" "⚠️"
    warnings+=("Input directory is empty")
  fi
  
  # Clean output directory
  rm -rf "${OUTPUT_DIR:?}"/* 2>/dev/null || true
  chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
  
  # GPU flag
  GPU_FLAG=""
  if $USE_GPU; then
    GPU_FLAG="--nv"
  fi
  
  # Run the container
  msg "$BLUE" "Running inference" "🧠"
  
  # Start process in background to monitor performance
  "$CONTAINER_CMD" run $GPU_FLAG \
    --bind "$INPUT_DIR:/input:ro" \
    --bind "$OUTPUT_DIR:/output" \
    "$IMAGE_PATH" &
  
  inference_pid=$!
  
  # Monitor performance
  monitor_performance $inference_pid
  
  # Wait for completion
  wait $inference_pid
  inference_exit_code=$?
  
  if [[ $inference_exit_code -eq 0 ]]; then
    msg "$GREEN" "Inference completed successfully" "✅"
    inference_ran=true
    check_outputs
    compute_metrics
    return 0
  else
    msg "$RED" "Inference failed with exit code $inference_exit_code" "❌"
    errors+=("Inference failed with exit code $inference_exit_code")
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
  # Parse arguments
  parse_args "$@"
  
  # Check environment
  check_env
  
  # Perform requested operation
  case $OPERATION in
    build)
      build_container
      ;;
    test)
      test_container
      save_results
      ;;
    run)
      run_inference
      save_results
      ;;
    validate)
      run_validation
      ;;
    *)
      msg "$RED" "Unknown operation: $OPERATION" "❌"
      exit 1
      ;;
  esac
  
  exit $?
}

# Trap errors
trap 'msg "$RED" "An error occurred. Exiting..." "❌"; save_results; exit 1' ERR

# Execute main function
main "$@"