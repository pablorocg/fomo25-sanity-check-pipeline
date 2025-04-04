#!/bin/bash
# Enhanced container testing and metric computation script with emoji indicators
# Color codes for better console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored status messages with emojis
print_status() {
  local color="$1"
  local message="$2"
  local emoji="$3"
  echo -e "${color}${emoji} ${message}${NC}"
}

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Initialize variables with default values
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
CONTAINER_NAME="fomo25-baseline-container"
DEF_FILE="Apptainer.def"
CONTAINER_PATH="${SCRIPT_DIR}/apptainer_images/${CONTAINER_NAME}.sif"
REBUILD=true
GENERATE_DATA=true
RUN_INFERENCE=true
COMPUTE_METRICS=true
GPU_CHECK=true
RESULT_FILE="${SCRIPT_DIR}/validation_result.json"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -n|--name) CONTAINER_NAME="$2"; shift ;;
    -d|--def) DEF_FILE="$2"; shift ;;
    --no-rebuild) REBUILD=false ;;
    --no-generate) GENERATE_DATA=false ;;
    --no-inference) RUN_INFERENCE=false ;;
    --no-metrics) COMPUTE_METRICS=false ;;
    --no-gpu) GPU_CHECK=false ;;
    --result) RESULT_FILE="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Update container path with provided name
CONTAINER_PATH="${SCRIPT_DIR}/apptainer_images/${CONTAINER_NAME}.sif"

# Directories
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# Create required directories
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Initialize validation results
container_exists=false
container_runtime_available=false
can_run_basic_commands=false
required_files_present=false
gpu_supported=false
inference_ran=false
outputs_generated=false
errors=()
warnings=()

# Check if container runtime is available
check_container_runtime() {
  local runtime_cmd=""
  
  if command_exists apptainer; then
    runtime_cmd="apptainer"
  elif command_exists singularity; then
    runtime_cmd="singularity"
    warnings+=("Using singularity instead of apptainer")
  else
    errors+=("Neither apptainer nor singularity found")
    return 1
  fi
  
  print_status "$BLUE" "Using container runtime: $runtime_cmd" "🐳"
  container_runtime_available=true
  return 0
}

# Check if container exists
check_container_exists() {
  if [[ -f "$CONTAINER_PATH" ]]; then
    print_status "$GREEN" "Container found: $CONTAINER_PATH" "✅"
    container_exists=true
    return 0
  else
    errors+=("Container not found: $CONTAINER_PATH")
    print_status "$RED" "Container not found: $CONTAINER_PATH" "❌"
    return 1
  fi
}

# Run a basic command in the container
test_basic_command() {
  print_status "$BLUE" "Testing if container runs basic commands..." "🔍"
  
  if apptainer exec "$CONTAINER_PATH" echo "Container is working" >/dev/null 2>&1; then
    print_status "$GREEN" "Container can run basic commands" "✅"
    can_run_basic_commands=true
    return 0
  else
    errors+=("Container failed to run basic command")
    print_status "$RED" "Container failed to run basic command" "❌"
    return 1
  fi
}

# Check container structure
check_container_structure() {
  print_status "$BLUE" "Checking container file structure..." "🔍"
  
  if apptainer exec "$CONTAINER_PATH" test -f /app/predict.py; then
    print_status "$GREEN" "Found prediction script: /app/predict.py" "✅"
    required_files_present=true
    return 0
  else
    errors+=("Required file /app/predict.py not found in container")
    print_status "$RED" "Required file /app/predict.py not found in container" "❌"
    return 1
  fi
}

# Check GPU support
check_gpu_support() {
  if ! $GPU_CHECK; then
    print_status "$YELLOW" "GPU check skipped as --no-gpu was specified" "⚠️"
    return 0
  fi
  
  print_status "$BLUE" "Checking GPU support..." "🔍"
  
  # Try to run nvidia-smi in the container
  if apptainer exec --nv "$CONTAINER_PATH" nvidia-smi >/dev/null 2>&1; then
    print_status "$GREEN" "GPU support verified with nvidia-smi" "🖥️"
    gpu_supported=true
    return 0
  else
    # Try alternative method
    local gpu_env=$(apptainer exec --nv "$CONTAINER_PATH" bash -c 'python -c "import os; print(\"GPU_AVAILABLE=\" + str(\"NVIDIA_VISIBLE_DEVICES\" in os.environ))"' 2>/dev/null)
    
    if [[ "$gpu_env" == *"GPU_AVAILABLE=True"* ]]; then
      print_status "$GREEN" "GPU support verified through environment variables" "🖥️"
      gpu_supported=true
      return 0
    else
      warnings+=("Container cannot access GPU, continuing with CPU")
      print_status "$YELLOW" "Container cannot access GPU, continuing with CPU" "⚠️"
      return 1
    fi
  fi
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
  
  print_status "$BLUE" "Performance metrics:" "📊"
  print_status "$BLUE" "  Memory usage: $((max_memory / 1024)) MB" "💾"
  print_status "$BLUE" "  Execution time: ${elapsed_time} seconds" "⏱️"
  
  return 0
}

# Check if outputs were generated
check_outputs() {
  print_status "$BLUE" "Checking for generated outputs..." "🔍"
  
  local output_files=$(find "$OUTPUT_DIR" -type f | wc -l)
  
  if [[ $output_files -gt 0 ]]; then
    print_status "$GREEN" "Found $output_files output files" "📄"
    outputs_generated=true
    return 0
  else
    errors+=("No output files generated")
    print_status "$RED" "No output files generated" "❌"
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

  print_status "$BLUE" "Results saved to $RESULT_FILE" "💾"
}

# Main execution
main() {
  print_status "$BLUE" "Starting validation process" "🚀"
  
  # Initial checks
  check_container_runtime || print_status "$YELLOW" "Continuing with limited functionality" "⚠️"
  
  # Rebuild container if requested
  if $REBUILD; then
    print_status "$BLUE" "Rebuilding container" "🔄"
    . "${SCRIPT_DIR}/do_build.sh" "$CONTAINER_NAME" "$DEF_FILE" || {
      errors+=("Failed to build container")
      print_status "$RED" "Failed to build container" "❌"
      save_results
      exit 1
    }
  fi
  
  # Check if container exists after potential rebuild
  check_container_exists || {
    save_results
    exit 1
  }
  
  # Test basic command execution
  test_basic_command || print_status "$YELLOW" "Container may have limited functionality" "⚠️"
  
  # Check container structure
  check_container_structure || print_status "$YELLOW" "Container structure validation failed" "⚠️"
  
  # Check GPU support
  check_gpu_support || print_status "$YELLOW" "GPU support not available, continuing with CPU" "⚠️"
  
  # Generate test data if input directory is empty
  if $GENERATE_DATA && [ ! "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
    print_status "$YELLOW" "Generating synthetic test data" "🧪"
    python3 "${SCRIPT_DIR}/validation/test_data_generator.py" "$INPUT_DIR" || {
      errors+=("Failed to generate test data")
      print_status "$RED" "Failed to generate test data" "❌"
      save_results
      exit 1
    }
  fi
  
  # Run container inference
  if $RUN_INFERENCE; then
    print_status "$GREEN" "Running container inference" "🧠"
    
    # Start process in background to monitor performance
    apptainer run \
      --contain \
      --nv \
      --bind "$INPUT_DIR":/input:ro \
      --bind "$OUTPUT_DIR":/output \
      "${CONTAINER_PATH}" &
    
    inference_pid=$!
    
    # Monitor performance
    monitor_performance $inference_pid
    
    # Wait for completion
    wait $inference_pid
    inference_exit_code=$?
    
    if [[ $inference_exit_code -eq 0 ]]; then
      print_status "$GREEN" "Inference completed successfully" "✅"
      inference_ran=true
    else
      errors+=("Inference failed with exit code $inference_exit_code")
      print_status "$RED" "Inference failed with exit code $inference_exit_code" "❌"
      save_results
      exit 1
    fi
    
    # Check if outputs were generated
    check_outputs || print_status "$YELLOW" "No outputs generated" "⚠️"
  fi
  
  # Compute metrics
  if $COMPUTE_METRICS; then
    print_status "$BLUE" "Computing metrics" "📊"
    python3 "${SCRIPT_DIR}/validation/compute_metrics.py" "$OUTPUT_DIR" "$INPUT_DIR" || {
      errors+=("Failed to compute metrics")
      print_status "$RED" "Failed to compute metrics" "❌"
      save_results
      exit 1
    }
    print_status "$GREEN" "Metrics computed successfully" "✅"
  fi
  
  # Save validation results
  save_results
  
  # Final status
  if [[ ${#errors[@]} -eq 0 ]]; then
    print_status "$GREEN" "Test run complete - ALL CHECKS PASSED" "🎉"
    exit 0
  else
    print_status "$YELLOW" "Test run complete - ${#errors[@]} CHECKS FAILED" "⚠️"
    for error in "${errors[@]}"; do
      print_status "$RED" "ERROR: $error" "❌"
    done
    exit 1
  fi
}

# Trap errors
trap 'print_status "$RED" "An error occurred. Exiting..." "❌"; save_results; exit 1' ERR

# Execute main function
main