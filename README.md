# FOMO25 Sanity Check Pipeline

A comprehensive validation framework for FOMO25 containerized models. This pipeline provides automated testing and validation of ML model containers, focused on medical imaging applications.

## Overview

The FOMO25 Sanity Check Pipeline validates containerized ML models by:

- Verifying container structure and required scripts
- Checking GPU support and functionality
- Running inference with synthetic test data
- Computing performance metrics
- Validating output formats and quality

## Requirements

- Apptainer (formerly Singularity) 1.1.0+
- Python 3.7+
- NVIDIA GPU with drivers (optional)
- Python packages:
  - nibabel
  - numpy
  - pandas
  - psutil

## Directory Structure

```
FOMO25-SANITY-CHECK-PIPELINE/
│
├── apptainer_images/         # Container images
│   └── fomo25-baseline-container.sif
│
├── src/                      # Core functionality
│   ├── pipeline.py
│   ├── predict.py
│   ├── Apptainer.def
│   └── requirements.txt
│
├── validation/               # Validation tools
│   ├── command_runner.py
│   ├── compute_metrics.py
│   ├── container_handler.py
│   ├── container_sanity_check.py
│   ├── performance_monitor.py
│   ├── test_data_generator.py
│   └── validation_result.py
│
├── test/                     # Test data
│   ├── input/
│   └── output/
│
├── do_build.sh               # Build container script
├── do_test_run.sh            # Test execution script
└── README.md
```

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/FOMO25-SANITY-CHECK-PIPELINE.git
   cd FOMO25-SANITY-CHECK-PIPELINE
   ```

2. Build the container:
   ```bash
   ./do_build.sh
   ```

3. Run the validation:
   ```bash
   ./do_test_run.sh
   ```

## Usage

The main execution script is `do_test_run.sh`, which manages the entire validation process:

```bash
./do_test_run.sh [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-n, --name CONTAINER_NAME` | Specify container name (default: fomo25-baseline-container) |
| `-d, --def DEF_FILE` | Specify Apptainer definition file (default: Apptainer.def) |
| `--no-rebuild` | Skip container rebuilding |
| `--no-generate` | Skip test data generation |
| `--no-inference` | Skip inference execution |
| `--no-metrics` | Skip metrics computation |
| `--no-cleanup` | Keep temporary files |
| `--no-gpu` | Disable GPU checks and usage |
| `--result RESULT_FILE` | Specify output JSON file for results |

### Example

Test with a custom container and skip rebuilding:

```bash
./do_test_run.sh --name custom-model-container --no-rebuild
```

## Validation Checks

The pipeline performs the following validation checks:

1. **Container Structure**
   - Verifies container file exists and is accessible
   - Checks that Apptainer/Singularity is installed
   - Verifies `/app/predict.py` exists inside the container

2. **GPU Support**
   - Tests if the container can access GPU resources
   - Verifies CUDA functionality via `nvidia-smi`

3. **Inference Execution**
   - Generates synthetic test data
   - Mounts input/output directories
   - Executes model prediction
   - Monitors memory usage and execution time

4. **Output Validation**
   - Verifies output files are generated
   - Validates output format and quality
   - Computes standard metrics (for segmentation tasks)

## Output

The validation results are saved to a JSON file with the following structure:

```json
{
  "status": "PASSED|FAILED",
  "checks": {
    "container_exists": true|false,
    "container_runtime_available": true|false,
    "can_run_basic_commands": true|false,
    "required_files_present": true|false,
    "gpu_supported": true|false,
    "inference_ran": true|false,
    "outputs_generated": true|false
  },
  "errors": [
    "Error message 1",
    "Error message 2"
  ],
  "warnings": [
    "Warning message 1",
    "Warning message 2"
  ]
}
```

## Container Requirements

For a container to pass validation, it must:

1. Have the core prediction script at `/app/predict.py`
2. Accept input via the `/input` mount point
3. Write output to the `/output` mount point
4. Follow proper format for medical imaging data

## Troubleshooting

### Common Issues

1. **Container build fails**
   - Verify Apptainer is installed correctly
   - Check for syntax errors in the definition file
   - Ensure internet connectivity for base image pulling

2. **GPU checks fail**
   - Verify NVIDIA drivers are installed
   - Check if CUDA is working properly on the host
   - Ensure the container has the correct CUDA libraries

3. **No output generated**
   - Check container logs for errors
   - Verify input directory contains valid test data
   - Check container specification against requirements

