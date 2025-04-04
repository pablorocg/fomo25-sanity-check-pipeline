# FOMO25 Validation Pipeline

This framework lets you validate any containerized code against a set of specific requirements using our validation script. The validation script checks that the code meets these conditions:
  
- The container file exists and can be executed using Apptainer (or Singularity).
- Essential scripts (e.g. the prediction script at `/app/predict.py`) are present in the container.
- Basic container commands run successfully.
- GPU support is available (if applicable) or the script continues in CPU mode.
- The container handles input via the `/input` mount point and writes outputs to `/output`.
- Inference runs correctly, generating outputs that meet the expected quality and format.
- Performance metrics are computed and saved as a JSON result.

## Validation Overview

The validation process performs several checks:
- **Container Structure & Runtime:** Verifies container file existence, proper structure (e.g. `/app/predict.py`), and runtime availability.
- **Basic Command Execution:** Tests that the container runs simple commands internally.
- **GPU Support:** Optionally checks GPU availability and falls back gracefully if not present.
- **Inference Execution:** Runs the supplied container using synthetic test data mounted from `/test/input` and confirms output generation in `/test/output`.
- **Metrics Computation:** Computes and saves standard metrics (like dice, accuracy) in a JSON format.

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
   ./do_test_run.sh [OPTIONS]
   ```
   Options include skipping container rebuild, test data generation, inference or metrics computations. Use `--result your_result.json` to specify an output results file.

## Usage Guidelines

- Make sure your container has the core prediction script at `/app/predict.py`.
- The container should accept inputs via the `/input` mount and write outputs to `/output`.
- The validation script returns a JSON summary with status (`PASSED` or `FAILED`), along with detailed checks, errors, and warnings.
- Follow the specific requirements described in this README to ensure your code is validated correctly.

## Troubleshooting

- **Build Failures:** Verify the Apptainer/Singularity installation; check your definition file syntax.
- **GPU Check Issues:** Confirm the host has an NVIDIA GPU and drivers configured, or use the `--no-gpu` flag to proceed.
- **Missing Outputs:** Ensure that the input directory has valid test data and that your container writes files to `/output`.

By following these guidelines, the validation script can be used to check any code against the cluster of specific container requirements.

