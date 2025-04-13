# FOMO25 Sanity Check Pipeline

A streamlined validation framework for neuroimaging models using Apptainer/Singularity containers. This pipeline helps you test your machine learning models against the FOMO25 Challenge requirements.

## 📋 Overview

This validation framework helps you:
- Build Apptainer/Singularity containers directly (no Docker required)
- Validate your model's structure and functionality
- Test GPU compatibility and performance
- Verify input/output interfaces
- Compute evaluation metrics

## ⚙️ Requirements

### System Prerequisites
- Apptainer (preferred) or Singularity
- Python 3.x
- Sudo privileges for container building
- NVIDIA GPU with drivers (optional, CPU fallback available)

### Model Requirements
Your neuroimaging model must:
1. Include a `predict.py` script in the root directory
2. Accept input from the `/input` directory (mounted read-only)
3. Write outputs to the `/output` directory
4. Process NIfTI format input files (`.nii` or `.nii.gz`)
5. Handle both GPU and CPU execution
6. List dependencies in `requirements.txt`

## 🚀 Quick Start

```bash
# Build a container
./container.sh -b -n my-container

# Test container structure and functionality
./container.sh -t -n my-container

# Run inference with the container
./container.sh -r -n my-container

# Run complete validation (all tests)
./container.sh -v -n my-container
```

## 📚 Usage Options

The `container.sh` script provides a unified interface for all operations:

```
Usage: ./container.sh [options]
Options:
  -n, --name NAME      Container name (default: fomo25-container)
  -b, --build          Build the container
  -t, --test           Test an existing container
  -r, --run            Run inference on a container
  -v, --validate       Run full validation (test+run+metrics)
  -i, --input DIR      Input directory (default: ./test/input)
  -o, --output DIR     Output directory (default: ./test/output)
  -d, --def FILE       Definition file (default: ./Apptainer.def)
  --no-gpu             Disable GPU support
  --no-generate        Skip test data generation
  --no-metrics         Skip metrics computation
  --result FILE        Specify output JSON file for results
  -h, --help           Show this help
```

### Examples:

```bash
# Build with a custom name
./container.sh -b -n custom-model

# Test with custom definition file
./container.sh -t -n my-model -d custom.def

# Run with specific input/output dirs
./container.sh -r -n my-model -i /path/to/inputs -o /path/to/results

# Validate without GPU support
./container.sh -v -n my-model --no-gpu
```

## 🔍 Validation Process

### Structure Tests
The script verifies:
- Container existence and accessibility
- Presence of required files (`/app/predict.py`)
- Basic command execution capability
- GPU support detection

### Runtime Tests
For inference testing, the pipeline:
1. Generates synthetic test data if needed
2. Mounts input/output directories
3. Runs prediction with performance monitoring
4. Verifies output files were generated
5. Computes evaluation metrics

## 🛠️ Project Structure

```
.
├── Apptainer.def          # Container definition
├── container.sh           # Main script
├── src/                   # Source code directory
│   └── predict.py         # Required prediction script
├── test/                  # Test directories
│   ├── input/             # Input data
│   └── output/            # Output data
└── validation/            # Validation tools
    ├── compute_metrics.py # Metrics computation
    └── test_data_generator.py # Test data generation
```

## 📊 Output & Metrics

After validation, the results are saved to a JSON file (default: `validation_result.json`). This includes:

- Status: `PASSED` or `FAILED`
- Detailed check results (container structure, GPU support, etc.)
- Errors and warnings
- Performance metrics (memory usage, execution time)

If the `--no-metrics` flag is not used, detailed segmentation metrics are computed and saved to:
- `test/output/results/metrics_results.json`
- `test/output/results/metrics_results.csv`

## 🔧 Troubleshooting

### Container Build Issues
- Verify Apptainer/Singularity is installed
- Check your `requirements.txt` for compatibility issues
- Ensure you have sudo privileges

### Validation Failures
- Container not found: Build it first with `-b` flag
- predict.py not found: Ensure it exists at the root level of your source code
- No output files: Make sure your model writes to the `/output` directory
- GPU not detected: Install NVIDIA drivers or use `--no-gpu`

## 📝 Note for Challenge Participants

This validation pipeline ensures your model will function correctly in the FOMO25 Challenge environment. Pass all validation checks to confirm your submission will be evaluated properly.

A successful validation confirms:
1. Your container can be built
2. Your model can run inference
3. I/O paths are correctly configured
4. Output format meets requirements

## 🔗 Contributing

To modify this pipeline:
1. Edit the `container.sh` script for pipeline changes
2. Modify `Apptainer.def` for container configuration
3. Update validation scripts in the `validation/` directory

## 📄 License

This pipeline is provided for use in the FOMO25 Challenge.

