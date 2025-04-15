# FOMO25 Challenge: Improved Submission Guide

Welcome to the FOMO25 Foundation Models Challenge! This improved guide will walk you through preparing and validating your submission, with clear instructions that you should adapt to your specific model and code.

## Table of Contents

1. [Overview and Core Requirements](#overview-and-core-requirements)
2. [Installing Apptainer](#installing-apptainer)
3. [Submission Structure](#submission-structure)
4. [Creating Your Apptainer Container](#creating-your-apptainer-container)
5. [Prediction Script Requirements](#prediction-script-requirements)
6. [Validating Your Submission](#validating-your-submission)
7. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
8. [Submission Checklist](#submission-checklist)

## Overview and Core Requirements

The FOMO25 challenge requires all participants to submit their models in an **Apptainer container**. This standardized approach ensures fair and reproducible evaluation. Your container will run on our evaluation cluster **without internet access**, so it must include all dependencies your specific model needs.

**Requirements:**
- An Apptainer container with your model and all dependencies your model requires (.sif file)
- A file named `/app/predict.py` as the entry point (this path and filename cannot be changed)
- Ability to read input from `/input` and write to `/output` directories
- Compatibility with both GPU and CPU execution

## Installing Apptainer

Apptainer is the container platform required for this challenge. It lets you package your model and dependencies in a standardized environment.

To install Apptainer on your specific operating system:
1. Visit the [official Apptainer installation documentation](https://apptainer.org/docs/admin/main/installation.html)
2. Follow the instructions for your operating system (Linux, macOS, or Windows)

After installation, verify Apptainer works on your system:

```bash
apptainer --version
```

## Submission Structure

Your submission must follow this basic structure, but you will need to adapt the content based on your specific model:

```
my-submission/
├── Apptainer.def         # Definition file YOU create (example provided below)
├── src/                  # YOUR source code (folder name can be different)
│   ├── predict.py        # YOUR main entry point (MUST be named predict.py)
│   ├── models/           # YOUR model implementation
│   └── ...               # YOUR other code files
├── requirements.txt      # YOUR Python dependencies
└── ...                   # Any other files YOUR model needs
```

**Important Note:** The folder structure in your local development environment can be different, but when building the container, your code must be organized to have `predict.py` in the `/app` directory.

## Creating Your Apptainer Container

### 1. Write an Apptainer Definition File

Create a file named `Apptainer.def` in your project directory. Below is an example that you must modify for your specific model:

```
Bootstrap: docker
From: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# CUSTOMIZE: Change the base image if you use a different framework (e.g., TensorFlow) 
# or need a different CUDA version for your specific model

%labels
    Author <YOUR Name>
    Version v1.0
    Description FOMO25 Challenge Submission

%environment
    export PYTHONUNBUFFERED=1
    export LC_ALL=C.UTF-8
    # ADD any additional environment variables YOUR model needs

%files
    # CUSTOMIZE: Replace "src" with the actual folder containing YOUR code
    src /app
    
    # CUSTOMIZE: Replace with YOUR actual requirements file
    requirements.txt /app/requirements.txt
    
    # ADD any additional files YOUR model needs (e.g., pre-trained weights)
    # example: model_weights.pth /app/model_weights.pth

%post
    # DO NOT CHANGE: These directories are required
    mkdir -p /input /output

    # CUSTOMIZE: Install YOUR specific dependencies
    pip install --no-cache-dir -U pip setuptools wheel
    pip install --no-cache-dir -r /app/requirements.txt
    
    # ADD any additional installation commands YOUR model needs
    # example: apt-get update && apt-get install -y libsm6 libxext6
    
    # DO NOT CHANGE: Make the prediction script executable
    chmod +x /app/predict.py

%test
    # Basic checks to verify your container
    python -c "import sys; print(f'Python {sys.version}')"
    if [ -f "/app/predict.py" ]; then
        echo "✓ Found predict.py"
    else
        echo "✗ predict.py not found"
        exit 1
    fi
    
    # CUSTOMIZE: Add any specific tests for YOUR model
    # example: python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

**Critical Customization Points:**
- Base image: Choose one compatible with your model's framework and CUDA requirements
- Files section: Update paths to match your code organization
- Post section: Install only the dependencies your model needs
- Add any model-specific files, like pre-trained weights

### 2. Include Your Dependencies

Create a `requirements.txt` file listing only the Python packages your model requires:

```
# EXAMPLE ONLY - REPLACE with YOUR actual dependencies
torch
torchvision
monai
nibabel
numpy
pandas
scikit-learn
```

**Important:** Do not copy the example list directly. Only include packages your specific model uses.

### 3. Build Your Container

```bash
sudo apptainer build <your-model-name>.sif Apptainer.def
```

This creates a single `.sif` file containing your model and all its dependencies.

## Prediction Script Requirements

The `predict.py` script is the mandatory entry point for your container. You must implement it to:

1. Accept input data from the `/input` directory
2. Process this data using YOUR model
3. Write results to the `/output` directory
4. Support the specific command-line interface described below

### Command-Line Interface Requirements (MANDATORY)

Your script **must** support this exact command-line interface:

```bash
python /app/predict.py --input /path/to/input/file.nii.gz --output /path/to/output/file.nii.gz
```

The evaluation system will call your script with these parameters:
- `--input`: Path to a specific input NIfTI file
- `--output`: Path where your script should save the output file

**Important Implementation Requirements:**

1. Your script must detect and use a GPU if available, but also work on CPU
2. It must process NIfTI files (`.nii` or `.nii.gz`) from the provided input path
3. Output files must be saved at the specified output path
4. The script should handle errors gracefully

Here is a minimal example structure for your `predict.py` (you must adapt this to your model):

```python
import argparse
import os
import torch
import nibabel as nib

def parse_args():
    parser = argparse.ArgumentParser(description='Process NIfTI files for FOMO25 challenge')
    parser.add_argument('--input', required=True, help='Path to input NIfTI file')
    parser.add_argument('--output', required=True, help='Path to save output NIfTI file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    input_path = args.input
    output_path = args.output
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return 1
        
    try:
        # Load NIfTI file
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # REPLACE THIS SECTION with YOUR model's inference code
        # --------------------------------------------------
        # Determine device (GPU if available, otherwise CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load YOUR model (this is just an example)
        # model = YourModel().to(device)
        # model.load_state_dict(torch.load('/app/model_weights.pth', map_location=device))
        # model.eval()
        
        # Process with YOUR model
        # input_tensor = torch.from_numpy(data).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     output_data = model(input_tensor).cpu().numpy().squeeze()
        # --------------------------------------------------
        
        # For this example, we're just creating dummy output
        # REMOVE this and use YOUR actual model output
        output_data = data  # Replace with your model's output
        
        # Save output
        output_img = nib.Nifti1Image(output_data, img.affine, img.header)
        nib.save(output_img, output_path)
        print(f"Saved output to {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
```

**Critical Reminder:** The example above is a template. You must replace the model loading and inference sections with code specific to your model.

## Validating Your Submission

The FOMO25 Challenge provides a comprehensive validation framework to ensure your container meets all requirements before submission. This framework tests various aspects of your container, including:

1. Container structure and accessibility
2. Presence of required files (`/app/predict.py`)
3. Basic command execution capability
4. GPU support detection
5. Correct handling of input/output paths
6. End-to-end inference testing
7. Performance metrics computation

### Setting Up the Validation Framework

To properly validate your submission, you need to set up the complete validation framework:

```bash
# Clone the validation repository
git clone https://github.com/pablorocg/fomo25-sanity-check-pipeline.git
cd fomo25-sanity-check-pipeline

# Install required dependencies
pip install -r requirements_.txt  # Note the underscore in the filename
```

Alternatively, if you prefer not to clone the entire repository, you can download just the necessary files:

```bash
# Create directories
mkdir -p fomo25-validation/validation

# Download main validation script
wget -O fomo25-validation/container.sh https://raw.githubusercontent.com/pablorocg/fomo25-sanity-check-pipeline/main/container.sh
chmod +x fomo25-validation/container.sh

# Download validation scripts
wget -O fomo25-validation/validation/compute_metrics.py https://raw.githubusercontent.com/pablorocg/fomo25-sanity-check-pipeline/main/validation/compute_metrics.py
wget -O fomo25-validation/validation/test_data_generator.py https://raw.githubusercontent.com/pablorocg/fomo25-sanity-check-pipeline/main/validation/test_data_generator.py
chmod +x fomo25-validation/validation/*.py

# Download requirements
wget -O fomo25-validation/requirements_.txt https://raw.githubusercontent.com/pablorocg/fomo25-sanity-check-pipeline/main/requirements_.txt

# Install dependencies
pip install -r fomo25-validation/requirements_.txt

# Create test directories
mkdir -p fomo25-validation/test/input fomo25-validation/test/output

# Navigate to the validation directory
cd fomo25-validation
```

### Creating a Configuration File

For easier validation, create a `config.yml` file in the validation directory:

```yaml
# FOMO25 Container Configuration
container:
  path: "/path/to/your-model.sif"  # CUSTOMIZE: Path to YOUR container
  command: "apptainer"             # Change only if you have a custom path to run Apptainer

# Resource settings
use_gpu: true                      # Set to false to disable GPU testing

# Directory paths that are going to be mounted on your container
directories:
  input: "test/input"              # Path to your test data
  output: "test/output"            # Path for output
```

### Running the Validation

With the framework set up, you can run the validation:

```bash
# For full validation (test structure, run inference, compute metrics)
./container.sh -v -n your-model

# Using a configuration file
./container.sh --config config.yml

# Test only container structure
./container.sh -t -n your-model

# Run only inference
./container.sh -r -n your-model
```

### Understanding the Validation Process

The validation framework performs these steps:

1. **Structure Tests**:
   - Verifies container exists and is accessible
   - Checks for required files like `/app/predict.py`
   - Tests basic command execution
   - Detects GPU support

2. **Runtime Tests**:
   - Generates synthetic test data if none is provided (using `validation/test_data_generator.py`)
   - Mounts input/output directories to your container
   - Runs your prediction script with performance monitoring
   - Verifies output files are generated

3. **Metrics Computation**:
   - Computes standard segmentation metrics using `validation/compute_metrics.py`
   - Saves results to JSON and CSV files in the output directory

### Validation Results

After running validation, results are saved to a JSON file (default: `validation_result.json`). This includes:

- Overall status: `PASSED` or `FAILED`
- Detailed check results
- Errors and warnings
- Performance metrics

Additionally, if your model generates valid segmentation outputs, detailed metrics will be available in:
- `test/output/results/metrics_results.json`
- `test/output/results/metrics_results.csv`

These metrics help you assess your model's performance before final submission.

### Troubleshooting Validation Issues

If validation fails, check these common problems:

- **Missing Dependencies**: Ensure all dependencies in `requirements_.txt` are installed
- **Missing Validation Scripts**: Verify you have all required files in the validation directory
- **Container Not Found**: Check the path to your .sif file in the config.yml
- **No Test Data**: If your test/input directory is empty, the validation will attempt to generate test data
- **predict.py Not Found**: Ensure your container has the script at `/app/predict.py`
- **No Output Files**: Check that your model writes to the `/output` directory
- **Permission Issues**: Ensure the validation scripts have execution permissions (`chmod +x`)

## Common Issues and Troubleshooting

### Installation Problems

- **Error**: "Failed to build container: sudo: apptainer: command not found"
  - **Solution**: Verify Apptainer is properly installed and in your PATH

- **Error**: "FATAL: kernel too old"
  - **Solution**: Use a more recent Linux kernel or add `--fakeroot` when building

### Container Build Issues

- **Error**: "Failed to build container: [Package] not found"
  - **Solution**: Verify the base image includes necessary libraries or install them in the `%post` section

- **Error**: "no space left on device"
  - **Solution**: Free up disk space or use `APPTAINER_TMPDIR` to specify a directory with more space

### Validation Failures

- **Error**: "predict.py not found in container"
  - **Solution**: Ensure your code is correctly copied to `/app` in the definition file

- **Error**: "Inference failed"
  - **Solution**: Check for errors in your predict.py script, especially in input/output paths and GPU handling

- **Error**: "No output files generated"
  - **Solution**: Verify your script saves files to the specified output path

## Submission Checklist

Before submitting, ensure:

- [ ] Your container builds successfully with Apptainer
- [ ] Container includes all dependencies your model needs (remember: no internet during evaluation)
- [ ] `/app/predict.py` exists and processes files correctly with the required command-line interface
- [ ] Your code handles both CPU and GPU execution
- [ ] All validation tests pass successfully
- [ ] Output format matches the expected format for the challenge

## Final Submission

Once validated, follow these steps to submit:

1. Name your final SIF file according to the challenge guidelines
2. Upload your SIF file to the FOMO25 submission portal.
3. Complete the submission form with details about your approach
4. Submit and wait for the confirmation email

For any questions or issues, please contact the FOMO25 challenge organizers.

Good luck with your submission!