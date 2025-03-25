# Container Sanity Check for Pre-trained MONAI Models

A validation tool for testing containerized pre-trained models in medical imaging.

## Description

This project provides a comprehensive validation framework for testing containers with trained computer vision models, particularly focused on medical imaging applications. The tool performs sanity checks by generating synthetic test data, running inference within the container, and validating the outputs. It's designed to ensure that containerized models are properly packaged and functioning correctly before deployment in production environments.

The framework includes both a validation tool and a participant template structure with a sample MONAI UNet implementation for 3D medical image processing.

## Getting Started

### Dependencies

* Python +3.10
* Apptainer (formerly Singularity) or Singularity container runtime
* For development:
  * nibabel
  * numpy

### Installing

1. Clone the repository:
```bash
git clone https://github.com/pablorocg/fomo25-sanity-check-pipeline.git
cd fomo25-sanity-check-pipeline
```

2. Install the required Python packages:
```bash
conda create -n fomo25env
conda activate fomo25env
pip install nibabel numpy
```


3. Build the container:
```bash
# apptainer build <name-of-the-image.sif> <name-of-the-definition-file.def>
apptainer build apptainer-images/image.sif participant-template/container.def 
```

### Executing program

#### Running the validation tool

The validation tool can be used to test containers with trained models:

```bash
python container_sanity_check.py apptainer-images/image.sif
```

Additional options:
```bash
python container_sanity_check.py /path/to/your/container.sif --no-gpu --keep-files --output-dir /path/to/output
```

Options:
* `--no-gpu`: Run with CPU instead of GPU
* `--keep-files`: Preserve temporary files after validation
* `--output-dir`: Specify a directory for temporary files
* `-v`, `--verbose`: Enable verbose output

#### Using the participant template

1. Add your pre-trained model to the participant template or use the randomly initialized model.

2. Build the container:
```
cd participant-template
apptainer build ../my_model.sif container.def
```

3. Run inference directly with the container:
```
apptainer run my_model.sif /path/to/input.nii.gz /path/to/output.nii.gz cuda
```

## Help

### Common Issues

1. **Apptainer/Singularity not found**:
   - Ensure that Apptainer or Singularity is installed on your system
   - Add it to your PATH if necessary

2. **CUDA/GPU issues**:
   - Use the `--no-gpu` flag if you don't have a compatible GPU
   - Ensure NVIDIA drivers are properly installed for GPU usage

3. **Container build failures**:
   - Check that all required files are in the correct locations
   - Verify that you have root or sudo access for building containers

### Debugging

To enable verbose logging:
```
python container_sanity_check.py /path/to/your/container.sif -v
```

## Version History

* 0.1
  * Initial Release with MONAI UNet implementation

## Acknowledgments

* [MONAI](https://monai.io/) - Medical imaging deep learning framework
* [PyTorch](https://pytorch.org/) - Deep learning framework
* [Apptainer/Singularity](https://apptainer.org/) - Container platform for scientific computing