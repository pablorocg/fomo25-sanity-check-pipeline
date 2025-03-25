#!/usr/bin/env python3
"""
Container Validator for Pre-trained Models

This script validates containers with pre-trained models by:
1. Checking if the container runs properly
2. Generating dummy test data
3. Running inference using the container's built-in model
4. Validating outputs and calculating metrics
"""

import os
import sys
import time
import tempfile
import subprocess
import logging
import argparse
import shutil
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandRunner:
    """Handles running shell commands and capturing outputs."""
    
    @staticmethod
    def run(cmd, check=True, shell=False):
        """Run a command and return the result."""
        cmd_str = cmd if shell else ' '.join(cmd) if isinstance(cmd, list) else cmd
        logger.info(f"Running command: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                check=check,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.stdout:
                logger.info(result.stdout.strip())
            if result.stderr:
                logger.warning(result.stderr.strip())
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            if check:
                raise
            return e

class ApptainerHandler:
    """Handler for Apptainer/Singularity operations."""
    
    def __init__(self):
        """Initialize and check for Apptainer/Singularity."""
        self.cmd = self._find_command()
    
    def _find_command(self):
        """Find the Apptainer/Singularity command."""
        for cmd in ["apptainer", "singularity"]:
            try:
                result = CommandRunner.run([cmd, "--version"], check=False)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                pass
        
        logger.error("Neither Apptainer nor Singularity found")
        return None
    
    def is_available(self):
        """Check if Apptainer/Singularity is available."""
        return self.cmd is not None
    
    def run_command(self, container_path, command, binds=None, gpu=False):
        """Run a command in the container."""
        if not self.is_available():
            logger.error("Apptainer/Singularity not available")
            return None
        
        cmd = [self.cmd, "exec"]
        
        if gpu:
            cmd.append("--nv")
        
        if binds:
            for host_path, container_mount in binds.items():
                cmd.extend(["-B", f"{host_path}:{container_mount}"])
        
        # Add the container path
        cmd.append(container_path)  
        
        # Then add the command
        cmd.extend(command)
        
        return CommandRunner.run(cmd, check=False)

class TestDataGenerator:
    """Generates test data for validation."""
    
    @staticmethod
    def generate_nifti_data(output_dir, num_samples=3, size=64):
        """Generate synthetic NIfTI files for testing."""
        os.makedirs(output_dir, exist_ok=True)
        file_paths = []
        
        # Try to use nibabel if available
        try:
            import nibabel as nib
            
            for i in range(num_samples):
                # Create random 3D data
                data = np.random.rand(size, size, size).astype(np.float32)
                
                # Create affine matrix (identity)
                affine = np.eye(4)
                
                # Create NIfTI image
                img = nib.Nifti1Image(data, affine)
                
                # Save to file
                file_path = os.path.join(output_dir, f"test_sample_{i:03d}.nii.gz")
                nib.save(img, file_path)
                file_paths.append(file_path)
                
                logger.info(f"Generated NIfTI file: {file_path}")
            
        except ImportError:
            # If nibabel is not available, create dummy files
            logger.warning("nibabel not available, creating simple binary files")
            
            for i in range(num_samples):
                # Create a simple binary file with .nii.gz extension
                file_path = os.path.join(output_dir, f"test_sample_{i:03d}.nii.gz")
                
                # Generate random binary data with appropriate size
                data = np.random.rand(size, size, size).astype(np.float32).tobytes()
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                file_paths.append(file_path)
                logger.info(f"Generated binary file: {file_path}")
        
        return [os.path.basename(f) for f in file_paths]

class ContainerValidator:
    """Validates container with pre-trained model."""
    
    def __init__(self, container_path, workdir=None, gpu=True):
        """Initialize validator."""
        self.container_path = container_path
        self.workdir = workdir or tempfile.mkdtemp()
        self.use_gpu = gpu
        self.apptainer = ApptainerHandler()
        
        # Create directories
        self.input_dir = os.path.join(self.workdir, "input")
        self.output_dir = os.path.join(self.workdir, "output")
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.workdir) and self.workdir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.workdir)
            logger.info(f"Cleaned up temporary directory: {self.workdir}")
    
    def check_container(self):
        """Check if the container exists and can run."""
        if not os.path.exists(self.container_path):
            logger.error(f"Container file not found: {self.container_path}")
            return False
            
        if not self.apptainer.is_available():
            logger.error("Apptainer/Singularity not installed")
            return False
            
        # Test if container can run a basic command
        logger.info("Testing if container runs...")
        result = self.apptainer.run_command(
            self.container_path, 
            ["echo", "Container is working"]
        )
        
        if result is None or result.returncode != 0:
            logger.error("Container cannot run basic commands")
            return False
            
        return True
    
    def generate_test_data(self):
        """Generate test data for inference."""
        logger.info("Generating test data...")
        input_files = TestDataGenerator.generate_nifti_data(self.input_dir)
        
        if not input_files:
            logger.error("Failed to generate test data")
            return []
            
        return input_files
    
    def run_inference(self, input_files):
        """Run inference using the container's pre-trained model."""
        logger.info("Running inference with pre-trained model...")
        all_succeeded = True
        
        for input_file in input_files:
            input_path = f"/input/{input_file}"
            output_name = f"{os.path.splitext(os.path.splitext(input_file)[0])[0]}_pred.nii.gz"
            output_path = f"/output/{output_name}"
            
            logger.info(f"Processing {input_file}...")
            
            # Command to run inference using the container's internal model
            cmd = ["python", "/app/predict.py", "--input", input_path, "--output", output_path]
            
            # Add --device cpu if GPU is disabled
            if not self.use_gpu:
                cmd.extend(["--device", "cpu"])
                
            start_time = time.time()
            result = self.apptainer.run_command(
                self.container_path,
                cmd,
                binds={
                    self.input_dir: "/input",
                    self.output_dir: "/output"
                },
                gpu=self.use_gpu
            )
            elapsed = time.time() - start_time
            
            if result is None or result.returncode != 0:
                logger.error(f"Inference failed for {input_file}")
                all_succeeded = False
            else:
                logger.info(f"Inference completed in {elapsed:.2f}s")
                
                # Check if output file was created
                expected_output = os.path.join(self.output_dir, output_name)
                if not os.path.exists(expected_output):
                    logger.error(f"Output file not created: {expected_output}")
                    all_succeeded = False
        
        return all_succeeded
    
    def validate_outputs(self):
        """Validate inference outputs."""
        logger.info("Validating outputs...")
        
        # Find all output files
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith('_pred.nii.gz')]
        
        if not output_files:
            logger.error("No output files found")
            return False
        
        # Try using nibabel for validation if available
        try:
            import nibabel as nib
            
            all_valid = True
            for output_file in output_files:
                output_path = os.path.join(self.output_dir, output_file)
                
                try:
                    # Load the NIfTI file
                    img = nib.load(output_path)
                    data = img.get_fdata()
                    
                    # Check for NaNs or infinities
                    if np.isnan(data).any() or np.isinf(data).any():
                        logger.error(f"Output {output_file} contains NaN or infinite values")
                        all_valid = False
                        continue
                    
                    # Output basic statistics
                    logger.info(f"Output {output_file}: shape={data.shape}, "
                                f"min={data.min():.4f}, max={data.max():.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to validate {output_file}: {str(e)}")
                    all_valid = False
            
            return all_valid
            
        except ImportError:
            # If nibabel is not available, just check if files exist
            logger.warning("nibabel not available, skipping detailed validation")
            logger.info(f"Found {len(output_files)} output files")
            return len(output_files) > 0
    
    def calculate_metrics(self):
        """Calculate metrics on the outputs (if applicable)."""
        # This is optional and would depend on having ground truth data
        # For now, we just report success
        logger.info("Metrics calculation not implemented yet")
        return True
    
    def validate(self):
        """Run the full validation process."""
        try:
            logger.info(f"Validating container: {self.container_path}")
            
            # Check if container exists and can run
            if not self.check_container():
                return False
            
            # Generate test data
            input_files = self.generate_test_data()
            if not input_files:
                return False
            
            # Run inference
            if not self.run_inference(input_files):
                return False
            
            # Validate outputs
            if not self.validate_outputs():
                return False
            
            # Calculate metrics
            if not self.calculate_metrics():
                return False
            
            logger.info("All validation checks PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            return False
            
        finally:
            if not args.keep_files:
                self.cleanup()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a container with a pre-trained model"
    )
    parser.add_argument(
        "container", 
        help="Path to the container file (.sif/.simg)"
    )
    parser.add_argument(
        "--no-gpu", 
        action="store_true",
        help="Disable GPU during validation"
    )
    parser.add_argument(
        "--keep-files", 
        action="store_true",
        help="Keep temporary files after validation"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for temporary files (default: auto-generated)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    workdir = args.output_dir or tempfile.mkdtemp()
    validator = ContainerValidator(
        args.container,
        workdir=workdir,
        gpu=not args.no_gpu
    )
    
    success = validator.validate()
    sys.exit(0 if success else 1)