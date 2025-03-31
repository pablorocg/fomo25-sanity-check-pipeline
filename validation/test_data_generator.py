import os
import logging
from typing import List, Optional
import numpy as np

class TestDataGenerator:
    """Generates synthetic test data for medical imaging models."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional logger.
        
        Args:
            logger: Logger instance to use. If None, a default logger is created.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_nifti_data(self, output_dir: str, num_samples: int = 5, size: int = 64) -> List[str]:
        """Generate synthetic NIfTI files for testing.
        
        Args:
            output_dir: Directory to save generated files
            num_samples: Number of test files to generate
            size: Size of cubic volume (size x size x size)
            
        Returns:
            List of paths to generated files
        """
        try:
            import nibabel as nib
        except ImportError:
            self.logger.error("nibabel not found - please install: pip install nibabel")
            return []
            
        file_paths = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test files
        affine = np.eye(4)
        for i in range(num_samples):
            data = np.random.rand(size, size, size).astype(np.float32)
            img = nib.Nifti1Image(data, affine)
            
            file_path = os.path.join(output_dir, f"test_sample_{i:03d}.nii.gz")
            nib.save(img, file_path)
            file_paths.append(file_path)
            
        self.logger.info(f"Generated {len(file_paths)} test files in {output_dir}")
        return file_paths
    
if __name__ == "__main__":
    
    test_data_gen = TestDataGenerator()
    output_dir = "test/input"

    test_data_gen.generate_nifti_data(output_dir, num_samples=10, size=64)
    
    
