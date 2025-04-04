"""
Utility functions for working with NIfTI files.
"""
import os
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import nibabel as nib
import torch

logger = logging.getLogger(__name__)

def save_nifti(
    prediction: Union[np.ndarray, torch.Tensor],
    output_path: str,
    reference_path: Optional[str] = None
) -> None:
    """
    Save a prediction as a NIfTI file.
    
    Args:
        prediction: Numpy array or PyTorch tensor of predictions
        output_path: Path to save the output NIfTI file
        reference_path: Optional path to a reference NIfTI file for header info
    """
    # Convert tensor to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If a reference file is provided, use its header
    if reference_path and os.path.exists(reference_path):
        try:
            reference_img = nib.load(reference_path)
            output_img = nib.Nifti1Image(
                prediction, reference_img.affine, reference_img.header
            )
        except Exception as e:
            logger.warning(f"Error using reference file: {e}, creating new header")
            output_img = nib.Nifti1Image(prediction, np.eye(4))
    else:
        # Create a new NIfTI image with identity affine
        output_img = nib.Nifti1Image(prediction, np.eye(4))
    
    nib.save(output_img, output_path)
    logger.info(f"Saved prediction to {output_path}")

def save_batch_nifti(
    predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
    output_dir: str,
    input_dir: Optional[str] = None
) -> List[str]:
    """
    Save multiple predictions as NIfTI files.
    
    Args:
        predictions: Dictionary mapping filenames to prediction arrays
        output_dir: Directory to save the predictions
        input_dir: Optional input directory for reference files
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for filename, prediction in predictions.items():
        output_path = os.path.join(output_dir, filename)
        
        # Try to find reference file
        reference_path = None
        if input_dir:
            reference_path = os.path.join(input_dir, filename)
            if not os.path.exists(reference_path):
                reference_path = None
        
        # Save the prediction
        save_nifti(prediction, output_path, reference_path)
        saved_paths.append(output_path)
        
    return saved_paths

def load_nifti(file_path: str) -> np.ndarray:
    """
    Load a NIfTI file as a numpy array.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Numpy array with the NIfTI data
    """
    img = nib.load(file_path)
    return img.get_fdata()

def get_nifti_files(directory: str) -> List[str]:
    """
    Get all NIfTI files in a directory.
    
    Args:
        directory: Directory to search for NIfTI files
        
    Returns:
        List of NIfTI file paths
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")
        
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".nii", ".nii.gz"))
    ]

def get_nifti_metadata(file_path: str) -> Dict:
    """
    Get metadata from a NIfTI file.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Dictionary with metadata
    """
    img = nib.load(file_path)
    return {
        "shape": img.shape,
        "affine": img.affine,
        "header": {k: img.header[k] for k in img.header.keys()},
        "dimensions": img.header.get_data_shape(),
        "voxel_size": img.header.get_zooms(),
        "data_type": img.header.get_data_dtype()
    }