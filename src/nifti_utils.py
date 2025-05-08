"""
Utility functions for working with NIfTI files with strict dimensional handling.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import nibabel as nib
import torch

logger = logging.getLogger(__name__)

def save_nifti(
    prediction: Union[np.ndarray, torch.Tensor],
    output_path: str,
    reference_path: Optional[str] = None,
    dtype: np.dtype = np.uint8
) -> None:
    """
    Save a prediction as a NIfTI file, preserving spatial metadata.
    
    Args:
        prediction: Numpy array or PyTorch tensor of predictions
        output_path: Path to save the output NIfTI file
        reference_path: Optional path to a reference NIfTI file for header info
        dtype: Data type to use for the output (uint8, uint16, etc.)
    """
    # Convert tensor to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle dimension and convert to specified dtype
    # For segmentation challenges, we standardize to 3D spatial volumes
    if len(prediction.shape) == 4 and prediction.shape[0] == 1:
        # Remove single channel dimension if present
        prediction = np.squeeze(prediction, axis=0)
    
    # Convert to specified data type
    prediction = prediction.astype(dtype)
    
    # If reference file provided, use its metadata
    if reference_path and os.path.exists(reference_path):
        reference_img = nib.load(reference_path)
        
        # Verify dimensions match reference
        if prediction.shape != reference_img.shape:
            logger.error(
                f"Shape mismatch: prediction {prediction.shape} vs reference {reference_img.shape}. "
                f"Cannot create valid NIfTI file."
            )
            raise ValueError(f"Output shape {prediction.shape} does not match reference {reference_img.shape}")
        
        # Create output with reference affine and header
        output_img = nib.Nifti1Image(prediction, reference_img.affine, reference_img.header)
        
        # Update header's data type
        output_img.set_data_dtype(dtype)
    else:
        # Create new NIfTI with identity affine when no reference is available
        output_img = nib.Nifti1Image(prediction, np.eye(4))
        output_img.set_data_dtype(dtype)
    
    # Save the file
    nib.save(output_img, output_path)
    logger.info(f"Saved NIfTI to {output_path} with shape {prediction.shape} and dtype {dtype}")

def save_batch_nifti(
    predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
    output_dir: str,
    input_dir: Optional[str] = None,
    dtype: np.dtype = np.uint8
) -> List[str]:
    """
    Save multiple predictions as NIfTI files with precise shape control.
    
    Args:
        predictions: Dictionary mapping filenames to prediction arrays
        output_dir: Directory to save the predictions
        input_dir: Optional input directory for reference files
        dtype: Data type to use for the output (uint8, uint16, etc.)
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for filename, prediction in predictions.items():
        output_path = os.path.join(output_dir, filename)
        
        # Find reference file if available
        reference_path = None
        if input_dir:
            candidate_path = os.path.join(input_dir, filename)
            if os.path.exists(candidate_path):
                reference_path = candidate_path
        
        # Save the prediction with strict shape validation
        try:
            save_nifti(prediction, output_path, reference_path, dtype)
            saved_paths.append(output_path)
        except ValueError as e:
            logger.error(f"Failed to save {filename}: {e}")
    
    return saved_paths

def load_nifti(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a NIfTI file as a numpy array with its metadata.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Tuple of (data_array, metadata_dict)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    
    # Load the image
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Extract metadata
    metadata = {
        "affine": img.affine,
        "header": img.header,
        "shape": img.shape,
        "voxel_size": img.header.get_zooms(),
        "data_type": img.header.get_data_dtype(),
        "original_path": file_path
    }
    
    return data, metadata

def resample_to_reference(
    source_data: np.ndarray, 
    source_affine: np.ndarray,
    reference_data: np.ndarray, 
    reference_affine: np.ndarray,
    interpolation: str = "nearest"
) -> np.ndarray:
    """
    Resample a volume to match the shape and affine of a reference volume.
    Standard approach for medical image processing challenges.
    
    Args:
        source_data: Source data array
        source_affine: Source affine matrix
        reference_data: Reference data array
        reference_affine: Reference affine matrix
        interpolation: Interpolation method ("nearest", "linear", etc.)
        
    Returns:
        Resampled data array
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    
    # Create NIfTI images
    source_nii = nib.Nifti1Image(source_data, source_affine)
    reference_nii = nib.Nifti1Image(reference_data, reference_affine)
    
    # Set interpolation order
    order = 0 if interpolation == "nearest" else 1
    
    # Resample source to reference space
    resampled_nii = resample_from_to(source_nii, reference_nii, order=order)
    
    # Return resampled data
    return resampled_nii.get_fdata()

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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    
    img = nib.load(file_path)
    return {
        "shape": img.shape,
        "affine": img.affine,
        "header": {k: img.header[k] for k in img.header.keys()},
        "dimensions": img.header.get_data_shape(),
        "voxel_size": img.header.get_zooms(),
        "data_type": img.header.get_data_dtype(),
        "orientation": nib.aff2axcodes(img.affine)
    }

def standardize_orientation(
    data: np.ndarray, 
    affine: np.ndarray, 
    target_orientation: str = "RAS"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize the orientation of a NIfTI volume to a target orientation.
    Critical for ensuring consistent processing in medical imaging challenges.
    
    Args:
        data: Input data array
        affine: Input affine matrix
        target_orientation: Target orientation code (e.g., "RAS")
        
    Returns:
        Tuple of (reoriented_data, reoriented_affine)
    """
    import nibabel as nib
    
    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)
    
    # Get current orientation
    current_orientation = "".join(nib.aff2axcodes(affine))
    
    # If already in target orientation, return as is
    if current_orientation == target_orientation:
        return data, affine
    
    # Reorient to target orientation
    reoriented_img = nib.as_closest_canonical(img)
    
    # Return reoriented data and affine
    return reoriented_img.get_fdata(), reoriented_img.affine