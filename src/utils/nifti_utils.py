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
    reference_path: Optional[str] = None,
    threshold: Optional[float] = 0.5
) -> None:
    """
    Save a prediction as a NIfTI file.
    
    Args:
        prediction: Numpy array or PyTorch tensor of predictions
        output_path: Path to save the output NIfTI file
        reference_path: Optional path to a reference NIfTI file for header info
        threshold: Threshold for binarizing predictions (if None, saves as float32)
    """
    # Convert tensor to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Remove any singleton dimensions (important for metrics computation)
    prediction = np.squeeze(prediction)
    
    # Apply threshold if specified
    if threshold is not None:
        prediction = (prediction > threshold).astype(np.int8)
    
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
    input_dir: Optional[str] = None,
    threshold: Optional[float] = 0.5
) -> List[str]:
    """
    Save multiple predictions as NIfTI files.
    
    Args:
        predictions: Dictionary mapping filenames to prediction arrays
        output_dir: Directory to save the predictions
        input_dir: Optional input directory for reference files
        threshold: Threshold for binarizing predictions
        
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
        save_nifti(
            prediction, 
            output_path, 
            reference_path, 
            threshold=threshold
        )
        saved_paths.append(output_path)
        
    return saved_paths