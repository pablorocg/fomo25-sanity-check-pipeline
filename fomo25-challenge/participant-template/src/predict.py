#!/usr/bin/env python3
"""
Model Prediction Script

This script is the primary entry point for running inference with a pretrained MONAI UNet model.
It follows a standard interface that will be used by the validation system.
"""

import os
import argparse
import logging
import time
import numpy as np
import nibabel as nib
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MONAI components
from monai.transforms import (
    LoadImage, 
    EnsureChannelFirst, 
    ScaleIntensity, 
    ToTensor,
    Compose
)

# Import your model function
from model import get_pretrained_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with the MONAI UNet model")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--model", required=True, help="Model weights path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    return parser.parse_args()

def load_image(input_path):
    """Load and preprocess the input image using MONAI transforms.
    
    Args:
        input_path: Path to the input NIfTI file
        
    Returns:
        Tuple of (preprocessed_data, original_image)
    """
    # Define MONAI transforms for preprocessing
    transforms = Compose([
        LoadImage(image_only=True),  # Load image
        EnsureChannelFirst(),        # Add channel dimension if missing
        ScaleIntensity(),            # Scale intensity to [0, 1]
        ToTensor()                   # Convert to PyTorch tensor
    ])
    
    # Load and transform image
    logger.info(f"Loading input file: {input_path}")
    data = transforms(input_path)
    
    # Also load original image for header/affine
    original_img = nib.load(input_path)
    
    return data, original_img

def run_inference(model, data, device):
    """Run inference with the model.
    
    Args:
        model: The loaded model
        data: Preprocessed input data
        device: Device to run inference on
        
    Returns:
        Model output as numpy array
    """
    logger.info("Running inference...")
    
    # Move data to device
    data = data.to(device)
    
    # Add batch dimension if not present
    if len(data.shape) == 3:
        data = data.unsqueeze(0)
    
    # Ensure we have a channel dimension
    if len(data.shape) == 4:
        data = data.unsqueeze(1)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        output = model(data)
        elapsed = time.time() - start_time
    
    # Apply sigmoid to get probability map
    output = torch.sigmoid(output)
    
    logger.info(f"Inference completed in {elapsed:.2f} seconds")
    
    # Convert to numpy and remove batch dimension
    return output.cpu().numpy().squeeze()

def save_output(output, original_img, output_path):
    """Save the output as a NIfTI file.
    
    Args:
        output: Model output as numpy array
        original_img: Original NIfTI image (for header/affine)
        output_path: Path to save the output file
    """
    logger.info(f"Saving output to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a new NIfTI image with the same header/affine as original
    output_img = nib.Nifti1Image(output, original_img.affine, original_img.header)
    
    # Save the output file
    nib.save(output_img, output_path)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set the device
    device = args.device
    logger.info(f"Using device: {device}")
    
    # Load and preprocess input
    data, original_img = load_image(args.input)
    
    # Load model with pretrained weights
    model = get_pretrained_model(args.model, device)
    
    # Run inference
    output = run_inference(model, data, device)
    
    # Save output
    save_output(output, original_img, args.output)
    
    logger.info("Prediction completed successfully")
    return True

if __name__ == "__main__":
    main()