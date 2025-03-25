"""
MONAI UNet model definition.

This file implements loading of a pretrained UNet model from MONAI.
"""

import os
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_pretrained_model(pretrained_weights_path=None, device="cpu"):
    """
    Create a MONAI UNet model with optional pretrained weights.
    
    Args:
        pretrained_weights_path: Path to pretrained weights file (optional)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model
    """
    # Create UNet model using MONAI's implementation
    model = UNet(
        spatial_dims=3,           # 3D medical images
        in_channels=1,            # Single channel input (grayscale)
        out_channels=1,           # Single channel output (segmentation mask)
        channels=(16, 32, 64, 128, 256),  # Channel sequence
        strides=(2, 2, 2, 2),     # Stride sequence
        num_res_units=2,          # Number of residual units per layer
        norm=Norm.BATCH          # Batch normalization
    )
    
    # Load pretrained weights if provided
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        try:
            # Attempt to load the weights
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
            print(f"Successfully loaded pretrained weights from {pretrained_weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            print("Using model with random initialization")
    else:
        print("No pretrained weights provided or file not found. Using model with random initialization.")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model