#!/usr/bin/env python3
"""
Model Initialization Script

This script initializes a MONAI UNet model with random weights and saves it.
Designed to run both in development environment and during container build.
"""

import os
import torch
import sys

# Define paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory (or src) to the Python path
if os.path.exists(os.path.join(script_dir, "model.py")):
    # We're already in the src directory (container environment)
    sys.path.append(script_dir)
    from model import get_pretrained_model
else:
    # We're in the parent directory (development environment)
    src_dir = os.path.join(script_dir, "src")
    sys.path.append(src_dir)
    from model import get_pretrained_model

def main():
    """Initialize a model with random weights and save it."""
    # Create model directory if it doesn't exist
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Initializing model with random weights...")
    
    # Initialize model with random weights
    device = "cpu"  # Use CPU for initialization
    model = get_pretrained_model(pretrained_weights_path=None, device=device)
    
    # Define save path
    save_path = os.path.join(model_dir, "model_weights.pth")
    
    # Save the model weights
    torch.save(model.state_dict(), save_path)
    
    print(f"Model weights saved to: {save_path}")
    
    # Verify the file exists
    if os.path.exists(save_path):
        print(f"Successfully created model weights file ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
    else:
        print("Error: Failed to create model weights file")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)