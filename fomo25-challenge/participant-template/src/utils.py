"""
Utility functions for data loading, processing, and visualization with MONAI.
"""

import os

import nibabel as nib
import numpy as np
import torch
from monai.data import NiftiSaver
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    ScaleIntensity,
    ToTensor,
)
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    """
    Dataset class for loading NIfTI files using MONAI transforms.
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing the NIfTI files
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".nii.gz")]

        # Default transforms
        self.transform = transform
        if self.transform is None:
            self.transform = Compose(
                [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    ScaleIntensity(),
                    ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = self.transform(file_path)
        return data, file_path


def calculate_dice(y_pred, y_true, threshold=0.5):
    """
    Calculate Dice coefficient using MONAI's DiceMetric.

    Args:
        y_pred: Predicted segmentation (as tensor or numpy array)
        y_true: Ground truth segmentation (as tensor or numpy array)
        threshold: Threshold for binarizing predictions

    Returns:
        Dice coefficient
    """
    # Convert to tensors if they are numpy arrays
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure 4D tensors [B, C, H, W] or 5D [B, C, D, H, W]
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)
    elif y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(0)

    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
    elif y_true.dim() == 3:
        y_true = y_true.unsqueeze(0)

    # Binarize predictions
    y_pred_bin = (y_pred > threshold).float()

    # Initialize DiceMetric
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Compute Dice
    dice_metric(y_pred_bin, y_true)
    result = dice_metric.aggregate().item()

    return result


def save_prediction(data, original_img_path, output_path):
    """
    Save prediction as a NIfTI file using MONAI's NiftiSaver.

    Args:
        data: Prediction data (tensor or numpy array)
        original_img_path: Path to the original input image
        output_path: Path to save the output
    """
    # Initialize NiftiSaver
    saver = NiftiSaver(output_dir=os.path.dirname(output_path), output_postfix="")

    # Convert to tensor if numpy array
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    # Ensure correct dimensions
    if data.dim() == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 3:
        data = data.unsqueeze(0)

    # Save the file with the original metadata
    saver.save(data, {"filename_or_obj": original_img_path})

    # Rename file if needed to match the requested output path
    saved_path = os.path.join(
        os.path.dirname(output_path),
        os.path.basename(original_img_path).split(".")[0]
        + saver.output_postfix
        + ".nii.gz",
    )

    if saved_path != output_path:
        os.rename(saved_path, output_path)
