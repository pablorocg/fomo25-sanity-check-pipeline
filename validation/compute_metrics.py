#!/usr/bin/env python3
"""
Enhanced metrics computation using MONAI.
Calculates segmentation metrics between predictions and ground truth.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import logging
from typing import Dict, List, Tuple, Optional

# Import MONAI metrics
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_nifti(file_path: str) -> np.ndarray:
    """Load a NIfTI file and return as a numpy array."""
    try:
        return nib.load(file_path).get_fdata().astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def find_matching_files(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """Find matching files between prediction and ground truth directories."""
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith((".nii", ".nii.gz"))]

    paired_files = []
    for pred_file in pred_files:
        base_name = pred_file.replace("_pred", "").replace("_output", "")
        matches = [
            f
            for f in os.listdir(gt_dir)
            if f.endswith((".nii", ".nii.gz")) and f.startswith(base_name.split("_")[0])
        ]

        if matches:
            paired_files.append(
                (os.path.join(pred_dir, pred_file), os.path.join(gt_dir, matches[0]))
            )

    return paired_files


def calculate_metrics(pred_data: np.ndarray, gt_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics using MONAI.
    
    Args:
        pred_data: Prediction data as numpy array
        gt_data: Ground truth data as numpy array
        
    Returns:
        Dictionary of metrics
    """
    # Convert to PyTorch tensors with batch dimension
    pred_tensor = torch.from_numpy(pred_data > 0.5).unsqueeze(0).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(gt_data > 0.5).unsqueeze(0).unsqueeze(0).float()
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    surface_distance = SurfaceDistanceMetric(include_background=True, reduction="mean")
    
    # Confusion matrix metrics
    confusion_matrix = ConfusionMatrixMetric(
        include_background=True, 
        metric_name=["sensitivity", "specificity", "precision", "accuracy"],
        reduction="mean"
    )
    
    # Calculate metrics
    results = {}
    
    try:
        # Dice coefficient
        dice_result = dice_metric(pred_tensor, gt_tensor)
        results["dice"] = dice_result.item()
        
        # Hausdorff distance
        haussdorf_result = hausdorff_metric(pred_tensor, gt_tensor)
        results["hausdorff95"] = haussdorf_result.item()
        
        # Surface distance
        distance_result = surface_distance(pred_tensor, gt_tensor)
        results["surface_distance"] = distance_result.item()
        
        # Confusion matrix metrics
        confusion_results = confusion_matrix(pred_tensor, gt_tensor)
        for i, metric_name in enumerate(["sensitivity", "specificity", "precision", "accuracy"]):
            results[metric_name] = confusion_results[i].item()
        
        # Calculate Jaccard index (IoU) from Dice
        results["jaccard"] = results["dice"] / (2 - results["dice"]) if results["dice"] < 1.0 else 1.0
    
    except Exception as e:
        logger.warning(f"Error calculating some metrics: {e}")
        # Fallback to basic metrics using numpy if MONAI metrics fail
        pred_binary, gt_binary = pred_data > 0.5, gt_data > 0.5
        
        tp = np.logical_and(pred_binary, gt_binary).sum()
        tn = np.logical_and(~pred_binary, ~gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        fn = np.logical_and(~pred_binary, gt_binary).sum()
        
        # Calculate basic metrics
        epsilon = 1e-6
        results["dice"] = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        results["sensitivity"] = tp / (tp + fn + epsilon)
        results["specificity"] = tn / (tn + fp + epsilon)
        results["precision"] = tp / (tp + fp + epsilon)
        results["jaccard"] = tp / (tp + fp + fn + epsilon)
        results["accuracy"] = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    return results


def main() -> int:
    """Main function to calculate metrics."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <prediction_dir> <ground_truth_dir>")
        return 1

    pred_dir, gt_dir = sys.argv[1], sys.argv[2]

    # Check if directories exist
    for directory in [pred_dir, gt_dir]:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 1

    # Find matching files
    paired_files = find_matching_files(pred_dir, gt_dir)
    if not paired_files:
        logger.error("No matching files found")
        return 1

    logger.info(f"Found {len(paired_files)} file pairs")

    # Calculate metrics for each pair
    all_results = []
    for pred_path, gt_path in paired_files:
        pred_name = os.path.basename(pred_path)
        try:
            # Load data
            pred_data = load_nifti(pred_path)
            gt_data = load_nifti(gt_path)
            
            # Calculate metrics
            results = calculate_metrics(pred_data, gt_data)
            results["filename"] = pred_name
            all_results.append(results)
            
            logger.info(f"{pred_name}: Dice={results['dice']:.4f}")
        except Exception as e:
            logger.error(f"Error processing {pred_name}: {e}")

    # Save and display results
    if all_results:
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        results_dir = os.path.join(pred_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, "metrics_results.csv"), index=False)
        
        # Compute summary metrics
        numeric_df = df.select_dtypes(include=[np.number])
        summary = numeric_df.mean().to_dict()
        
        # Display summary
        logger.info("=== Summary Metrics ===")
        for metric, value in summary.items():
            if metric != "hausdorff95" and metric != "surface_distance":
                logger.info(f"{metric.capitalize()}: {value:.4f}")
            else:
                # Different format for distance metrics
                logger.info(f"{metric.capitalize()}: {value:.2f} voxels")
        
        return 0
    
    logger.error("No results calculated")
    return 1


if __name__ == "__main__":
    sys.exit(main())