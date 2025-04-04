#!/usr/bin/env python3
"""
Simplified metrics computation for medical image segmentation evaluation.
Focuses on robust core metrics without complex spatial calculations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import only required MONAI metrics
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_segmentation(file_path: str) -> np.ndarray:
    """
    Load a segmentation NIfTI file.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Segmentation array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Log file information
        logger.info(f"Loaded: {os.path.basename(file_path)} - shape: {data.shape}")
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def find_matching_files(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching files between prediction and ground truth directories.
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
        
    Returns:
        List of tuples (pred_path, gt_path)
    """
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith((".nii", ".nii.gz"))]
    
    paired_files = []
    for pred_file in pred_files:
        # Check for exact match first
        if os.path.exists(os.path.join(gt_dir, pred_file)):
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, pred_file)
            paired_files.append((pred_path, gt_path))
            logger.info(f"Matched: {pred_file} -> {pred_file}")
            continue
        
        # Try base name matching
        base_name = pred_file.split('.')[0]
        base_name = base_name.replace("_pred", "").replace("_output", "")
        
        for gt_file in os.listdir(gt_dir):
            if gt_file.endswith((".nii", ".nii.gz")) and base_name in gt_file:
                pred_path = os.path.join(pred_dir, pred_file)
                gt_path = os.path.join(gt_dir, gt_file)
                paired_files.append((pred_path, gt_path))
                logger.info(f"Matched: {pred_file} -> {gt_file}")
                break
    
    return paired_files


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard metrics for segmentation evaluation.
    Uses both MONAI-based and fallback NumPy implementations.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        
    Returns:
        Dictionary of metric values
    """
    # Check shapes match
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    
    # Convert to binary masks (respecting user threshold decisions)
    # For float arrays, assume they're already thresholded
    # For integer arrays, use > 0 for binary mask
    if np.issubdtype(pred.dtype, np.floating):
        pred_bin = pred > 0
    else:
        pred_bin = pred > 0
    
    if np.issubdtype(gt.dtype, np.floating):
        gt_bin = gt > 0
    else:
        gt_bin = gt > 0
    
    # Calculate using NumPy (primary method for reliability)
    metrics = calculate_metrics_numpy(pred_bin, gt_bin)
    
    # Try MONAI calculation as a supplement if possible
    try:
        monai_metrics = calculate_metrics_monai(pred_bin, gt_bin)
        # Update with MONAI metrics, keep NumPy as fallback
        metrics.update(monai_metrics)
    except Exception as e:
        logger.warning(f"MONAI metrics calculation skipped: {e}")
    
    return metrics


def calculate_metrics_monai(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics using MONAI (reliable subset only).
    
    Args:
        pred: Binary prediction array
        gt: Binary ground truth array
        
    Returns:
        Dictionary of MONAI-based metrics
    """
    # Convert to torch tensors with batch and channel dimensions
    pred_tensor = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Dice metric
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_result = dice_metric(pred_tensor, gt_tensor)
    metrics["dice_monai"] = dice_result.mean().item()
    
    # Confusion matrix metrics
    confusion_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name=["sensitivity", "specificity", "precision", "accuracy"],
        reduction="mean"
    )
    
    confusion_results = confusion_metric(pred_tensor, gt_tensor)
    metric_names = ["sensitivity_monai", "specificity_monai", "precision_monai", "accuracy_monai"]
    
    for i, name in enumerate(metric_names):
        if i < len(confusion_results):
            metrics[name] = confusion_results[i].mean().item()
    
    return metrics


def calculate_metrics_numpy(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics using NumPy (core, reliable implementation).
    
    Args:
        pred: Binary prediction array
        gt: Binary ground truth array
        
    Returns:
        Dictionary of NumPy-based metrics
    """
    # Calculate overlap metrics
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-6
    
    # Calculate metrics
    metrics = {}
    
    # Dice coefficient
    metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    # Sensitivity (Recall)
    metrics["sensitivity"] = tp / (tp + fn + epsilon)
    
    # Specificity
    metrics["specificity"] = tn / (tn + fp + epsilon)
    
    # Precision
    metrics["precision"] = tp / (tp + fp + epsilon)
    
    # Accuracy
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    # Jaccard index (IoU)
    metrics["jaccard"] = tp / (tp + fp + fn + epsilon)
    
    return metrics


def evaluate_segmentation_pair(pred_path: str, gt_path: str) -> Dict[str, Any]:
    """
    Evaluate metrics for a prediction and ground truth pair.
    
    Args:
        pred_path: Path to prediction file
        gt_path: Path to ground truth file
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Load segmentations
        pred_data = load_segmentation(pred_path)
        gt_data = load_segmentation(gt_path)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_data, gt_data)
        
        # Add filename
        metrics["filename"] = os.path.basename(pred_path)
        
        # Log key metric
        logger.info(f"{metrics['filename']}: Dice={metrics['dice']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating {os.path.basename(pred_path)}: {e}")
        return {
            "filename": os.path.basename(pred_path),
            "error": str(e)
        }


def save_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save evaluation results to CSV and JSON files.
    
    Args:
        results: List of metric dictionaries
        output_dir: Directory to save results
    """
    # Skip if no results
    if not results:
        logger.error("No results to save")
        return
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    # Calculate aggregate metrics
    if valid_results:
        # Get numeric columns
        numeric_cols = []
        for col in valid_results[0].keys():
            if col != "filename" and isinstance(valid_results[0][col], (int, float)):
                numeric_cols.append(col)
        
        # Compute mean for each numeric column
        summary = {}
        for col in numeric_cols:
            values = [r[col] for r in valid_results if col in r]
            if values:
                summary[col] = np.mean(values)
    else:
        summary = {}
    
    # Create results directory
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results CSV
    csv_path = os.path.join(results_dir, "metrics_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info(f"Saved detailed metrics to {csv_path}")
    
    # Save summary CSV
    summary_path = os.path.join(results_dir, "metrics_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Saved summary metrics to {summary_path}")
    
    # Create simplified validation result
    validation_result = {
        "metrics": {
            "dice": summary.get("dice", 0.0),
            "jaccard": summary.get("jaccard", 0.0),
            "sensitivity": summary.get("sensitivity", 0.0),
            "specificity": summary.get("specificity", 0.0),
            "precision": summary.get("precision", 0.0),
            "accuracy": summary.get("accuracy", 0.0)
        }
    }
    
    # Save validation JSON
    json_path = os.path.join(output_dir, "validation_result.json")
    with open(json_path, "w") as f:
        json.dump(validation_result, f, indent=2)
    logger.info(f"Saved validation result to {json_path}")


def display_summary(results: List[Dict[str, Any]]) -> None:
    """
    Display summary metrics to console.
    
    Args:
        results: List of metric dictionaries
    """
    # Skip if no results
    if not results:
        logger.error("No results to display")
        return
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.error("No valid results to display")
        return
    
    # Calculate aggregate metrics
    numeric_cols = []
    for col in valid_results[0].keys():
        if col != "filename" and isinstance(valid_results[0][col], (int, float)):
            numeric_cols.append(col)
    
    # Compute mean for each numeric column
    summary = {}
    for col in numeric_cols:
        values = [r[col] for r in valid_results if col in r]
        if values:
            summary[col] = np.mean(values)
    
    # Display summary (only main metrics)
    logger.info("=== Summary Metrics ===")
    priority_metrics = ["dice", "jaccard", "sensitivity", "specificity", "precision", "accuracy"]
    
    for metric in priority_metrics:
        if metric in summary:
            logger.info(f"{metric.capitalize()}: {summary[metric]:.4f}")


def main() -> int:
    """
    Main function to calculate metrics.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <prediction_dir> <ground_truth_dir>")
        return 1

    pred_dir, gt_dir = sys.argv[1], sys.argv[2]

    try:
        # Check directories exist
        if not os.path.isdir(pred_dir):
            logger.error(f"Prediction directory not found: {pred_dir}")
            return 1
            
        if not os.path.isdir(gt_dir):
            logger.error(f"Ground truth directory not found: {gt_dir}")
            return 1
        
        # Find matching files
        paired_files = find_matching_files(pred_dir, gt_dir)
        if not paired_files:
            logger.error("No matching files found")
            return 1

        logger.info(f"Found {len(paired_files)} file pairs")

        # Evaluate each pair
        all_results = []
        for pred_path, gt_path in paired_files:
            result = evaluate_segmentation_pair(pred_path, gt_path)
            all_results.append(result)

        # Display summary
        display_summary(all_results)

        # Save results
        save_results(all_results, pred_dir)

        # Check for errors
        error_count = sum(1 for r in all_results if "error" in r)
        if error_count > 0:
            logger.warning(f"{error_count} of {len(all_results)} file pairs had errors")
            if error_count == len(all_results):
                return 1

        return 0

    except Exception as e:
        logger.error(f"Error in metric computation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())