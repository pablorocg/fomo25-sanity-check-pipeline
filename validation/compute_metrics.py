#!/usr/bin/env python3
"""
Simple segmentation metrics using MONAI.
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import logging
import torch
from monai.metrics import DiceMetric, ConfusionMatrixMetric

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_matching_files(pred_dir, gt_dir):
    """Find matching files between prediction and ground truth directories."""
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith((".nii", ".nii.gz"))]
    paired_files = []
    
    for pred_file in pred_files:
        base_name = pred_file.replace("_pred", "").replace("_output", "")
        matches = [f for f in os.listdir(gt_dir) 
                   if f.endswith((".nii", ".nii.gz")) and f.startswith(base_name.split("_")[0])]
        
        if matches:
            paired_files.append((os.path.join(pred_dir, pred_file), os.path.join(gt_dir, matches[0])))
    
    return paired_files

def calculate_metrics(pred_data, gt_data, threshold=None):
    """Calculate metrics using MONAI."""
    # Apply threshold if provided
    if threshold is not None:
        pred_binary = (pred_data > threshold).astype(np.float32)
    else:
        pred_binary = pred_data.astype(np.float32)
    
    # Convert to PyTorch tensors
    y_pred = torch.from_numpy(pred_binary).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    y_true = torch.from_numpy(gt_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Ensure same shape
    if y_pred.shape != y_true.shape:
        while len(y_pred.shape) > len(y_true.shape):
            y_pred = y_pred.squeeze(0)
        while len(y_true.shape) > len(y_pred.shape):
            y_true = y_true.squeeze(0)
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=True)
    confusion_matrix = ConfusionMatrixMetric(
        include_background=True,
        metric_name=["sensitivity", "specificity", "precision", "accuracy"]
    )
    
    # Calculate metrics
    dice_score = dice_metric(y_pred, y_true)
    confusion_scores = confusion_matrix(y_pred, y_true)
    
    # Compile results
    results = {
        "dice": dice_score.item(),
        "sensitivity": confusion_scores[0].item(),
        "specificity": confusion_scores[1].item(),
        "precision": confusion_scores[2].item(),
        "accuracy": confusion_scores[3].item(),
    }
    
    # Calculate Jaccard from Dice
    results["jaccard"] = results["dice"] / (2 - results["dice"]) if results["dice"] < 1.0 else 1.0
    
    return results

def main():
    """Main function."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <prediction_dir> <ground_truth_dir>")
        return 1
    
    pred_dir, gt_dir = sys.argv[1], sys.argv[2]
    
    # Check directories
    if not os.path.exists(pred_dir) or not os.path.exists(gt_dir):
        logger.error("Directory not found")
        return 1
    
    # Find paired files
    paired_files = find_matching_files(pred_dir, gt_dir)
    if not paired_files:
        logger.error("No matching files found")
        return 1
    
    logger.info(f"Found {len(paired_files)} file pairs")
    
    # Process each pair
    all_results = []
    for pred_path, gt_path in paired_files:
        pred_name = os.path.basename(pred_path)
        try:
            # Load data
            pred_data = nib.load(pred_path).get_fdata()
            gt_data = nib.load(gt_path).get_fdata()
            
            # Calculate metrics
            results = calculate_metrics(pred_data, gt_data)
            results["filename"] = pred_name
            all_results.append(results)
            
            logger.info(f"{pred_name}: Dice={results['dice']:.4f}")
        except Exception as e:
            logger.error(f"Error processing {pred_name}: {str(e)}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        results_dir = os.path.join(pred_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, "metrics_results.csv"), index=False)
        
        # Summary
        summary = df.select_dtypes(include=[np.number]).mean().to_dict()
        
        logger.info("=== Summary Metrics ===")
        for metric, value in summary.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        return 0
    
    logger.error("No results calculated")
    return 1

if __name__ == "__main__":
    sys.exit(main())