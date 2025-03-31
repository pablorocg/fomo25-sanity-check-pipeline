#!/usr/bin/env python3
"""
Compute segmentation metrics between predictions and ground truth.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def dice_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Dice coefficient between two binary arrays."""
    epsilon = 1e-6
    y_pred, y_true = y_pred.astype(bool), y_true.astype(bool)
    intersection = np.logical_and(y_pred, y_true).sum()
    return (2.0 * intersection + epsilon) / (y_pred.sum() + y_true.sum() + epsilon)


def calculate_metrics(pred_data: np.ndarray, gt_data: np.ndarray) -> dict:
    """Calculate comprehensive segmentation metrics."""
    pred_binary, gt_binary = pred_data > 0.5, gt_data > 0.5

    def safe_metric(func, *args):
        try:
            return func(*args)
        except Exception as e:
            logger.warning(f"{func.__name__} calculation failed: {e}")
            return float("nan")

    return {
        "dice": dice_coefficient(pred_binary, gt_binary),
        **confusion_matrix_metrics(pred_binary, gt_binary),
    }


def confusion_matrix_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Calculate confusion matrix-based metrics."""
    tp = np.logical_and(y_pred, y_true).sum()
    tn = np.logical_and(~y_pred, ~y_true).sum()
    fp = np.logical_and(y_pred, ~y_true).sum()
    fn = np.logical_and(~y_pred, y_true).sum()

    def safe_div(num, denom):
        return num / denom if denom > 0 else 0

    return {
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "jaccard": safe_div(tp, tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
    }


def find_matching_files(pred_dir: str, gt_dir: str) -> list:
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


def main():
    """Main function to calculate metrics."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <prediction_dir> <ground_truth_dir>")
        return 1

    pred_dir, gt_dir = sys.argv[1], sys.argv[2]

    for directory in [pred_dir, gt_dir]:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 1

    paired_files = find_matching_files(pred_dir, gt_dir)

    if not paired_files:
        logger.error("No matching files found")
        return 1

    logger.info(f"Found {len(paired_files)} file pairs")

    all_results = []
    for pred_path, gt_path in paired_files:
        pred_name = os.path.basename(pred_path)
        try:
            pred_data = nib.load(pred_path).get_fdata().astype(np.float32)
            gt_data = nib.load(gt_path).get_fdata().astype(np.float32)

            results = calculate_metrics(pred_data, gt_data)
            results["filename"] = pred_name
            all_results.append(results)

            logger.info(f"{pred_name}: Dice={results['dice']:.4f}")
        except Exception as e:
            logger.error(f"Error processing {pred_name}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        # Save results in test/results
        os.makedirs(os.path.join(pred_dir, "results"), exist_ok=True)
        df.to_csv(os.path.join(pred_dir, "results", "metrics_results.csv"), index=False)
        

        numeric_df = df.select_dtypes(include=[np.number])
        summary = numeric_df.mean().to_dict()

        logger.info("=== Summary Metrics ===")
        for metric, value in summary.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")

        return 0

    logger.error("No results calculated")
    return 1


if __name__ == "__main__":
    sys.exit(main())
