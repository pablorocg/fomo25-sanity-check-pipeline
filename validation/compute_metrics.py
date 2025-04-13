#!/usr/bin/env python3
"""
Enhanced segmentation metrics script that calculates metrics similarly to the
original Evaluator class.
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import logging
import json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define metric functions based on TP, FP, TN, FN values
def dice(tp, fp, tn, fn):
    """Calculate Dice coefficient."""
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

def jaccard(tp, fp, tn, fn):
    """Calculate Jaccard index (IoU)."""
    if tp + fp + fn == 0:
        return 1.0
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

def sensitivity(tp, fp, tn, fn):
    """Calculate sensitivity (recall)."""
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def precision_fn(tp, fp, tn, fn):
    """Calculate precision."""
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def volume_similarity(tp, fp, tn, fn):
    """Calculate volumetric similarity."""
    if tp + fn + fp == 0:
        return 1.0
    return 1 - abs(fn - fp) / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0.0

def TP(tp, fp, tn, fn):
    """Return true positives."""
    return tp

def FP(tp, fp, tn, fn):
    """Return false positives."""
    return fp

def FN(tp, fp, tn, fn):
    """Return false negatives."""
    return fn

def total_pos_gt(tp, fp, tn, fn):
    """Return total positives in ground truth."""
    return tp + fn

def total_pos_pred(tp, fp, tn, fn):
    """Return total positives in prediction."""
    return tp + fp

def find_matching_files(pred_dir, gt_dir):
    """Find matching files between prediction and ground truth directories."""
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith((".nii", ".nii.gz"))]
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith((".nii", ".nii.gz"))]
    
    paired_files = []
    
    for pred_file in pred_files:
        # Remove potential suffixes for matching
        base_name = pred_file.replace("_pred", "").replace("_output", "").split(".")[0]
        
        # Find matching ground truth file
        for gt_file in gt_files:
            gt_base = gt_file.split(".")[0]
            if base_name == gt_base or gt_base.startswith(base_name) or base_name.startswith(gt_base):
                paired_files.append((os.path.join(pred_dir, pred_file), os.path.join(gt_dir, gt_file)))
                break
    
    return paired_files

def get_surface_metrics_for_label(gt_img, pred_img, label, as_binary=False):
    """
    Calculate surface distance metrics.
    This is a simplified version that returns only Average Surface Distance.
    For a complete implementation, you'd need additional surface distance libraries.
    """
    # Placeholder for surface metrics - in a real implementation, you would
    # calculate actual surface distances here
    return {"Average Surface Distance": 0.0}  # Placeholder

def process_case(file_pair, labels, as_binary=False, do_surface_eval=False, ignore_labels=["0"]):
    """Process a single file pair and calculate metrics."""
    pred_path, gt_path = file_pair
    case_name = os.path.basename(pred_path)
    
    try:
        # Load the images
        pred = nib.load(pred_path)
        gt = nib.load(gt_path)
        
        pred_data = pred.get_fdata()
        gt_data = gt.get_fdata()
        
        # Calculate metrics
        if as_binary:
            cmat = confusion_matrix(
                np.around(gt_data.flatten()).astype(bool).astype(int),
                np.around(pred_data.flatten()).astype(bool).astype(int),
                labels=labels
            )
        else:
            cmat = confusion_matrix(
                np.around(gt_data.flatten()).astype(int),
                np.around(pred_data.flatten()).astype(int),
                labels=labels
            )
        
        metrics = {
            "Dice": dice,
            "Jaccard": jaccard,
            "Sensitivity": sensitivity,
            "Precision": precision_fn,
            "Volume Similarity": volume_similarity,
            "True Positives": TP,
            "False Positives": FP,
            "False Negatives": FN,
            "Total Positives Ground Truth": total_pos_gt,
            "Total Positives Prediction": total_pos_pred,
        }
        
        # Initialize results dict
        results = {}
        
        # Calculate metrics per label and aggregate
        tp_agg = 0
        fp_agg = 0
        fn_agg = 0
        tn_agg = 0
        
        for label in labels:
            label_results = {}
            
            # Extract values from confusion matrix
            tp = cmat[label, label]
            fp = sum(cmat[:, label]) - tp
            fn = sum(cmat[label, :]) - tp
            tn = np.sum(cmat) - tp - fp - fn
            
            # Skip aggregation for ignored labels
            label_str = str(label)
            if label_str not in ignore_labels:
                tp_agg += tp
                fp_agg += fp
                fn_agg += fn
                tn_agg += tn
            
            # Calculate all metrics for this label
            for metric_name, metric_function in metrics.items():
                label_results[metric_name] = round(metric_function(tp, fp, tn, fn), 4)
            
            # Add surface metrics if requested
            if do_surface_eval:
                surface_metrics = get_surface_metrics_for_label(gt, pred, label, as_binary)
                for surface_metric, val in surface_metrics.items():
                    label_results[surface_metric] = round(val, 4)
            
            results[str(label)] = label_results
        
        # Calculate aggregated metrics for all labels combined
        results["all"] = {
            metric_name: round(metric_function(tp_agg, fp_agg, tn_agg, fn_agg), 4)
            for metric_name, metric_function in metrics.items()
        }
        
        return (case_name, results)
    
    except Exception as e:
        logger.error(f"Error processing {case_name}: {str(e)}")
        return (case_name, {"error": str(e)})

def main():
    """Main function."""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <prediction_dir> <ground_truth_dir> [--binary] [--surface] [--labels=0,1,2,3]")
        return 1
    
    pred_dir, gt_dir = sys.argv[1], sys.argv[2]
    
    # Parse optional arguments
    as_binary = "--binary" in sys.argv
    do_surface_eval = "--surface" in sys.argv
    
    # Extract labels
    labels_arg = [arg for arg in sys.argv if arg.startswith("--labels=")]
    if labels_arg:
        labels = [int(l) for l in labels_arg[0].split("=")[1].split(",")]
    else:
        # Default: try to determine labels from files
        try:
            # Try loading the first GT file to determine labels
            first_gt = [f for f in os.listdir(gt_dir) if f.endswith((".nii", ".nii.gz"))][0]
            gt_data = nib.load(os.path.join(gt_dir, first_gt)).get_fdata()
            unique_labels = np.unique(np.around(gt_data).astype(int))
            labels = [int(l) for l in unique_labels]
            logger.info(f"Automatically detected labels: {labels}")
        except:
            # Fallback to binary labels
            labels = [0, 1]
            logger.info("Using default binary labels [0, 1]")
    
    # Set ignored labels
    ignore_labels = ["0"]  # By default ignore background
    
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
    logger.info(f"Evaluating performance on labels: {labels}")
    logger.info(f"Binary mode: {as_binary}")
    logger.info(f"Surface evaluation: {do_surface_eval}")
    
    # Process cases
    results_dict = {}
    all_case_results = {}
    
    # Use parallel processing if there are multiple files
    if len(paired_files) > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_case, 
                    file_pair, 
                    labels, 
                    as_binary, 
                    do_surface_eval, 
                    ignore_labels
                ) 
                for file_pair in paired_files
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
                case_name, case_results = future.result()
                all_case_results[case_name] = case_results
    else:
        # Process single file without parallelization
        for file_pair in paired_files:
            case_name, case_results = process_case(
                file_pair, 
                labels, 
                as_binary, 
                do_surface_eval, 
                ignore_labels
            )
            all_case_results[case_name] = case_results
    
    # Compute mean values across all cases
    mean_results = {}
    all_labels = [str(label) for label in labels] + ["all"]
    
    for label in all_labels:
        mean_results[label] = {}
        
        # Gather all metrics available
        all_metrics = set()
        for case_result in all_case_results.values():
            if label in case_result:
                all_metrics.update(case_result[label].keys())
        
        # Calculate mean for each metric
        for metric in all_metrics:
            values = [
                case_result[label][metric] 
                for case_result in all_case_results.values() 
                if label in case_result and metric in case_result[label]
            ]
            
            if values:
                mean_results[label][metric] = round(np.nanmean(values), 4)
            else:
                mean_results[label][metric] = 0
    
    # Final results dictionary
    results_dict = {"cases": all_case_results, "mean": mean_results}
    
    # Save results
    results_dir = os.path.join(pred_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON (like in the original implementation)
    json_path = os.path.join(results_dir, "metrics_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, default=float, indent=4)
    
    # Also save as CSV for convenience
    csv_path = os.path.join(results_dir, "metrics_results.csv")
    
    # Flatten the mean results for CSV
    csv_data = []
    for label in mean_results:
        row = {"label": label}
        row.update(mean_results[label])
        csv_data.append(row)
    
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    # Print summary of mean results
    logger.info("=== Summary Metrics ===")
    for label in all_labels:
        logger.info(f"Label {label}:")
        for metric, value in mean_results[label].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Results saved to {json_path} and {csv_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())