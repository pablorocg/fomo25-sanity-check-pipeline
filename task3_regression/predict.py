#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 3: Brain Age Prediction (Regression)
"""
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 3 Brain Age Prediction")
    
    # Input paths for T1 and T2 modalities
    parser.add_argument("--t1", type=str, help="Path to T1-weighted image")
    parser.add_argument("--t2", type=str, help="Path to T2-weighted image")
    
    # Output path for predictions
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV")
    
    return parser.parse_args()

def predict_age(args):
    """
    Predict brain age based on T1 and T2 modalities.
    
    Returns:
        float: Predicted brain age in years
    """
    
    #########################################################################
    # PLACEHOLDER: ADD YOUR BRAIN AGE PREDICTION CODE HERE
    #########################################################################
    # 
    # Available image paths:
    #   - args.t1: T1-weighted image path
    #   - args.t2: T2-weighted image path
    #
    # Example steps you might implement:
    #   1. Load T1 and T2 images
    #   2. Preprocess images (normalize, skull-strip, register, etc.)
    #   3. Extract features or prepare input for your model
    #   4. Load your trained regression model
    #   5. Run inference to predict age
    #   6. Return predicted age value
    #
    # Example (replace with your actual code):
    #   model = load_your_age_prediction_model()
    #   t1_image = load_and_preprocess_image(args.t1)
    #   t2_image = load_and_preprocess_image(args.t2)
    #   features = extract_features(t1_image, t2_image)
    #   predicted_age = model.predict(features)
    #
    #########################################################################
    
    # Dummy age prediction - REPLACE THIS WITH YOUR ACTUAL PREDICTION
    predicted_age = 45.0
    
    return predicted_age

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Get age prediction
    predicted_age = predict_age(args)
    
    # Create output TXT file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"{predicted_age:.2f}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())