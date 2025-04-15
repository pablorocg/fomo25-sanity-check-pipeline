#!/usr/bin/env python
"""
Prediction script for medical image segmentation.
Simple version: 1 file input, 1 file output.
"""

import argparse
import os
import sys
import logging
import numpy as np
import nibabel as nib

# Add the current directory to the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("predict")

# Import inference pipeline
from inference.pipeline import LightningInferencePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical Image Segmentation Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input NIfTI file",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to pre-trained model weights",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save output prediction file",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Threshold for binary segmentation (0-1)",
    )

    parser.add_argument(
        "--channels",
        "-c",
        type=int,
        default=1,
        help="Number of output channels (classes)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16"],
        help="Output data type for segmentation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Validate input file
    if not os.path.isfile(args.input):
        logger.error(f"Input must be a file: {args.input}")
        return False

    # Validate model path if provided
    if args.model and not os.path.exists(args.model):
        logger.error(f"Model path does not exist: {args.model}")
        return False

    # Validate threshold
    if args.threshold < 0 or args.threshold > 1:
        logger.error(f"Threshold must be between 0 and 1: {args.threshold}")
        return False

    # Ensure output directory parent exists
    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent and not os.path.exists(output_parent):
        os.makedirs(output_parent, exist_ok=True)
        logger.info(f"Created output directory: {output_parent}")

    return True


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not validate_args(args):
        return 1

    try:
        # Get output data type
        output_dtype = getattr(np, args.dtype)

        # Create the inference pipeline
        logger.info("Initializing inference pipeline...")
        pipeline = LightningInferencePipeline(
            model_path=args.model,
            device=args.device,
            output_channels=args.channels,
            threshold=args.threshold,
            output_dtype=output_dtype,
        )

        # Display model summary
        if args.verbose:
            pipeline.show_model_summary()

        # Run inference
        logger.info(f"Running inference on {args.input}...")
        predictions = pipeline.predict(args.input)

        if not predictions or len(predictions) == 0:
            logger.error("No prediction generated")
            return 1

        # Get the prediction (there should be only one)
        prediction = next(iter(predictions.values()))
        
        # Get reference metadata from input file
        reference_img = nib.load(args.input)
        
        # Create output image with reference metadata
        output_img = nib.Nifti1Image(
            prediction.astype(output_dtype),
            reference_img.affine,
            reference_img.header
        )
        output_img.set_data_dtype(output_dtype)
        
        # Save the file
        nib.save(output_img, args.output)
        logger.info(f"Saved prediction to {args.output}")
        
        logger.info("Inference completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
