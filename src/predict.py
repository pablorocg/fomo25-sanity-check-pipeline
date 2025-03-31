#!/usr/bin/env python
"""
Command-line interface for running the Medical Image Inference Pipeline.
"""

import argparse
import logging
import os
import sys

from pipeline import MedicalImageInferencePipeline


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: Whether to enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical Image Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input NIfTI file or directory containing NIfTI files",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to pre-trained model weights",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save output predictions (directory will be created if it doesn't exist)",
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main() -> int:
    """Main function for the CLI application."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    
    try:
        # Check if input exists
        if not os.path.exists(args.input):
            logger.error(f"Input path not found: {args.input}")
            return 1
        
        # Create the inference pipeline
        logger.info("Initializing inference pipeline...")
        pipeline = MedicalImageInferencePipeline(
            model_path=args.model, device=args.device
        )

        pipeline.show_model_summary()
        
        # Run inference
        logger.info(f"Running inference on {args.input}...")
        predictions = pipeline.predict(args.input)
        
        # Save predictions
        if os.path.isfile(args.input) and args.input.endswith((".nii", ".nii.gz")):
            # Single file prediction
            output_filename = os.path.basename(args.input)
            output_path = os.path.join(args.output, output_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
            
            # Use input path as reference
            pipeline.save_prediction(predictions, output_path, args.input)
            logger.info(f"Saved prediction to {output_path}")
            
        elif os.path.isdir(args.input):
            # Directory prediction - create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Save each prediction
            for filename, prediction in predictions.items():
                output_path = os.path.join(args.output, filename)
                input_file_path = os.path.join(args.input, filename)
                
                pipeline.save_prediction(prediction, output_path, input_file_path)
                
            logger.info(f"Saved {len(predictions)} predictions to {args.output}")
            
        logger.info("Inference completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
