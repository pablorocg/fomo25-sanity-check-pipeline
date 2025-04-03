#!/usr/bin/env python
"""
Compatibility script for running either the original or 
Lightning-based inference pipeline.
"""

import argparse
import os
import sys
import logging

# Add the current directory to the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("main")

# Try to use the new Lightning pipeline if available
try:
    # First try the original pipeline (for backward compatibility)
    try:
        from pipeline import MedicalImageInferencePipeline
        
        def parse_args():
            parser = argparse.ArgumentParser(
                description="Medical Image Inference Pipeline",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
            
            parser.add_argument(
                "--input", "-i", type=str, required=True,
                help="Path to input NIfTI file or directory containing NIfTI files",
            )
            
            parser.add_argument(
                "--model", "-m", type=str, default=None,
                help="Path to pre-trained model weights",
            )
            
            parser.add_argument(
                "--output", "-o", type=str, required=True,
                help="Path to save output predictions",
            )
            
            parser.add_argument(
                "--device", "-d", type=str, default="cuda",
                choices=["cuda", "cpu"],
                help="Device to run inference on",
            )
            
            parser.add_argument(
                "--verbose", "-v", action="store_true", 
                help="Enable verbose logging"
            )
            
            return parser.parse_args()

        def main():
            # Parse arguments
            args = parse_args()
            
            # Setup logging
            level = logging.DEBUG if args.verbose else logging.INFO
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )
            
            try:
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
                os.makedirs(args.output, exist_ok=True)
                for filename, prediction in predictions.items():
                    input_path = os.path.join(args.input, filename) if os.path.isdir(args.input) else args.input
                    output_path = os.path.join(args.output, filename)
                    pipeline.save_prediction(prediction, output_path, input_path)
                
                logger.info(f"Saved predictions to {args.output}")
                logger.info("Inference completed successfully")
                return 0
                
            except Exception as e:
                logger.error(f"Error during inference: {e}", exc_info=True)
                return 1
    
    # If original pipeline not found, try the Lightning pipeline
    except ImportError:
        logger.info("Using Lightning-based inference pipeline")
        from inference.pipeline import LightningInferencePipeline
        
        def parse_args():
            parser = argparse.ArgumentParser(
                description="Lightning Medical Image Inference Pipeline",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
            
            parser.add_argument(
                "--input", "-i", type=str, required=True,
                help="Path to input NIfTI file or directory",
            )
            
            parser.add_argument(
                "--model", "-m", type=str, default=None,
                help="Path to pre-trained model weights",
            )
            
            parser.add_argument(
                "--output", "-o", type=str, required=True,
                help="Path to save output predictions",
            )
            
            parser.add_argument(
                "--device", "-d", type=str, default="auto",
                choices=["auto", "cuda", "cpu"],
                help="Device to run inference on",
            )
            
            parser.add_argument(
                "--verbose", "-v", action="store_true", 
                help="Enable verbose logging"
            )
            
            return parser.parse_args()

        def main():
            # Parse arguments
            args = parse_args()
            
            try:
                # Create the inference pipeline
                logger.info("Initializing inference pipeline...")
                pipeline = LightningInferencePipeline(
                    model_path=args.model, device=args.device
                )

                pipeline.show_model_summary()
                
                # Run inference
                logger.info(f"Running inference on {args.input}...")
                predictions = pipeline.predict(args.input)
                
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Save predictions
                saved_paths = pipeline.save_predictions(
                    predictions, 
                    args.output, 
                    args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
                )
                
                logger.info(f"Saved {len(saved_paths)} predictions to {args.output}")
                logger.info("Inference completed successfully")
                return 0
                
            except Exception as e:
                logger.error(f"Error during inference: {e}", exc_info=True)
                return 1

except ImportError as e:
    logger.error(f"Failed to import inference pipeline: {e}")
    logger.error("Make sure the code structure is correct and dependencies are installed.")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())