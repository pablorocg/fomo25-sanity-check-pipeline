"""
Lightning-based pipeline for medical image inference with rigorous dimensional controls.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np
import pytorch_lightning as pl
import nibabel as nib

# Import from project
from models.model import MedicalSegmentationModel
from data.data_module import MedicalImageDataModule
from utils.nifti_utils import (
    save_batch_nifti, load_nifti, get_nifti_metadata,
    standardize_orientation, resample_to_reference
)


class LightningInferencePipeline:
    """
    A Lightning-based pipeline for medical image inference with strict dimension handling.
    
    This class manages the entire inference workflow with rigorous controls for
    medical segmentation challenges.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 input_channels: int = 1,
                 output_channels: int = 1,
                 features: Tuple[int, ...] = (32, 32, 64, 128, 256, 32),
                 spatial_dims: int = 3,
                 spatial_size: Tuple[int, int, int] = (64, 64, 64),
                 intensity_range: Tuple[float, float] = (-57, 164),
                 threshold: float = 0.5,
                 output_dtype: np.dtype = np.uint8,
                 preprocess_orientation: bool = True) -> None:
        """
        Initialize the inference pipeline with strict parameters.
        
        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run inference on: "auto", "cuda", "cpu"
            input_channels: Number of input channels
            output_channels: Number of output channels (classes)
            features: Feature dimensions for the model layers
            spatial_dims: Number of spatial dimensions (2D or 3D)
            spatial_size: Spatial dimensions for image resizing
            intensity_range: Intensity range for normalization (a_min, a_max)
            threshold: Threshold for binary segmentation (0-1)
            output_dtype: Data type for output masks (uint8, uint16)
            preprocess_orientation: Whether to standardize orientation
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.features = features
        self.spatial_dims = spatial_dims
        self.spatial_size = spatial_size
        self.intensity_range = intensity_range
        self.threshold = threshold
        self.output_dtype = output_dtype
        self.preprocess_orientation = preprocess_orientation
        
        # Set device
        self.device = self._resolve_device(device)
        
        # Initialize model for the pipeline
        self.model = self._create_model()
        
        # Store input/output metadata
        self.input_metadata = {}
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device string to a valid PyTorch device.
        
        Args:
            device: Device string ("auto", "cuda", "cpu")
            
        Returns:
            Resolved device string
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
            return device
        else:
            self.logger.warning(f"Unknown device '{device}', falling back to auto")
            return self._resolve_device("auto")
    
    def _create_model(self) -> MedicalSegmentationModel:
        """
        Create the Lightning model.
        
        Returns:
            Initialized model
        """
        self.logger.info(f"Creating segmentation model with {self.input_channels} input channels "
                        f"and {self.output_channels} output channels")
        
        model = MedicalSegmentationModel(
            model_path=self.model_path,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            features=self.features
        )
        
        # Set model to evaluation mode
        model.eval()
        return model
    
    def _load_input_metadata(self, input_path: str) -> None:
        """
        Load and store metadata from input files for precise output generation.
        
        Args:
            input_path: Path to input data
        """
        self.input_metadata = {}
        
        if os.path.isfile(input_path) and input_path.endswith((".nii", ".nii.gz")):
            # Single file case
            filename = os.path.basename(input_path)
            self.input_metadata[filename] = get_nifti_metadata(input_path)
            self.logger.info(f"Loaded metadata for {filename}: shape={self.input_metadata[filename]['shape']}")
            
        elif os.path.isdir(input_path):
            # Directory case
            nifti_files = [
                os.path.join(input_path, f) 
                for f in os.listdir(input_path)
                if f.endswith((".nii", ".nii.gz"))
            ]
            
            for file_path in nifti_files:
                filename = os.path.basename(file_path)
                self.input_metadata[filename] = get_nifti_metadata(file_path)
                self.logger.info(f"Loaded metadata for {filename}: shape={self.input_metadata[filename]['shape']}")
    
    def _create_data_module(self, input_path: str) -> MedicalImageDataModule:
        """
        Create the Lightning data module with dimension validation.
        
        Args:
            input_path: Path to input data
            
        Returns:
            Initialized data module
        """
        self.logger.info(f"Creating data module for {input_path}")
        
        # Load input metadata before creating the data module
        self._load_input_metadata(input_path)
        
        # Create data module
        return MedicalImageDataModule(
            data_dir=input_path,
            batch_size=1,  # For inference, we use batch size 1
            spatial_size=self.spatial_size,
            intensity_range=self.intensity_range,
            mode="predict"
        )
    
    def _create_trainer(self) -> pl.Trainer:
        """
        Create the Lightning trainer.
        
        Returns:
            Initialized trainer
        """
        accelerator = "auto"
        devices = 1 
        
        self.logger.info(f"Creating trainer with accelerator={accelerator}, devices={devices}")
        
        return pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,  # Disable logging for prediction
            enable_progress_bar=True,
            enable_model_summary=False,
        )
    
    def show_model_summary(self) -> None:
        """Print the model summary."""
        self.logger.info("Model Summary:")
        input_size = (1, self.input_channels, *self.spatial_size)
        summary = self.model.show_summary(input_size=input_size)
        self.logger.info(f"\n{summary}")
    
    def predict(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Run inference on medical images with dimension validation.
        
        Args:
            input_path: Path to a NIfTI file or directory with NIfTI files
            
        Returns:
            Dictionary mapping filenames to prediction arrays
        """
        self.logger.info(f"Running inference on {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        # Create the data module
        data_module = self._create_data_module(input_path)
        data_module.setup(stage="predict")
        
        # Create a trainer for prediction
        trainer = self._create_trainer()
        
        # Run prediction
        self.logger.info("Starting model inference...")
        predictions = trainer.predict(self.model, datamodule=data_module)
        
        # Extract filenames
        filenames = data_module.get_filenames()
        
        # Create a mapping of filenames to predictions
        result = {}
        for i, batch_predictions in enumerate(predictions):
            if i < len(filenames):
                filename = filenames[i]
                
                # Post-process prediction
                processed_pred = self._post_process_prediction(batch_predictions, filename)
                if processed_pred is not None:
                    result[filename] = processed_pred
            else:
                self.logger.warning(f"More predictions than filenames at index {i}")
        
        self.logger.info(f"Processed {len(result)} predictions")
        return result
    
    def _post_process_prediction(self, prediction: torch.Tensor, filename: str) -> Optional[np.ndarray]:
        """
        Post-process a prediction tensor with precise dimensional control.
        
        Args:
            prediction: Raw prediction tensor from the model
            filename: Filename of the input image
            
        Returns:
            Processed numpy array or None if processing failed
        """
        try:
            # Convert to numpy and move to CPU
            pred = prediction.detach().cpu().numpy()
            
            # Apply sigmoid if using binary segmentation
            if self.output_channels == 1:
                pred = 1.0 / (1.0 + np.exp(-pred))
                
                # Apply threshold
                if self.threshold > 0:
                    pred = (pred > self.threshold).astype(np.float32)
            
            # Remove singleton dimensions
            pred = np.squeeze(pred)
            
            # Restore to original dimensions if we have metadata
            if filename in self.input_metadata:
                pred = self._restore_dimensions(pred, filename)
            
            # Convert to output data type
            pred = pred.astype(self.output_dtype)
            
            self.logger.info(f"Post-processed {filename}: shape={pred.shape}, dtype={pred.dtype}")
            return pred
            
        except Exception as e:
            self.logger.error(f"Failed to post-process {filename}: {e}")
            return None
    
    def _restore_dimensions(self, prediction: np.ndarray, filename: str) -> np.ndarray:
        """
        Restore prediction to original input dimensions for challenge submission.
        
        Args:
            prediction: Prediction array
            filename: Filename of the original image
            
        Returns:
            Resized prediction array
        """
        # Get original metadata
        original_metadata = self.input_metadata[filename]
        original_shape = original_metadata["shape"]
        
        # Create temporary input reference
        try:
            # Create reference array
            reference_data = np.zeros(original_shape)
            reference_affine = original_metadata["affine"]
            
            # Create prediction with identity affine
            pred_affine = np.eye(4)
            
            # Resample prediction to original space
            resampled_pred = resample_to_reference(
                prediction, pred_affine,
                reference_data, reference_affine,
                interpolation="nearest"  # Use nearest neighbor for segmentation
            )
            
            self.logger.info(f"Restored dimensions for {filename}: {prediction.shape} -> {resampled_pred.shape}")
            return resampled_pred
            
        except Exception as e:
            self.logger.error(f"Failed to restore dimensions for {filename}: {e}")
            return prediction
    
    def save_predictions(self, 
                         predictions: Dict[str, np.ndarray],
                         output_path: str,
                         input_dir: Optional[str] = None) -> List[str]:
        """
        Save predictions as NIfTI files with strict metadata handling.
        
        Args:
            predictions: Dictionary of filename to prediction array
            output_dir: Directory to save the predictions
            input_dir: Optional input directory for reference files
            
        Returns:
            List of saved file paths
        """
        self.logger.info(f"Saving {len(predictions)} predictions to {output_path}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save each prediction
        saved_paths = []
        for filename, prediction in predictions.items():
            # Construct output path
            # output_path = os.path.join(output_dir, filename)
            
            # Find reference file for metadata
            reference_path = None
            if input_dir:
                candidate_path = os.path.join(input_dir, filename)
                if os.path.exists(candidate_path):
                    reference_path = candidate_path
            
            try:
                # Create NIfTI with proper metadata
                if reference_path:
                    # Load reference image to get affine and header
                    reference_img = nib.load(reference_path)
                    
                    # Verify dimensions match
                    if prediction.shape != reference_img.shape:
                        self.logger.error(
                            f"Shape mismatch for {filename}: prediction {prediction.shape} vs "
                            f"reference {reference_img.shape}. Skipping."
                        )
                        continue
                    
                    # Create output image with reference metadata
                    output_img = nib.Nifti1Image(
                        prediction.astype(self.output_dtype),
                        reference_img.affine,
                        reference_img.header
                    )
                    output_img.set_data_dtype(self.output_dtype)
                    
                else:
                    # No reference, use identity affine
                    output_img = nib.Nifti1Image(
                        prediction.astype(self.output_dtype),
                        np.eye(4)
                    )
                    output_img.set_data_dtype(self.output_dtype)
                
                # Save the file
                nib.save(output_img, output_path)
                # saved_paths.append(output_path)
                self.logger.info(f"Saved {filename} to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save {filename}: {e}")
        
        return saved_paths