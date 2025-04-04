"""
Lightning-based pipeline for medical image inference.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar

# Fix imports to use absolute imports with app as root
from models.model import MedicalSegmentationModel
from data.data_module import MedicalImageDataModule
from utils.nifti_utils import save_batch_nifti


class LightningInferencePipeline:
    """
    A Lightning-based pipeline for medical image inference.
    
    This class handles loading a model, preprocessing images, and running inference
    on medical images using PyTorch Lightning.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 input_channels: int = 1,
                 output_channels: int = 1,
                 features: Tuple[int, ...] = (32, 32, 64, 128, 256, 32),
                 spatial_size: Tuple[int, int, int] = (64, 64, 64),
                 intensity_range: Tuple[float, float] = (-57, 164),
                 threshold: float = 0.5) -> None:
        """
        Initialize the Lightning inference pipeline.
        
        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run inference on: "auto", "cuda", "cpu"
            input_channels: Number of input channels
            output_channels: Number of output channels
            features: Feature dimensions for the UNet layers
            spatial_size: Spatial dimensions for image resizing
            intensity_range: Intensity range for normalization (a_min, a_max)
            threshold: Threshold for binary segmentation (0-1)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.features = features
        self.spatial_size = spatial_size
        self.intensity_range = intensity_range
        self.threshold = threshold
        
        # Set device
        self.device = self._resolve_device(device)
        
        # Initialize model for the pipeline
        self.model = self._create_model()
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device string to a valid PyTorch Lightning accelerator.
        
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
        self.logger.info(f"Creating model with {self.input_channels} input channels "
                         f"and {self.output_channels} output channels")
        
        return MedicalSegmentationModel(
            model_path=self.model_path,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            features=self.features
        )
    
    def _create_data_module(self, input_path: str) -> MedicalImageDataModule:
        """
        Create the Lightning data module.
        
        Args:
            input_path: Path to input data
            
        Returns:
            Initialized data module
        """
        self.logger.info(f"Creating data module for {input_path}")
        
        return MedicalImageDataModule(
            data_dir=input_path,
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
        accelerator = self.device if self.device == "cuda" else None
        devices = 1 if self.device == "cuda" else None
        
        self.logger.info(f"Creating trainer with accelerator={accelerator}, devices={devices}")
        
        return pl.Trainer(
            accelerator=accelerator,
            devices=devices,
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
        Run inference on medical images.
        
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
        predictions = trainer.predict(self.model, datamodule=data_module)
        
        # Extract filenames
        filenames = data_module.get_filenames()
        
        # Create a mapping of filenames to predictions
        result = {}
        for i, batch_predictions in enumerate(predictions):
            if i < len(filenames):
                filename = filenames[i]
                # Process prediction (sigmoid + threshold if needed)
                pred = self._process_prediction(batch_predictions)
                result[filename] = pred
            else:
                self.logger.warning(f"More predictions than filenames at index {i}")
        
        return result
    
    def _process_prediction(self, prediction: torch.Tensor) -> np.ndarray:
        """
        Process a prediction tensor.
        
        Args:
            prediction: Raw prediction tensor from the model
            
        Returns:
            Processed numpy array
        """
        # Convert to numpy
        pred = prediction.detach().cpu().numpy()
        
        # Apply sigmoid for probability [0,1]
        if self.output_channels == 1:
            # Sigmoid not needed if already applied in the model
            pred = 1.0 / (1.0 + np.exp(-pred))
            
            # Apply threshold for binary segmentation
            if self.threshold > 0:
                pred = (pred > self.threshold).astype(np.float32)
        
        # Squeeze batch dimension if present
        if pred.shape[0] == 1:
            pred = np.squeeze(pred, axis=0)
        
        return pred
    
    def save_predictions(self, 
                         predictions: Dict[str, np.ndarray],
                         output_dir: str,
                         input_dir: Optional[str] = None) -> List[str]:
        """
        Save predictions as NIfTI files.
        
        Args:
            predictions: Dictionary of filename to prediction array
            output_dir: Directory to save the predictions
            input_dir: Optional input directory for reference files
            
        Returns:
            List of saved file paths
        """
        self.logger.info(f"Saving {len(predictions)} predictions to {output_dir}")
        
        return save_batch_nifti(predictions, output_dir, input_dir)
