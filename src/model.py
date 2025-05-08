"""
LightningModule for medical image segmentation model.
"""
import os
import logging
from typing import Optional, Dict, Any, List

import torch
import pytorch_lightning as pl
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torchinfo import summary

class MedicalSegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for medical image segmentation."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 features: tuple = (32, 32, 64, 128, 256, 32),
                 spatial_dims: int = 3,
                 learning_rate: float = 1e-3):
        """
        Initialize the Lightning Module.
        
        Args:
            model_path: Path to pre-trained weights if available
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            features: Feature dimensions for the UNet layers
            spatial_dims: Number of spatial dimensions (2D or 3D)
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.save_hyperparameters()
        # Use _log instead of logger to avoid conflict with Lightning's logger property
        self._log = logging.getLogger(self.__class__.__name__)
        
        # Create the model - can be easily extended for other model types
        self.model = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )
        
        # Define loss function and metrics
        self.loss_function = DiceLoss(sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        elif model_path:
            self._log.warning(f"Model path {model_path} does not exist")
    
    def _load_weights(self, model_path: str) -> None:
        """Load model weights from file."""
        self._log.info(f"Loading model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self._log.info("Model loaded successfully")
        except Exception as e:
            self._log.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for Lightning."""
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step for Lightning."""
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        
        # Calculate Dice score
        self.dice_metric(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self) -> None:
        """End of validation epoch."""
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", dice_score, prog_bar=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Test step for Lightning."""
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.dice_metric(outputs, labels)
        self.log("test_loss", loss)
        return {"test_loss": loss}
    
    def on_test_epoch_end(self) -> None:
        """End of test epoch."""
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("test_dice", dice_score)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run prediction on a batch."""
        outputs = self.forward(batch["image"])
        return outputs
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def show_summary(self, input_size: tuple = (1, 1, 64, 64, 64)) -> str:
        """Generate and return model summary."""
        try:
            model_summary = summary(self.model, input_size=input_size, verbose=0)
            return str(model_summary)
        except Exception as e:
            self._log.error(f"Error generating model summary: {e}")
            return f"Error generating model summary: {e}"