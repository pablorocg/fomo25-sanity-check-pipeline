"""
FOMO25 Medical Imaging Pipeline

A comprehensive framework for medical image segmentation
using PyTorch Lightning.
"""

__version__ = "0.1.0"

# Import main components to make them accessible directly
from . import data
from . import models
from . import utils
from . import inference

# You can also import specific classes/functions to expose at the top level
# This makes them directly importable from your package
from .inference.pipeline import LightningInferencePipeline
from .models.model import MedicalSegmentationModel
from .data.data_module import MedicalImageDataModule

__all__ = [
    "data",
    "models", 
    "utils",
    "inference",
    "LightningInferencePipeline",
    "MedicalSegmentationModel",
    "MedicalImageDataModule"
]