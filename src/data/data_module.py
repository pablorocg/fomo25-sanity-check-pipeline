"""
LightningDataModule for medical image data.
"""
import os
import logging
from typing import Optional, List, Dict, Tuple, Union

import torch
import pytorch_lightning as pl
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)


class MedicalImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for medical images."""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 1,
                 train_val_ratio: float = 0.8,
                 num_workers: int = 4,
                 cache_rate: float = 1.0,
                 spatial_size: Tuple[int, int, int] = (64, 64, 64),
                 intensity_range: Tuple[float, float] = (-57, 164),
                 image_key: str = "image",
                 label_key: str = "label",
                 mode: str = "predict"):
        """
        Initialize the DataModule.
        
        Args:
            data_dir: Path to data directory
            batch_size: Batch size for DataLoader
            train_val_ratio: Ratio for train/validation split
            num_workers: Number of workers for DataLoader
            cache_rate: Cache rate for dataset (1.0 = full cache)
            spatial_size: Spatial dimensions for resizing
            intensity_range: Intensity range for normalization (a_min, a_max)
            image_key: Key for image in data dictionary
            label_key: Key for label in data dictionary
            mode: Operation mode - 'train', 'validate', 'test', or 'predict'
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.spatial_size = spatial_size
        self.intensity_range = intensity_range
        self.image_key = image_key
        self.label_key = label_key
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Will be set in setup()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
    
    def _create_train_transforms(self) -> Compose:
        """Create training transforms."""
        a_min, a_max = self.intensity_range
        keys = [self.image_key, self.label_key] if self.label_key else [self.image_key]
        
        return Compose(
            [
                LoadImaged(keys=keys, ensure_channel_first=True),
                EnsureTyped(keys=keys, dtype=torch.float32),
                Orientationd(keys=keys, axcodes="RAS"),
                Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Resized(keys=keys, spatial_size=self.spatial_size, mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=a_min,
                    a_max=a_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Add augmentations for training
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key=self.label_key,
                    spatial_size=self.spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                ) if self.label_key else Compose([]),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
                RandScaleIntensityd(keys=[self.image_key], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=[self.image_key], offsets=0.1, prob=0.5),
                ToTensord(keys=keys),
            ]
        )
    
    def _create_val_transforms(self) -> Compose:
        """Create validation transforms."""
        a_min, a_max = self.intensity_range
        keys = [self.image_key, self.label_key] if self.label_key else [self.image_key]
        
        return Compose(
            [
                LoadImaged(keys=keys, ensure_channel_first=True),
                EnsureTyped(keys=keys, dtype=torch.float32),
                Orientationd(keys=keys, axcodes="RAS"),
                Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Resized(keys=keys, spatial_size=self.spatial_size, mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=a_min,
                    a_max=a_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=keys),
            ]
        )
    
    def _create_predict_transforms(self) -> Compose:
        """Create prediction transforms."""
        a_min, a_max = self.intensity_range
        
        return Compose(
            [
                LoadImaged(keys=[self.image_key], ensure_channel_first=True),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
                Orientationd(keys=[self.image_key], axcodes="RAS"),
                Spacingd(keys=[self.image_key], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Resized(
                    keys=[self.image_key], 
                    spatial_size=self.spatial_size, 
                    mode="trilinear"
                ),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=a_min,
                    a_max=a_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=[self.image_key]),
            ]
        )
    
    def prepare_data(self) -> None:
        """
        Download data or verify data exists.
        This method is called only once and on 1 GPU only.
        """
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup data sources for each stage.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        if stage == "fit" or stage is None:
            # For training and validation
            if self.mode == "train":
                self._setup_train_val_data()
        
        if stage == "test" or stage is None:
            # For testing
            if self.mode == "test":
                self._setup_test_data()
        
        if stage == "predict" or stage is None:
            # For prediction
            if self.mode == "predict":
                self._setup_predict_data()
    
    def _setup_train_val_data(self) -> None:
        """Setup training and validation data."""
        # Example: Find all matching image-label pairs
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Expected directory for train/val data: {self.data_dir}")
            
        # Find all NIfTI files in the data directory
        images = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith((".nii", ".nii.gz")) and "label" not in f.lower()
        ]
        
        # Find matching labels if in training mode
        train_data = []
        if self.label_key:
            for img_path in images:
                dirname = os.path.dirname(img_path)
                basename = os.path.basename(img_path)
                label_candidates = [
                    os.path.join(dirname, f)
                    for f in os.listdir(dirname)
                    if f.endswith((".nii", ".nii.gz")) and "label" in f.lower() and
                    f.startswith(basename.split(".")[0])
                ]
                
                if label_candidates:
                    train_data.append({
                        self.image_key: img_path,
                        self.label_key: label_candidates[0]
                    })
        else:
            train_data = [{self.image_key: img_path} for img_path in images]
        
        # Train/val split
        n_train = int(len(train_data) * self.train_val_ratio)
        self.train_data = train_data[:n_train]
        self.val_data = train_data[n_train:]
    
    def _setup_test_data(self) -> None:
        """Setup test data."""
        if os.path.isdir(self.data_dir):
            # Find all NIfTI files in the data directory
            test_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith((".nii", ".nii.gz"))
            ]
            
            # Create test dataset
            self.test_data = [{self.image_key: f} for f in test_files]
        else:
            raise ValueError(f"Expected directory for test data: {self.data_dir}")
    
    def _setup_predict_data(self) -> None:
        """Setup prediction data."""
        if os.path.isfile(self.data_dir) and self.data_dir.endswith((".nii", ".nii.gz")):
            # Single file prediction
            self.predict_data = [{self.image_key: self.data_dir}]
        elif os.path.isdir(self.data_dir):
            # Directory prediction - get all NIfTI files
            self.predict_data = [
                {self.image_key: os.path.join(self.data_dir, f)}
                for f in os.listdir(self.data_dir)
                if f.endswith((".nii", ".nii.gz"))
            ]
        else:
            raise ValueError(f"Input path must be a NIfTI file or directory: {self.data_dir}")
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if not self.train_data:
            raise ValueError("Training data not set. Call setup('fit') first.")
            
        transforms = self._create_train_transforms()
        dataset = CacheDataset(
            data=self.train_data,
            transform=transforms,
            cache_rate=self.cache_rate
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation."""
        if not self.val_data:
            raise ValueError("Validation data not set. Call setup('fit') first.")
            
        transforms = self._create_val_transforms()
        dataset = CacheDataset(
            data=self.val_data,
            transform=transforms,
            cache_rate=self.cache_rate
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for testing."""
        if not self.test_data:
            raise ValueError("Test data not set. Call setup('test') first.")
            
        transforms = self._create_val_transforms()
        dataset = Dataset(data=self.test_data, transform=transforms)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create DataLoader for prediction."""
        if not self.predict_data:
            raise ValueError("Prediction data not set. Call setup('predict') first.")
            
        transforms = self._create_predict_transforms()
        dataset = Dataset(data=self.predict_data, transform=transforms)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def get_filenames(self) -> List[str]:
        """Get the list of filenames for prediction."""
        if hasattr(self, 'predict_data') and self.predict_data:
            return [os.path.basename(item[self.image_key]) for item in self.predict_data]
        return []