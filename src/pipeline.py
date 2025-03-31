import logging
import os
from typing import Dict, Optional, Union

import nibabel as nib
import numpy as np
import torch
from torchinfo import summary
from monai.data import DataLoader, Dataset
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.nn import Module


class MedicalImageInferencePipeline:
    """
    A pipeline for medical image inference using MONAI and PyTorch.

    This class handles loading a model, preprocessing images, and running inference
    on NIfTI medical images.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda") -> None:
        """
        Initialize the inference pipeline.

        Args:
            model_path: Path to the pre-trained model weights. If None, uses an initialized model
                        without pre-trained weights.
            device: Device to run inference on, either "cuda" or "cpu".
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")

        self.model = self._load_model(model_path)
        self.transforms = self._create_transforms()

    def _load_model(self, model_path: Optional[str]) -> Module:
        """
        Load the pretrained model if model_path is provided.

        Args:
            model_path: Path to the model weights file

        Returns:
            The loaded model in evaluation mode

        Raises:
            FileNotFoundError: If the model path doesn't exist
            RuntimeError: If there's an error loading the model
        """
        model = (
            BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                features=(32, 32, 64, 128, 256, 32),
            )
            .to(self.device)
            .eval()
        )

        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Model loaded from {model_path}")
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"Failed to load model: {e}")
        else:
            self.logger.warning(
                "No model path provided. Using initialized model without pre-trained weights."
            )

        return model

    def _create_transforms(self) -> Compose:
        """
        Create the preprocessing transforms for the input images.

        Returns:
            A composition of transforms to apply to the input images
        """
        # LoadImaged, EnsureTyped, Orientationd, Spacingd, Resized, ScaleIntensityRanged
        return Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"], dtype=torch.float32),
                # Explicitly ensure 3D
                Orientationd(keys=["image"], axcodes="RAS"),
                # Resize to dimensions divisible by 32
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                # Resize to fixed dimensions (64x64x64)
                Resized(keys=["image"], spatial_size=(64, 64, 64), mode="trilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

    def predict(self, input_path: str) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on medical images.

        Args:
            input_path: Path to a single NIfTI file or a directory containing multiple files

        Returns:
            For a single file: numpy array of predictions
            For a directory: dictionary mapping filenames to prediction arrays

        Raises:
            FileNotFoundError: If the input path doesn't exist
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Single file prediction
        if os.path.isfile(input_path) and input_path.endswith((".nii", ".nii.gz")):
            return self._predict_single_file(input_path)

        # Directory prediction
        elif os.path.isdir(input_path):
            return self._predict_directory(input_path)

        else:
            raise ValueError(
                f"Input path must be a NIfTI file or directory: {input_path}"
            )

    def _predict_single_file(self, file_path: str) -> np.ndarray:
        """
        Run inference on a single NIfTI file.

        Args:
            file_path: Path to a NIfTI file

        Returns:
            Numpy array of predictions
        """
        self.logger.info(f"Running inference on file: {file_path}")

        # Create a dataset with a single sample
        data = [{"image": file_path}]
        dataset = Dataset(data=data, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Run inference
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["image"].to(self.device)
                outputs = self.model(inputs)
                # Convert outputs to numpy array
                prediction = outputs.cpu().numpy()

        return prediction

    def show_model_summary(self) -> None:
        """
        Print the model summary.
        """
        self.logger.info("Model Summary:")
        try:
            model_summary = summary(self.model, input_size=(1, 1, 64, 64, 64))
            self.logger.info(model_summary)
        except Exception as e:
            self.logger.error(f"Error generating model summary: {e}")


    def _predict_directory(self, dir_path: str) -> Dict[str, np.ndarray]:
        """
        Run inference on all NIfTI files in a directory.

        Args:
            dir_path: Path to a directory containing NIfTI files

        Returns:
            Dictionary mapping filenames to prediction arrays
        """
        self.logger.info(f"Running inference on directory: {dir_path}")

        # Get all NIfTI files in the directory
        nifti_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith((".nii", ".nii.gz"))
        ]

        if not nifti_files:
            self.logger.warning(f"No NIfTI files found in {dir_path}")
            return {}

        # Run prediction on each file
        results = {}
        for file_path in nifti_files:
            filename = os.path.basename(file_path)
            results[filename] = self._predict_single_file(file_path)

        return results

    def save_prediction(
        self,
        prediction: np.ndarray,
        output_path: str,
        reference_path: Optional[str] = None,
    ) -> None:
        """
        Save a prediction as a NIfTI file.

        Args:
            prediction: Numpy array of predictions
            output_path: Path to save the output NIfTI file
            reference_path: Optional path to a reference NIfTI file to copy header information
        """
        self.logger.info(f"Saving prediction to {output_path}")

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # If a reference file is provided, use its header
        if reference_path and os.path.exists(reference_path):
            reference_img = nib.load(reference_path)
            output_img = nib.Nifti1Image(
                prediction, reference_img.affine, reference_img.header
            )
        else:
            # Create a new NIfTI image with identity affine
            output_img = nib.Nifti1Image(prediction, np.eye(4))

        nib.save(output_img, output_path)
