import argparse
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import nibabel as nib
import numpy as np


class OutputType(Enum):
    """Types of outputs supported by the validator."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"


@dataclass
class TaskConfig:
    """Configuration for a validation task."""
    sequences: List[str]
    min_sequences: int
    output_type: OutputType


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ContainerValidator:
    """Validates container environments and runs predictions for medical imaging tasks."""
    
    # Constants
    REQUIRED_CONTAINER_PATHS = ["/app", "/input", "/output", "/app/predict.py"]
    SESSION_DIR = "ses_1"
    SUBJECT_PREFIX = "sub_"
    
    # Task configurations
    TASK_CONFIGS = {
        "task1": TaskConfig(
            sequences=["adc", "dwi_b1000", "flair", "t2s", "swi"],
            min_sequences=4,
            output_type=OutputType.CLASSIFICATION
        ),
        "task2": TaskConfig(
            sequences=["dwi_b1000", "flair", "t2s", "swi"],
            min_sequences=3,
            output_type=OutputType.SEGMENTATION
        ),
        "task3": TaskConfig(
            sequences=["t1", "t2"],
            min_sequences=2,
            output_type=OutputType.REGRESSION
        ),
    }

    def __init__(self, container_path: Path, task: str, apptainer_cmd: str = "apptainer"):
        """Initialize the container validator.
        
        Args:
            container_path: Path to the container file
            task: Task identifier (task1, task2, or task3)
            apptainer_cmd: Command to run apptainer
            
        Raises:
            ValidationError: If container doesn't exist or task is unsupported
        """
        self.container_path = Path(container_path).resolve()
        if not self.container_path.exists():
            raise ValidationError(f"Container {self.container_path} does not exist")
        
        if task not in self.TASK_CONFIGS:
            raise ValidationError(f"Task {task} not supported")
        
        self.task = task
        self.config = self.TASK_CONFIGS[task]
        self.apptainer_cmd = apptainer_cmd

    def _run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run a subprocess command and return success status and output.
        
        Args:
            cmd: Command to execute as list of strings
            check: Whether to raise exception on non-zero exit code
            
        Returns:
            Tuple of (success, output_or_error_message)
        """
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            return False, error_msg
        except FileNotFoundError as e:
            return False, str(e)

    def _check_apptainer_installation(self) -> Tuple[bool, str]:
        """Check if apptainer is properly installed."""
        success, msg = self._run_command([self.apptainer_cmd, "--version"])
        if success:
            print(f"[OK] Apptainer: {msg}")
        return success, msg

    def _check_gpu_support(self) -> Tuple[bool, str]:
        """Check if GPU support is available in the container."""
        success, msg = self._run_command([
            self.apptainer_cmd, "exec", "--nv", 
            str(self.container_path), "nvidia-smi"
        ])
        if success:
            print("[OK] GPU support available")
        return success, msg

    def _check_container_paths(self) -> Tuple[bool, str]:
        """Check if all required paths exist in the container."""
        for path in self.REQUIRED_CONTAINER_PATHS:
            flag = "-f" if path.endswith(".py") else "-d"
            success, _ = self._run_command([
                self.apptainer_cmd, "exec", str(self.container_path),
                "test", flag, path
            ])
            if not success:
                return False, f"Missing required path: {path}"
        
        print("[OK] All required paths present")
        return True, "All paths validated"

    def validate_environment(self, skip_gpu_check: bool = False) -> Tuple[bool, str]:
        """Validate the complete container environment.
        
        Args:
            skip_gpu_check: Whether to skip GPU availability check
        
        Returns:
            Tuple of (success, message)
        """
        # Check apptainer installation
        success, msg = self._check_apptainer_installation()
        if not success:
            return False, f"Apptainer not available: {msg}"

        # Check GPU support (optional)
        if not skip_gpu_check:
            success, msg = self._check_gpu_support()
            if not success:
                return False, f"GPU not available: {msg}"
        else:
            print("[SKIPPED] GPU check skipped by user request")

        # Check required paths
        success, msg = self._check_container_paths()
        if not success:
            return False, msg

        return True, "Environment validation passed"

    def _find_session_directory(self, subject_dir: Path) -> Path:
        """Find and validate the session directory."""
        scan_dir = subject_dir / self.SESSION_DIR
        if not scan_dir.exists():
            raise ValidationError(f"Expected {self.SESSION_DIR} directory not found in {subject_dir}")
        
        if not any(scan_dir.iterdir()):
            raise ValidationError(f"No scan data found in {scan_dir}")
        
        return scan_dir

    def _match_sequences(self, scan_dir: Path) -> Dict[str, Path]:
        """Match found files to required sequences."""
        sequences = {}
        print(f"[DEBUG] Scanning {scan_dir} for sequences...")

        for nii_file in scan_dir.glob("*.nii.gz"):
            print(f"[DEBUG] Found file: {nii_file.name}")
            for seq_name in self.config.sequences:
                if seq_name in nii_file.name.lower():
                    sequences[seq_name] = nii_file
                    print(f"[DEBUG] Matched {seq_name} -> {nii_file.name}")
                    break

        return sequences

    def find_sequences(self, subject_dir: Path) -> Dict[str, Path]:
        """Find valid sequences in subject directory.
        
        Args:
            subject_dir: Path to subject directory
            
        Returns:
            Dictionary mapping sequence names to file paths
            
        Raises:
            ValidationError: If insufficient sequences found
        """
        subject_dir = Path(subject_dir).resolve()
        scan_dir = self._find_session_directory(subject_dir)
        sequences = self._match_sequences(scan_dir)

        print(f"[DEBUG] Found {len(sequences)} sequences: {list(sequences.keys())}")

        if len(sequences) < self.config.min_sequences:
            raise ValidationError(
                f"Found {len(sequences)} sequences {list(sequences.keys())}, "
                f"need {self.config.min_sequences} minimum"
            )

        return sequences

    def _build_prediction_command(self, subject_dir: Path, sequences: Dict[str, Path], 
                                output_dir: Path, subject_id: str) -> List[str]:
        """Build the command for running predictions."""
        cmd = [
            self.apptainer_cmd, "exec", "--nv",
            "--bind", f"{subject_dir}:/input:ro",
            "--bind", f"{output_dir}:/output",
            str(self.container_path),
            "python", "/app/predict.py"
        ]

        # Add sequence arguments
        for seq_name, seq_path in sequences.items():
            relative_path = seq_path.relative_to(subject_dir)
            cmd.extend([f"--{seq_name}", f"/input/{relative_path}"])

        # Add output argument
        output_ext = ".nii.gz" if self.config.output_type == OutputType.SEGMENTATION else ".txt"
        cmd.extend(["--output", f"/output/{subject_id}{output_ext}"])

        return cmd

    def predict_subject(self, subject_id: str, subject_dir: Path, output_dir: Path) -> Tuple[bool, str]:
        """Run prediction for a single subject.
        
        Args:
            subject_id: Subject identifier
            subject_dir: Path to subject data
            output_dir: Path for output files
            
        Returns:
            Tuple of (success, message)
        """
        try:
            subject_dir = Path(subject_dir).resolve()
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            sequences = self.find_sequences(subject_dir)
            cmd = self._build_prediction_command(subject_dir, sequences, output_dir, subject_id)

            print(f"\n[INFO] Predicting {subject_id}...")
            print(f"[DEBUG] Subject dir: {subject_dir}")
            print(f"[DEBUG] Output dir: {output_dir}")
            print(f"[DEBUG] Sequences found: {len(sequences)}")
            print(f"[DEBUG] Command: {' '.join(cmd)}")

            success, output = self._run_command(cmd)

            if success:
                print(f"[OK] {subject_id} completed successfully")
                if output:
                    print(f"[INFO] Output: {output}")
            else:
                print(f"[ERROR] {subject_id} failed: {output}")

            return success, output

        except Exception as e:
            error_msg = f"Exception for {subject_id}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg

    def _get_subject_directories(self, data_dir: Path) -> List[Path]:
        """Get list of subject directories from preprocessed data."""
        preprocessed_dir = data_dir / "preprocessed"
        print(f"[DEBUG] Looking for preprocessed dir: {preprocessed_dir}")

        if not preprocessed_dir.exists():
            raise ValidationError(f"Preprocessed directory not found: {preprocessed_dir}")

        subject_dirs = [d for d in preprocessed_dir.iterdir() if d.is_dir()]
        if not subject_dirs:
            raise ValidationError(f"No subject directories found in {preprocessed_dir}")

        return subject_dirs

    def predict_all(self, data_dir: Path, output_dir: Path) -> Tuple[bool, str]:
        """Run predictions for all subjects.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path for output files
            
        Returns:
            Tuple of (success, message)
        """
        try:
            data_dir = Path(data_dir).resolve()
            output_dir = Path(output_dir).resolve()

            subject_dirs = self._get_subject_directories(data_dir)
            print(f"[INFO] Found {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

            results = []
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                success, msg = self.predict_subject(subject_id, subject_dir, output_dir)
                results.append((subject_id, success, msg))

            failed = [subj for subj, success, _ in results if not success]
            if failed:
                return False, f"Failed subjects: {', '.join(failed)}"

            return True, f"All {len(results)} subjects completed successfully"

        except Exception as e:
            return False, str(e)

    def _validate_text_outputs(self, output_dir: Path) -> Tuple[bool, str]:
        """Validate classification or regression text outputs."""
        txt_files = list(output_dir.glob(f"{self.SUBJECT_PREFIX}*.txt"))
        if not txt_files:
            return False, "No output text files found"

        for txt_file in txt_files:
            try:
                with txt_file.open("r") as f:
                    value = float(f.read().strip())
                    print(f"[DEBUG] {txt_file.name}: {value}")
            except ValueError:
                return False, f"Invalid float in {txt_file.name}"

        return True, f"All {len(txt_files)} {self.config.output_type.value} outputs valid"

    def _validate_segmentation_outputs(self, output_dir: Path, data_dir: Path) -> Tuple[bool, str]:
        """Validate segmentation mask outputs."""
        data_dir = Path(data_dir).resolve()
        preprocessed_dir = data_dir / "preprocessed"
        mask_files = list(output_dir.glob(f"{self.SUBJECT_PREFIX}*.nii.gz"))

        if not mask_files:
            return False, "No segmentation masks found"

        for mask_file in mask_files:
            subject_id = mask_file.stem.replace(self.SUBJECT_PREFIX, "")
            subject_dir = preprocessed_dir / f"{self.SUBJECT_PREFIX}{subject_id}"

            try:
                sequences = self.find_sequences(subject_dir)
                ref_img = nib.load(str(list(sequences.values())[0]))
                mask_img = nib.load(str(mask_file))

                # Check shape consistency
                if ref_img.shape != mask_img.shape:
                    return False, f"Shape mismatch for {subject_id}"

                # Check binary mask
                unique_vals = np.unique(mask_img.get_fdata())
                if not np.array_equal(unique_vals, [0.0, 1.0]):
                    return False, f"Non-binary mask for {subject_id}"

            except Exception as e:
                return False, f"Error validating {subject_id}: {str(e)}"

        return True, f"All {len(mask_files)} segmentation masks valid"

    def validate_outputs(self, output_dir: Path, data_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """Validate prediction outputs based on task type.
        
        Args:
            output_dir: Directory containing output files
            data_dir: Data directory (required for segmentation validation)
            
        Returns:
            Tuple of (success, message)
        """
        output_dir = Path(output_dir).resolve()

        if self.config.output_type in [OutputType.CLASSIFICATION, OutputType.REGRESSION]:
            return self._validate_text_outputs(output_dir)
        elif self.config.output_type == OutputType.SEGMENTATION:
            if not data_dir:
                return False, "Data directory required for segmentation validation"
            return self._validate_segmentation_outputs(output_dir, data_dir)

    def test_single_subject(self, data_dir: Path, output_dir: Path, 
                          subject_id: Optional[str] = None) -> Tuple[bool, str]:
        """Test a single subject for debugging purposes.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path for output files
            subject_id: Specific subject ID (uses first found if None)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            data_dir = Path(data_dir).resolve()
            output_dir = Path(output_dir).resolve()
            preprocessed_dir = data_dir / "preprocessed"

            if subject_id is None:
                subject_dirs = [d for d in preprocessed_dir.iterdir() if d.is_dir()]
                if not subject_dirs:
                    return False, "No subjects found"
                subject_dir = subject_dirs[0]
                subject_id = subject_dir.name
            else:
                subject_dir = preprocessed_dir / subject_id
                if not subject_dir.exists():
                    return False, f"Subject {subject_id} not found"

            print(f"=== Testing single subject: {subject_id} ===")
            return self.predict_subject(subject_id, subject_dir, output_dir)

        except Exception as e:
            return False, str(e)

    def run_full_validation(self, data_dir: Path, output_dir: Path, skip_gpu_check: bool = False) -> Tuple[bool, str]:
        """Run complete validation pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path for output files
            skip_gpu_check: Whether to skip GPU availability check
            
        Returns:
            Tuple of (success, message)
        """
        print("=== Container Validation ===")
        success, msg = self.validate_environment(skip_gpu_check)
        if not success:
            return False, msg

        print("\n=== Running Predictions ===")
        success, msg = self.predict_all(data_dir, output_dir)
        if not success:
            return False, msg

        print("\n=== Validating Outputs ===")
        success, msg = self.validate_outputs(output_dir, data_dir)
        if not success:
            return False, msg

        return True, "Full validation completed successfully"


import argparse
import sys


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Container Validator for Medical Imaging Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation for task1
  python validator.py --task task1 --container /path/to/task1container.sif --data-dir /path/to/data --output-dir /path/to/output

  # Test single subject
  python validator.py --task task2 --container /path/to/task2container.sif --data-dir /path/to/data --output-dir /path/to/output --test-subject sub_001

  # Only validate environment
  python validator.py --task task3 --container /path/to/task3container.sif --validate-env-only

  # Only validate outputs (skip prediction)
  python validator.py --task task1 --output-dir /path/to/output --data-dir /path/to/data --validate-outputs-only
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--task",
        choices=["task1", "task2", "task3"],
        required=True,
        help="Task to validate (task1=classification, task2=segmentation, task3=regression)"
    )
    
    # Container and environment arguments
    parser.add_argument(
        "--container",
        type=Path,
        help="Path to the container file (.sif)"
    )
    
    parser.add_argument(
        "--apptainer-cmd",
        default="apptainer",
        help="Command to run apptainer (default: apptainer)"
    )
    
    # Data directories
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to the data directory containing preprocessed subjects"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the output directory for predictions"
    )
    
    # Operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--validate-env-only",
        action="store_true",
        help="Only validate the container environment (skip predictions)"
    )
    
    mode_group.add_argument(
        "--validate-outputs-only",
        action="store_true",
        help="Only validate existing outputs (skip environment check and predictions)"
    )
    
    mode_group.add_argument(
        "--test-subject",
        type=str,
        help="Test only a specific subject (e.g., sub_001) for debugging"
    )
    
    # Additional options
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU availability check during environment validation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments and check for required combinations."""
    # Container is required unless only validating outputs
    if not args.validate_outputs_only and not args.container:
        raise ValueError("--container is required unless using --validate-outputs-only")
    
    # Data directory is required for most operations
    if not args.validate_env_only and not args.data_dir:
        raise ValueError("--data-dir is required unless using --validate-env-only")
    
    # Output directory is required for predictions and output validation
    if not args.validate_env_only and not args.output_dir:
        raise ValueError("--output-dir is required unless using --validate-env-only")
    
    # Check if files/directories exist
    if args.container and not args.container.exists():
        raise ValueError(f"Container file not found: {args.container}")
    
    if args.data_dir and not args.data_dir.exists():
        raise ValueError(f"Data directory not found: {args.data_dir}")


def run_environment_validation(validator: ContainerValidator, skip_gpu_check: bool = False) -> bool:
    """Run environment validation and return success status."""
    print("=== Container Environment Validation ===")
    success, msg = validator.validate_environment(skip_gpu_check)
    
    if success:
        print(f"[SUCCESS] {msg}")
    else:
        print(f"[FAILURE] {msg}")
    
    return success


def run_predictions(validator: ContainerValidator, data_dir: Path, output_dir: Path, 
                   test_subject: Optional[str] = None) -> bool:
    """Run predictions and return success status."""
    if test_subject:
        print(f"=== Testing Single Subject: {test_subject} ===")
        success, msg = validator.test_single_subject(data_dir, output_dir, test_subject)
    else:
        print("=== Running Predictions for All Subjects ===")
        success, msg = validator.predict_all(data_dir, output_dir)
    
    if success:
        print(f"[SUCCESS] {msg}")
    else:
        print(f"[FAILURE] {msg}")
    
    return success


def run_output_validation(validator: ContainerValidator, output_dir: Path, 
                         data_dir: Optional[Path] = None) -> bool:
    """Run output validation and return success status."""
    print("=== Output Validation ===")
    success, msg = validator.validate_outputs(output_dir, data_dir)
    
    if success:
        print(f"[SUCCESS] {msg}")
    else:
        print(f"[FAILURE] {msg}")
    
    return success


def main():
    """Main execution function with command-line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Create validator instance (only if container is provided)
        validator = None
        if args.container:
            validator = ContainerValidator(
                container_path=args.container,
                task=args.task,
                apptainer_cmd=args.apptainer_cmd
            )
        
        print(f"=== {args.task.upper()} Container Validation ===")
        
        # Track overall success
        overall_success = True
        
        # Execute based on mode
        if args.validate_env_only:
            # Only validate environment
            if not validator:
                print("[ERROR] Container path required for environment validation")
                sys.exit(1)
            success = run_environment_validation(validator, args.skip_gpu_check)
            overall_success = success
            
        elif args.validate_outputs_only:
            # Only validate outputs
            if not validator:
                # Create minimal validator for output validation
                validator = ContainerValidator.__new__(ContainerValidator)
                validator.task = args.task
                validator.config = ContainerValidator.TASK_CONFIGS[args.task]
            
            success = run_output_validation(validator, args.output_dir, args.data_dir)
            overall_success = success
            
        else:
            # Full validation or single subject test
            if not validator:
                print("[ERROR] Container path required for predictions")
                sys.exit(1)
            
            # Environment validation
            success = run_environment_validation(validator, args.skip_gpu_check)
            overall_success = overall_success and success
            
            if success:  # Only proceed if environment validation passed
                # Run predictions
                success = run_predictions(validator, args.data_dir, args.output_dir, args.test_subject)
                overall_success = overall_success and success
                
                if success:  # Only validate outputs if predictions succeeded
                    # Output validation
                    success = run_output_validation(validator, args.output_dir, args.data_dir)
                    overall_success = overall_success and success
        
        # Final result
        if overall_success:
            print(f"\n[FINAL RESULT] {args.task.upper()}: SUCCESS - All validations passed")
            sys.exit(0)
        else:
            print(f"\n[FINAL RESULT] {args.task.upper()}: FAILURE - One or more validations failed")
            sys.exit(1)
            
    except ValidationError as e:
        print(f"[ERROR] Validation Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        parser.print_help()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
