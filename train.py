"""
YOLO Training Script for Defect Detection

This script trains a YOLO model on the defect detection dataset.
It integrates with MLflow for experiment tracking and model management.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
load_dotenv()
import torch
import mlflow
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


def train_yolo(
    data_yaml: str = "data/data.yaml",
    model_name: str = "yolo11n",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    patience: int = 20,
    device: Optional[Union[int, str]] = None,
    experiment_name: str = "yolo-training",
    run_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train YOLO model on the defect detection dataset.
    Supports YOLOv8 and YOLOv11 (requires ultralytics>=8.3.0).
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_name: YOLO model variant (yolo11n, yolo11s, yolo11m, yolo11l, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        patience: Early stopping patience
        device: Device to use - int for CUDA GPU ID, 'mps' for Apple Silicon, 'cpu', or None for auto-select (CUDA > MPS > CPU)
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        **kwargs: Additional arguments to pass to YOLO.train()
        
    Returns:
        Dictionary containing training results
    """
    
    logger.info("=" * 50)
    logger.info("YOLO Training Script")
    logger.info("=" * 50)
    
    # Validate data.yaml exists
    data_path = PROJECT_ROOT / data_yaml
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_path}")
    
    logger.info(f"Data YAML: {data_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {imgsz}")
    
    # Check device availability (CUDA GPU > MPS > CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Determine device name for logging
    if isinstance(device, int):
        device_name = f"CUDA GPU {device}"
    elif device == "mps":
        device_name = "MPS (Apple Silicon)"
    else:
        device_name = device.upper()
    
    logger.info(f"Device: {device_name}")
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Configure MLflow for tracking
    # Use file-based backend by default to avoid SQLite concurrency issues
    # For production, consider using a remote MLflow server with PostgreSQL
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    if experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    if run_name:
        os.environ["MLFLOW_RUN_NAME"] = run_name
    
    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow run name: {run_name}")
    
    try:
        # Check ultralytics version
        import ultralytics
        ul_version = ultralytics.__version__
        logger.info(f"Ultralytics version: {ul_version}")

        # Initialize YOLO model
        logger.info(f"Loading {model_name} model...")
        logger.info("Note: If this is the first time, the model will be downloaded automatically...")
        model_path = MODELS_DIR / f"{model_name}.pt"
        model = YOLO(model_path)

        # Train the model
        logger.info("Starting training...")
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            save=True,
            cache=False,
            plots=True,
            **kwargs
        )
        
        logger.info("Training completed successfully!")
        
        # Log training metrics from results
        try:
            logger.info("\n" + "=" * 50)
            logger.info("Training Metrics Summary")
            logger.info("=" * 50)
            
            # Try to get metrics from results object
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics' in metrics:
                    m = metrics['metrics']
                    logger.info(f"Precision: {m.get('precision', 'N/A')}")
                    logger.info(f"Recall: {m.get('recall', 'N/A')}")
                    logger.info(f"mAP50: {m.get('mAP50', 'N/A')}")
                    logger.info(f"mAP50-95: {m.get('mAP50-95', 'N/A')}")
            
            # Try to read from results.csv file (YOLO saves metrics here)
            results_csv = RUNS_DIR / run_name / "results.csv"
            if results_csv.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(results_csv)
                    if not df.empty:
                        # Get the last row (final epoch metrics)
                        last_row = df.iloc[-1]
                        logger.info(f"\nFinal Epoch Metrics (from results.csv):")
                        if 'metrics/precision(B)' in df.columns:
                            logger.info(f"  Precision: {last_row.get('metrics/precision(B)', 'N/A'):.4f}")
                        if 'metrics/recall(B)' in df.columns:
                            logger.info(f"  Recall: {last_row.get('metrics/recall(B)', 'N/A'):.4f}")
                        if 'metrics/mAP50(B)' in df.columns:
                            logger.info(f"  mAP50: {last_row.get('metrics/mAP50(B)', 'N/A'):.4f}")
                        if 'metrics/mAP50-95(B)' in df.columns:
                            logger.info(f"  mAP50-95: {last_row.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
                        logger.info(f"  Best epoch: {last_row.get('epoch', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Could not read results.csv: {e}")
            
            logger.info("=" * 50)
        except Exception as e:
            logger.warning(f"Could not extract training metrics: {e}")
        
        # Save the best model
        best_model_path = MODELS_DIR / f"{model_name}-best-{run_name}.pt"
        model.save(str(best_model_path))
        logger.info(f"Best model saved to: {best_model_path}")
        
        return {
            "status": "success",
            "run_name": run_name,
            "model_path": str(best_model_path),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def validate_model(
    model_path: str,
    data_yaml: str = "data/data.yaml",
    imgsz: int = 640,
    device: Optional[Union[int, str]] = None,
    split: str = "val",
) -> Dict[str, Any]:
    """
    Validate a trained model on a specific dataset split.
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to data.yaml configuration file
        imgsz: Input image size
        device: GPU device ID, 'mps', 'cpu', or None for auto-select
        split: Dataset split to validate on ('val' or 'test')
        
    Returns:
        Dictionary with validation metrics
    """
    
    logger.info(f"Validating model on {split} set: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    results = model.val(
        data=str(PROJECT_ROOT / data_yaml),
        imgsz=imgsz,
        device=device,
        split=split
    )
    
    # Extract key metrics
    metrics = {
        "precision": float(results.box.p.mean()) if hasattr(results.box, 'p') else 0.0,
        "recall": float(results.box.r.mean()) if hasattr(results.box, 'r') else 0.0,
        "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
    }
    
    logger.info(f"{split.capitalize()} set metrics:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  mAP@50: {metrics['mAP50']:.4f}")
    logger.info(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    sys.stdout.flush()  # Ensure logs are visible in CI/CD
    
    return metrics


def _upload_model_to_s3(model_path: str, s3_bucket: str, s3_key: str) -> None:
    """
    Upload the trained model file to AWS S3.
    
    Args:
        model_path: Local path to the model file
        s3_bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)
    """
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    
    # Clean bucket name (remove trailing slash if any)
    s3_bucket = s3_bucket.rstrip('/')
    
    logger.info(f"Uploading model to S3: s3://{s3_bucket}/{s3_key}")
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.upload_file(model_path, s3_bucket, s3_key)
        logger.info("Model uploaded to S3 successfully!")
    except FileNotFoundError:
        logger.error(f"The model file was not found: {model_path}")
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
    except ClientError as e:
        logger.error(f"Failed to upload model to S3: {e}")


def register_model_to_mlflow(
    model_path: str,
    model_name: str,
    test_metrics: Dict[str, float],
    run_name: str,
    experiment_name: str,
    promotion_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Register the trained model to MLflow Model Registry with test metrics.
    Finds the existing training run and updates it instead of creating a new one.
    
    Args:
        model_path: Path to the trained model file
        model_name: Name for the registered model
        test_metrics: Dictionary of test set metrics
        run_name: MLflow run name
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with model URI, promotion status, and version
    """
    logger.info("Registering model to MLflow...")
    
    # Set MLflow tracking URI and experiment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    # Find the existing training run by run_name
    experiment = client.get_experiment_by_name(experiment_name)
    run_id = None
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
            order_by=["start_time DESC"]
        )
        if runs:
            run_id = runs[0].info.run_id
            logger.info(f"Found existing training run: {run_id}")
    
    # Update existing run or create new one if not found
    if run_id:
        # Update existing run with test metrics and model artifact
        client.log_metric(run_id, "test_precision", test_metrics["precision"])
        client.log_metric(run_id, "test_recall", test_metrics["recall"])
        client.log_metric(run_id, "test_mAP50", test_metrics["mAP50"])
        client.log_metric(run_id, "test_mAP50-95", test_metrics["mAP50-95"])
        client.log_artifact(run_id, model_path, artifact_path="model")
    else:
        # Fallback: create a new run if training run not found
        logger.warning(f"Training run '{run_name}' not found, creating new run")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_metrics({
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_mAP50": test_metrics["mAP50"],
                "test_mAP50-95": test_metrics["mAP50-95"],
            })
            mlflow.log_artifact(model_path, artifact_path="model")
            run_id = run.info.run_id
    
    # Register the model
    model_filename = os.path.basename(model_path)
    artifact_uri = f"runs:/{run_id}/model/{model_filename}"
    
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # Model already exists
    
    result = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id
    )
    
    logger.info(f"Model registered: {model_name} v{result.version}")
    
    # Check if should promote to production (compare with previous best)
    current_map = test_metrics["mAP50-95"]
    should_promote = True
    previous_map = None
    improvement = None
    
    logger.info(f"\nPromotion threshold: {promotion_threshold:.4f} (model must improve by at least this amount)")
    
    try:
        # Get production model's metrics if exists
        prod_version = client.get_model_version_by_alias(model_name, "production")
        if prod_version and prod_version.version != result.version:
            prod_run = client.get_run(prod_version.run_id)
            previous_map = prod_run.data.metrics.get("test_mAP50-95", 0.0)
            improvement = current_map - previous_map
            
            logger.info(f"\nComparing with production model:")
            logger.info(f"  Current model mAP@50-95: {current_map:.4f}")
            logger.info(f"  Production model mAP@50-95: {previous_map:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            logger.info(f"  Required improvement: >= {promotion_threshold:.4f}")
            
            if previous_map >= current_map:
                should_promote = False
                logger.info(f"  → Production model is better or equal, NOT promoting")
            elif improvement < promotion_threshold:
                should_promote = False
                logger.info(f"  → Improvement ({improvement:.4f}) < threshold ({promotion_threshold:.4f}), NOT promoting")
            else:
                logger.info(f"  → Improvement ({improvement:.4f}) >= threshold ({promotion_threshold:.4f}), WILL promote")
        else:
            logger.info(f"\nNo existing production model found - will promote to production")
    except Exception:
        logger.info(f"\nNo existing production model found - will promote to production")
        pass  # No production model exists yet
    
    # Set production alias if best model
    if should_promote:
        client.set_registered_model_alias(model_name, "production", result.version)
        logger.info(f"\n✅ Model v{result.version} PROMOTED TO PRODUCTION")
        s3_bucket = os.getenv("S3_BUCKET", "").rstrip('/')
        if s3_bucket:
            _upload_model_to_s3(
                model_path=model_path,
                s3_bucket=s3_bucket,
                s3_key=f"{model_name}/best.pt"
            )
        else:
            logger.warning("S3_BUCKET not set, skipping S3 upload")
    else:
        logger.info(f"\nℹ Model v{result.version} registered but NOT promoted to production")
    
    return {
        "model_uri": f"models:/{model_name}/{result.version}",
        "promoted": should_promote,
        "version": result.version,
        "previous_map": previous_map,
        "improvement": improvement,
        "threshold": promotion_threshold
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv11 for defect detection")
    
    # Training arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to data.yaml configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="YOLO model variant (yolo11n, yolo11s, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: GPU ID (e.g., 0), 'mps' for Apple Silicon, 'cpu', or None for auto-select"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="yolo-training",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name"
    )
    
    # Validation arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Show validation set metrics (test set evaluation is always performed)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse device argument (convert numeric strings to int)
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        
        # Train the model (uses train/val sets internally)
        logger.info("Training uses train/val sets to find the best model...")
        result = train_yolo(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            patience=args.patience,
            device=device,
            experiment_name=args.experiment_name,
            run_name=args.run_name
        )
        
        model_path = result["model_path"]
        run_name = result["run_name"]
        
        # Validate on validation set if requested
        if args.validate:
            logger.info("\n" + "=" * 50)
            logger.info("Validation Set Evaluation")
            logger.info("=" * 50)
            validate_model(model_path, args.data, imgsz=args.imgsz, device=device, split="val")
        
        # Evaluate on test set if available (optional)
        test_metrics = None
        test_images_dir = PROJECT_ROOT / args.data.replace("data.yaml", "").replace("\\", "/").rstrip("/") / "test" / "images"
        if test_images_dir.exists() and list(test_images_dir.glob("*.jpg")) or list(test_images_dir.glob("*.png")):
            logger.info("\n" + "=" * 50)
            logger.info("Test Set Evaluation")
            logger.info("=" * 50)
            try:
                test_metrics = validate_model(
                    model_path=model_path,
                    data_yaml=args.data,
                    imgsz=args.imgsz,
                    device=device,
                    split="test"
                )
            except Exception as e:
                logger.warning(f"Test set evaluation skipped: {e}")
                test_metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
        else:
            logger.info("\n" + "=" * 50)
            logger.info("Test Set Evaluation")
            logger.info("=" * 50)
            logger.info("Test set not found or empty - skipping test evaluation")
            test_metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
        
        # Log test metrics clearly
        logger.info("\n" + "=" * 50)
        logger.info("Test Set Metrics Summary")
        logger.info("=" * 50)
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  mAP@50: {test_metrics['mAP50']:.4f}")
        logger.info(f"  mAP@50-95: {test_metrics['mAP50-95']:.4f}")
        logger.info("=" * 50)
        
        # Register model to MLflow with test metrics
        logger.info("\n" + "=" * 50)
        logger.info("Model Registration")
        logger.info("=" * 50)
        registered_model_name = f"{args.model}-detection"
        # Promotion threshold: model must improve by at least 0.6 mAP to be promoted
        promotion_threshold = 0.6
        registration_result = register_model_to_mlflow(
            model_path=model_path,
            model_name=registered_model_name,
            test_metrics=test_metrics,
            run_name=run_name,
            experiment_name=args.experiment_name,
            promotion_threshold=promotion_threshold
        )
        
        logger.info("\n" + "=" * 50)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 50)
        logger.info(f"Best model: {model_path}")
        logger.info(f"Registered as: {registered_model_name}")
        logger.info(f"Model version: {registration_result['version']}")
        logger.info(f"Model URI: {registration_result['model_uri']}")
        logger.info("")
        logger.info("Test Metrics:")
        logger.info(f"  mAP@50-95: {test_metrics['mAP50-95']:.4f}")
        logger.info(f"  mAP@50: {test_metrics['mAP50']:.4f}")
        logger.info("")
        if registration_result['promoted']:
            logger.info(f"✅ PROMOTED TO PRODUCTION!")
            logger.info(f"   Current mAP@50-95: {test_metrics['mAP50-95']:.4f}")
            if registration_result.get('previous_map') is not None:
                logger.info(f"   Previous production mAP@50-95: {registration_result['previous_map']:.4f}")
                logger.info(f"   Improvement: {registration_result.get('improvement', 0):+.4f}")
        else:
            logger.info(f"ℹ Model registered but NOT promoted to production")
            if registration_result.get('previous_map') is not None:
                logger.info(f"   Current mAP@50-95: {test_metrics['mAP50-95']:.4f}")
                logger.info(f"   Production mAP@50-95: {registration_result['previous_map']:.4f}")
                improvement = registration_result.get('improvement', 0)
                threshold = registration_result.get('threshold', 0.6)
                if improvement is not None:
                    logger.info(f"   Improvement: {improvement:+.4f}")
                    logger.info(f"   Required improvement: >= {threshold:.4f}")
                    if improvement < threshold:
                        logger.info(f"   → Improvement too small, need at least {threshold:.4f} to promote")
                    else:
                        logger.info(f"   → Production model is better or equal")
            else:
                logger.info(f"   → No production model to compare")
        logger.info("=" * 50)
        sys.stdout.flush()  # Ensure logs are visible in CI/CD
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

##comment push
