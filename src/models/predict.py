import json
import logging
import os

import boto3

from src.config import PREDICTIONS_DIR, USE_S3
from src.data.data_ingestion import DataIngestion
from src.data.data_loader import DataLoader
from src.models.pollution_predictor import PollutionPredictor

logger = logging.getLogger(__name__)


def upload_predictions_to_s3(prediction):
    """Upload DataFrame to S3 as parquet file"""

    predictions_filename = "latest_predictions.json"

    try:
        s3_client = boto3.client("s3")
        s3_key = f"predictions/{predictions_filename}"

        bucket = os.environ.get("AWS_S3_BUCKET_NAME", "air-pollution-models")
        bucket = bucket.replace("s3://", "").strip()

        # print(credentials.access_key, credentials.secret_key, credentials.token)
        # print(f"Uploading data to s3://{bucket}/{key}")
        # Convert DataFrame to parquet in memory
        # buffer = io.BytesIO()
        # df.to_parquet(buffer, index=False)
        # buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(prediction),
            ContentType="application/json",
        )

        logger.info(f"Predictions saved to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload data to S3: {e}")
        raise


def save_predictions_locally(prediction, file_name):
    """Save predictions to local file"""
    try:
        with open(file_name, "w") as f:
            json.dump(prediction, f, indent=2)
        logger.info(f"Predictions saved locally to {file_name}")
    except Exception as e:
        logger.error(f"Failed to save predictions locally: {e}")
        raise


def main():
    try:
        logger.info("Starting prediction")
        # Collect training data
        data_ingestion = DataIngestion(use_s3=USE_S3)
        data_ingestion.fetch_pollution_data(chunk_size_hours=48, week_number=1)

        # Load and validate the data
        data_loader = DataLoader(use_s3=USE_S3)
        df = data_loader.load_predicting_dataset()

        print(f"Loaded {len(df)} records for prediction")
        print(f"Data interval: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

        if df is None or df.empty:
            raise ValueError("No valid training data found.")

        predictor = PollutionPredictor()
        predictor.setup_mlflow()

        # Load model if not already loaded
        if not predictor.model:
            model_loaded = predictor.load_model_from_mlflow()
            if not model_loaded:
                raise ValueError(
                    "Model not loaded from MLflow. Please train a model first."
                )

        prediction = predictor.predict(df, target_timestamp=df["Timestamp"].max())
        logger.info("Model prediction completed")

        # Upload predictions to S3
        if USE_S3:
            upload_predictions_to_s3(prediction)

        else:
            logger.warning("S3 upload is disabled, saving predictions locally only")
            local_file_path = os.path.join(PREDICTIONS_DIR, "latest_predictions.json")
            save_predictions_locally(prediction, local_file_path)

    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
