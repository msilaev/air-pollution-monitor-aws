import logging
import os

from src.config import USE_S3
from src.data.data_ingestion import DataIngestion
from src.data.data_loader import DataLoader
from src.models.pollution_predictor import PollutionPredictor

logger = logging.getLogger(__name__)

# Debug log for S3 bucket
print("[DEBUG] AWS_S3_DATA_BUCKET:", os.environ.get("AWS_S3_DATA_BUCKET"))


def main():
    try:
        logger.info("Starting model training")
        # Collect training data
        data_ingestion = DataIngestion(use_s3=USE_S3)
        data_ingestion.fetch_pollution_data(chunk_size_hours=168, week_number=2)

        # Load and validate the data
        data_loader = DataLoader(use_s3=USE_S3)
        df = data_loader.load_train_dataset()

        if df is None or df.empty:
            raise ValueError("No valid training data found.")

        predictor = PollutionPredictor()
        predictor.setup_mlflow()
        metrics = predictor.train(df)

        logger.info(f"Model training completed with metrics: {metrics}")

        return metrics

    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
