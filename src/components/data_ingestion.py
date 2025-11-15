# data_ingestion.py

import os, sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_csv_file, read_yaml_file
from db.connection import get_connection
from db.queries import get_all_data


@dataclass
class DataIngestionConfig:
    """Holds file paths and parameters for the data ingestion process."""
    raw_data_dir: str = os.path.join("artifacts", "data_ingestion")
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    table_name_path: str = os.path.join("config", "db.yaml")


class DataIngestion:
    """Handles loading data from the database and splitting into train/test."""

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """Initialize with a configuration object and read table name."""
        self.config = config
        self.table_name_config = read_yaml_file(self.config.table_name_path)
        os.makedirs(self.config.raw_data_dir, exist_ok=True)

    def fetch_data_from_db(self) -> pd.DataFrame:
        """Fetch all rows from the configured database table as a pandas DataFrame."""
        try:
            table_name = self.table_name_config.get("data_table")
            if not table_name:
                raise ValueError("Table name not found in db.yaml")

            with get_connection() as conn:
                stmt = get_all_data(table_name)
                result = conn.execute(stmt)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            logging.info(f"Fetched {df.shape[0]} rows and {df.shape[1]} columns from table '{table_name}'")
            return df

        except Exception as e:
            logging.error(f"Failed to fetch data from table '{self.table_name_config.get('data_table', 'unknown')}'")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """Perform the full data ingestion pipeline: fetch, save raw, split, and save train/test."""
        logging.info("===== Data Ingestion Process Started =====")

        try:
            # 1️⃣ Fetch data
            data = self.fetch_data_from_db()
            logging.info(f"Data shape: {data.shape}")

            # 2️⃣ Save raw data
            save_csv_file(data, self.config.raw_data_path)
            logging.info(f"Raw data saved at {self.config.raw_data_path}")

            # 3️⃣ Split into train/test
            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            # 4️⃣ Save splits
            save_csv_file(train_set, self.config.train_data_path)
            save_csv_file(test_set, self.config.test_data_path)

            logging.info(f"Train data saved at {self.config.train_data_path} (shape: {train_set.shape})")
            logging.info(f"Test data saved at {self.config.test_data_path} (shape: {test_set.shape})")
            logging.info("===== Data Ingestion Completed Successfully =====")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"✅ Train data saved to: {train_path}")
    print(f"✅ Test data saved to: {test_path}")
