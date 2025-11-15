from src.utils.main_utils import save_csv_file, save_object, read_csv_file, read_yaml_file
from src.utils.text_utils import remove_urls, convert_emojis_to_text, remove_punct_numbers, lemmatize_words, remove_accents_diacritics
import pandas as pd
import contractions
from nltk.tokenize import word_tokenize
import re
import os, sys
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import logging
from src.exception import CustomException


# Custom Text Preprocessor Transformer
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)

            # Apply all preprocessing steps
            text = text.lower()
            text = re.sub(r'@\w+', '', text)                        # Remove mentions
            text = remove_urls(text)                                # Remove URLs
            text = remove_accents_diacritics(text)                  # Normalize accents
            text = contractions.fix(text)                           # Expand contractions
            text = convert_emojis_to_text(text)                     # Convert emojis to text
            text = remove_punct_numbers(text)                       # Remove punctuation & numbers
            text = word_tokenize(text)                              # Tokenize
            text = lemmatize_words(text)                            # Lemmatize
            text = " ".join(text)                                   # Rejoin
            cleaned.append(text)
        return cleaned


# Config Class
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_file_path: str = os.path.join('artifacts', 'data_transformation', 'preprocessor.pkl')
        self.train_file_path: str = os.path.join('artifacts', 'data_transformation', 'train.csv')
        self.test_file_path: str = os.path.join('artifacts', 'data_transformation', 'test.csv')
        self.SCHEMA_PATH: str = os.path.join('config', 'schema.yaml')
        os.makedirs(os.path.dirname(self.preprocessor_file_path), exist_ok=True)


# Main Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        # Load schema file
        self.schema = read_yaml_file(self.config.SCHEMA_PATH)
        self.target_col = self.schema.get('target')
        self.text_col = self.schema.get('text')
        self.drop_col = self.schema.get('drop')


    def _feature_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info('Feature cleaning step started')

            df = df.copy()
            # Keep only English
            df = df[df[self.drop_col] == 'en']

            # Drop Language column
            df.drop(self.drop_col, axis=1, inplace=True)

            # Drop duplicates
            df.drop_duplicates(inplace=True, ignore_index=True)

            # Map target labels
            df[self.target_col] = df[self.target_col].map({'negative': 0, 'positive': 1})

            logging.info('Feature cleaning step completed successfully')
            return df

        except Exception as e:
            logging.error("Error in Feature cleaning")
            raise CustomException(e, sys)
    

    def _text_preprocessing(self, df: pd.DataFrame, preprocessor) -> pd.DataFrame:
        try:
            logging.info('Text Preprocessing step started')
            df = df.copy()
            df[self.text_col] = preprocessor.transform(df[self.text_col].values)

            logging.info('Text Preprocessing step completed successfully')
            return df

        except Exception as e:
            logging.error("Error in Text Preprocessing")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("===== Data Transformation Started =====")

            # Load data
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Feature cleaning
            train_df = self._feature_cleaning(train_df)
            test_df = self._feature_cleaning(test_df)

            # Instantiate preprocessor
            preprocessor = TextPreprocessor()

            # Apply preprocessing
            train_df = self._text_preprocessing(train_df, preprocessor)
            test_df = self._text_preprocessing(test_df, preprocessor)

            # Split features and target
            X_train = train_df[self.text_col].values
            y_train = train_df[self.target_col].values
            X_test = test_df[self.text_col].values
            y_test = test_df[self.target_col].values

            # Save preprocessed data
            save_csv_file(train_df, self.config.train_file_path)
            save_csv_file(test_df, self.config.test_file_path)

            # Save preprocessor object
            save_object(self.config.preprocessor_file_path, preprocessor)

            logging.info("===== Data Transformation Completed Successfully =====")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error in data transformation pipeline")
            raise CustomException(e, sys)


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation  # make sure it's imported

    # Initialize data ingestion
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    # Perform ingestion to get train and test file paths
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize DataTransformation
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Quick sanity checks
    print("✅ Transformed X_train shape:", X_train_transformed.shape)
    print("✅ Transformed X_test shape:", X_test_transformed.shape)
    print("✅ y_train shape:", y_train.shape)
    print("✅ y_test shape:", y_test.shape)