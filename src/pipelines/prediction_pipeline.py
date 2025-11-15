import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object
from keras.models import load_model
import tensorflow as tf

class PredictionPipeline:
    def __init__(self):
        """
        Initializes the prediction pipeline by loading the preprocessor and trained model.
        """
        try:
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model_trainer", "lstm.keras")

            # Load preprocessor and model
            self.preprocessor = load_object(preprocessor_path)
            self.model = load_model(model_path)

            logging.info("✅ PredictionPipeline initialized successfully.")
        except Exception as e:
            logging.error("❌ Error initializing PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, text: str):
        """
        Transforms input features using the preprocessor and generates model predictions.
        Args:
            text (str): Raw text from user.
        Returns:
            np.ndarray: Model predictions.
        """
        try:

            logging.info(f"Input text recieved -> {text}")
            logging.info("🔄 Transforming input text using custom preprocessor")
            transformed_features = self.preprocessor.transform([text])

            logging.info("🧠 Generating predictions using trained model...")
            preds = self.model.predict(tf.constant([transformed_features]))

            logging.info(f"✅ Predictions generated successfully: {preds}")
            return preds

        except Exception as e:
            logging.error("❌ Error occurred during prediction.")
            raise CustomException(e, sys)