import pandas as pd
import numpy as np
import yaml
import os, sys
from src.logger import logging
from src.exception import CustomException
import joblib

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
    -----
    file_path : str
        Path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        The data from the CSV file.
    """
    try:
        logging.info("Entered the read_csv_file method of main_utils")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Exited the read_csv_file method of main_utils")
        return df
        
    except Exception as e:
        logging.error(f"Error reading CSV file: {file_path}")
        raise CustomException(e, sys)


def save_csv_file(data: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
    -----
    data : pd.DataFrame
        The DataFrame to save.
    file_path : str
        Path where the CSV file will be saved.
    """
    try:
        logging.info("Entered the save_csv_file method of main_utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logging.info("Exited the save_csv_file method of main_utils")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    

    try:
        logging.info("Entered the load_object method of main_utils")
        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)

        logging.info("Exited the load_object method of main_utils")
        return obj

    except Exception as e:
        raise CustomException(e, sys)
    


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        logging.info("Entered the save_numpy_array_data method of main_utils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info("Exited the save_numpy_array_data method of main_utils")

    except Exception as e:
        raise CustomException(e, sys)
    

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CustomException(e, sys)
    
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys)
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    