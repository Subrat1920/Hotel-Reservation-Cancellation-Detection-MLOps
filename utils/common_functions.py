import os
import sys
from pathlib import Path
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import yaml

def read_yaml(file_path:Path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not the given path")
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.info("Succesfully read the YAML File")
            return config
    except Exception as e:
        logging.error("An [ERROR] while reading the YAML File")
        raise CustomException(e, sys)
    

def load_data(csv_path):
    try:
        logging.info("loading the data from the given path")
        return pd.read_csv(csv_path)
    except Exception as e:
        raise CustomException(e, sys)