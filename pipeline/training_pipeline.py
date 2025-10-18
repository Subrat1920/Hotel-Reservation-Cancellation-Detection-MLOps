import os, sys
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTrainer
from utils.common_functions import read_yaml, load_data
from config.path_config import *
from config.model_params import *
from src.logger import logging
from src.exception import CustomException


if __name__=="__main__":
    # 1. Data Ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    
    # 2. Data Processing
    logging.info("Initiating the processing the data")
    processor = DataProcessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    logging.info("Calling the wrapper method for completing the processing")
    processor.process()
    logging.info("Completed processing of the data")

    # 3. Model Training
    logging.info("Model Trainer Main function Wrapper Started")
    trianer = ModelTrainer(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
    trianer.run()
    logging.info("Model trainer with mlflow logging completed")