import os
import sys
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from utils.common_functions import read_yaml
from config.path_config import *
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok = True)
        logging.info(f"Data ingestion from {self.bucket_name}/{self.bucket_file_name} to {RAW_DIR}")
    
    def download_csv_from_gcp(self):
        try:
            # gcp client
            client = storage.Client()
            # initiating bucket
            bucket = client.bucket(self.bucket_name)
            # initiating blob
            blob = bucket.blob(self.bucket_file_name)
            # downloading to the ras csv file path
            blob.download_to_filename(RAW_FILE_PATH)
            logging.info("CSV file is succesfully downloaded")
        except Exception as e:
            logging.error("An error has occured while initiating the client in downloading the csv file")
            raise CustomException(e, sys)
    
    def split_data(self):
        try:
            logging.info("Splitting the data into training and testing files")
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(data, test_size=1-self.train_ratio, random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            logging.info("Train data saved")
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logging.info("Test data saved")
        
        except Exception as e:
            logging.error("Error while splitting the data and loading")
            raise CustomException(e, sys)
    
    def run(self):
        try:
            logging.info("Initiating the data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logging.info("Data ingestion completed")
        except Exception as e:
            logging.error("Error while running the whole data ingestion pipeline")
            raise CustomException(e, sys)
        
if __name__=='__main__':
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

