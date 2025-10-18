import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class DataProcessing:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def preprocess_data(self, df):
        try:
            logging.info("Preprocessing of data started")
            logging.info("Droping the Booking ID columns => not required")
            df = df.drop(columns=['Booking_ID'], axis=1)
            logging.info("Droping the duplicated columns")
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logging.info("Numerical and Categorical are extracted")
            logging.info("Applying Label Encoder to Categorical Colums")
            label_encoder = LabelEncoder()
            mappings={}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logging.info(f"Printing the mapped labels=>")
            for col, mapping in mappings.items():
                logging.info(f"{col}: {mapping}")
            
            logging.info("Managing the skeewness")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]

            skewness = df[num_cols].apply(lambda x:x.skew())

            logging.info("Applying log transformation")

            for col in skewness[skewness>skew_threshold].index:
                df[col] = np.log1p(df[col])
            
            return df

        except Exception as e:
            logging.error("There is an error while processing the data")
            raise CustomException(e, sys)
    
    def balance_data(self, df):
        try:
            logging.info("Managing the imbalanced dataset")
            smote=SMOTE(random_state=42)
            x=df.drop(columns=['booking_status'])
            y=df['booking_status']
            x_res, y_res = smote.fit_resample(x, y)
            logging.info("Handline imbalanced dataset is over")
            logging.info("Merging them into a single dataframe")
            balanced_df = pd.DataFrame(x_res, columns=x.columns)
            balanced_df['booking_status']=y_res
            logging.info("Data balanced and merged succesfully")
            return balanced_df

        except Exception as e:
            logging.error("There is an error while managing the the imbalanced data")
            raise CustomException(e, sys)
    
    def select_features(self, df):
        try:
            logging.info("Initiating feature selection")
            x=df.drop(columns=['booking_status'])
            y=df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {
                    "features":x.columns,
                    "importance":feature_importance
                }
            )
            top_feature_df = feature_importance_df.sort_values(by='importance',     ascending=False)
            numer_of_features_to_select = self.config["data_processing"]["number_of_features"]
            
            top_10_features = top_feature_df["features"].head(numer_of_features_to_select).values

            top_10_df = df[top_10_features.tolist() + ["booking_status"]]

            logging.info("Feature selection completed succesfully")
            logging.info(f"The top 10 features selected are: {top_10_features}")
            return top_10_df

        except Exception as e:
            logging.error("There is an error while selecting the features")
            raise CustomException(e, sys)
    
    def save_data(self, df, file_path):
        try:
            logging.info('Saving the data in the processed folder')
            df.to_csv(file_path, index=False)
            logging.info("Data saved succesfully")
        except Exception as e:
            logging.error("An error has been occured while saving the data")
            raise CustomException(e, sys)
    
    def process(self):
        try:
            logging.info("Loading the data from RAW directory")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]
            self.save_data(train_df, PROCESSED_TRAIN_PATH)
            self.save_data(test_df, PROCESSED_TEST_PATH)
            logging.info("Data processing completed succesfully")

        except Exception as e:
            logging.error("An error has been occured while saving the data")
            raise CustomException(e, sys)
        
    
if __name__=="__main__":
    try:
        logging.info("Initiating the processing the data")
        processor = DataProcessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
        logging.info("Calling the wrapper method for completing the processing")
        processor.process()
        logging.info("Completed processing of the data")
    except Exception as e:
        logging.error("Issue in the main function data processing")
        raise CustomException(e, sys)