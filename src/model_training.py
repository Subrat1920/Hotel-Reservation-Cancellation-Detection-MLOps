import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
from config.path_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
import mlflow
import dagshub
load_dotenv()

dagshub.init(repo_owner='Subrat1920', repo_name='Hotel-Reservation-Cancellation-Detection-MLOps', mlflow=True)
today = datetime.now()
experiment_name = f"Hotel Reservation Prediction Model Trained on {today.strftime('%d %m %Y')}"
mlflow.set_experiment(experiment_name=experiment_name)


mlflow.set_tracking_uri("https://dagshub.com/Subrat1920/Hotel-Reservation-Cancellation-Detection-MLOps.mlflow")


class ModelTrainer:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params = PARAMS 
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logging.info('Loading training and testing data')
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            x_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            x_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logging.info('Data split into features and target successfully')
            return x_train, y_train, x_test, y_test

        except Exception as e:
            logging.error('Error occurred while loading and splitting data')
            raise CustomException(e, sys)

    def _build_estimator(self, model_obj):
        """
        Accept either:
          - a class (e.g. LogisticRegression) -> instantiate it
          - an instance (e.g. LogisticRegression()) -> use as-is
        """
        try:
            if isinstance(model_obj, type):
                return model_obj()
            return model_obj
        except Exception as e:
            raise

    def hyper_parameter_tuning(self, x_train, y_train):
        try:
            logging.info("Starting hyperparameter tuning")
            best_models = []

            for tup in self.params:
                if len(tup) != 3:
                    raise ValueError("Each PARAMS entry must be (name, model_class_or_instance, param_grid)")

                model_name, model_obj, param_dist = tup
                logging.info(f"Running RandomizedSearchCV for {model_name}")

                estimator = self._build_estimator(model_obj)

                random_search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_dist,
                    n_iter=self.random_search_params.get("n_iter", 10),
                    cv=self.random_search_params.get("cv", 3),
                    n_jobs=self.random_search_params.get("n_jobs", -1),
                    scoring=self.random_search_params.get("scoring", "accuracy"),
                    verbose=self.random_search_params.get("verbose", 0),
                    random_state=self.random_search_params.get("random_state", None),
                )

                random_search.fit(x_train, y_train)

                best_models.append(
                    (model_name, random_search.best_estimator_, random_search.best_params_)
                )

                logging.info(f"Completed tuning for {model_name}")

            return best_models

        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            raise CustomException(e, sys)

    def train_models(self, models_info, x_train, x_test, y_train, y_test):
        try:
            results = []

            for model_name, model, params in models_info:
                logging.info(f"Training model: {model_name}")
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, zero_division=0)
                precision = precision_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                results.append({
                    "Model Name": model_name,
                    "Model": model,
                    "Parameters": params,
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1": f1
                })

            return pd.DataFrame(results)

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

    def find_best_model(self, results_df):
        try:
            if results_df.empty:
                raise ValueError("results_df is empty")
            best_row = results_df.loc[results_df['Accuracy'].idxmax()]
            logging.info(f"Best model: {best_row['Model Name']} with Accuracy={best_row['Accuracy']}")
            return best_row
        except Exception as e:
            logging.error("Error while finding best model")
            raise CustomException(e, sys)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logging.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logging.error("Error saving the model")
            raise CustomException(e, sys)

    def run(self):
        try:
            with mlflow.start_run(run_name=f"Best Model on {today.strftime('%d %m %Y')}"):
                logging.info("=== Starting Training Pipeline ===")

                x_train, y_train, x_test, y_test = self.load_and_split_data()

                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path='datasets')
                else:
                    logging.warning(f"Train path {self.train_path} does not exist - skipping artifact log")

                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path='datasets')
                else:
                    logging.warning(f"Test path {self.test_path} does not exist - skipping artifact log")

                tuned_models = self.hyper_parameter_tuning(x_train, y_train)
                results_df = self.train_models(tuned_models, x_train, x_test, y_train, y_test)

                best_model_info = self.find_best_model(results_df)
                best_model = best_model_info["Model"]

                try:
                    mlflow.log_params(best_model.get_params())
                except Exception:
                    logging.warning("Could not log all params for the selected model")

                mlflow.log_metrics({
                    "Accuracy": float(best_model_info["Accuracy"]),
                    "Precision": float(best_model_info["Precision"]),
                    "Recall": float(best_model_info["Recall"]),
                    "F1": float(best_model_info["F1"])
                })

                # Save model locally then log artifact
                self.save_model(best_model)
                if os.path.exists(self.model_output_path):
                    mlflow.log_artifact(self.model_output_path, artifact_path='models')
                else:
                    logging.warning(f"Model artifact {self.model_output_path} not found to log")

                logging.info("=== Model training completed successfully ===")

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Model Trainer Main Execution Started")
        trainer = ModelTrainer(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
        trainer.run()
        logging.info("Model Training with MLflow Logging Completed Successfully")
    except Exception as e:
        logging.error("Error in Model Trainer Main Execution")
        raise CustomException(e, sys)
