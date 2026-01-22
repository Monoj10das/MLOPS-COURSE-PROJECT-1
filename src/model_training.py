import os
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import uniform, randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, proessed_train_path , processed_test_path,model_output_path):
        self.train_path = proessed_train_path
        self.test_path = processed_test_path
        self.model_output_path = model_output_path

        self.param_distributions = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'], axis=1)
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'], axis=1)
            y_test = test_df['booking_status']

            logger.info("Data splited successfully for model training")
            
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while loading data : {e}")
            raise CustomException(f"Failed to load data " ,e)
        
    def train_lgbm(self,X_train, y_train):
        try:
            logger.info("Initializing our model")
            lgbm = lgb.LGBMClassifier()

            logger.info("Starting our hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.param_distributions,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                n_jobs=self.random_search_params['n_jobs'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            logger.info("Starting our Hyperparameter tuning fit process")

            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters found: {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model : {e}")
            raise CustomException(f"Failed to train model " ,e)
        
    def evaluate_model(self, model, X_test, y_test):

        try:
            logger.info("Starting model evaluation")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Model evaluation completed with Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1-Score: {f1}")

            return {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            }

        except Exception as e:
            logger.error(f"Error while evaluating model : {e}")
            raise CustomException(f"Failed to evaluate model " ,e)
        

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error while saving model : {e}")
            raise CustomException(f"Failed to save model " ,e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")
                logger.info("Starting our MLflow logging")
                logger.info("Logging the training and testing data sets to MLflow")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lbgm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lbgm_model, X_test, y_test)
                self.save_model(best_lbgm_model)

                logger.info("Logging the model into MLflow")
                mlflow.log_artifact(self.model_output_path, artifact_path="model")

                logger.info("Logging parameters and metrics to MLflow")
                mlflow.log_params(best_lbgm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training process completed successfully")
        
        except Exception as e:
            logger.error(f"Error in model training process : {e}")
            raise CustomException(f"Failed in model training process " ,e)
        
if __name__ == "__main__":
    trainer = ModelTraining(
        proessed_train_path=PROCESSED_TRAIN_DATA_PATH,
        processed_test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()
        