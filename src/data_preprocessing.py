import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data,read_yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        logger.info("DataProcessor initialized successfully")

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing steps")

            logger.info("Dropping the columns")
            df.drop(columns=['Booking_ID'] ,inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying label encoding to categorical columns")
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for code,label in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_)  )}
            
            logger.info("Label mappings are:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing skewness correction using SMOTE")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])

            return df
        
        except Exception as e:
            logger.error(f"Error during data preprocessing {e}")
            raise CustomException("Failed to preprocess data", e)
        
    def balance_data(self,df):
            try:
                logger.info("Handling Imbalanced data")
                X = df.drop('booking_status', axis=1)
                y = df['booking_status']

                smote = SMOTE(random_state=42)
                X_resampled , y_resampled = smote.fit_resample(X, y)

                balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
                balanced_df['booking_status'] = y_resampled
                logger.info("Data balancing completed using SMOTE")

                return balanced_df
            
            except Exception as e:
                logger.error(f"Error during data balancing {e}")
                raise CustomException("Failed to balance data", e)
        
    def select_features(self,df):
            try:
                logger.info("Starting feature selection step")
                X= df.drop('booking_status', axis=1)
                y= df['booking_status']

                model = RandomForestClassifier(random_state=42)
                model.fit(X, y)

                feature_importance = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': feature_importance
                        })
                
                top_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                num_features_to_select = self.config["data_processing"]["no_of_features"]
                top_10_features = top_feature_importance_df["Feature"].head(num_features_to_select).values
                top_10_df = df[top_10_features.tolist() + ['booking_status']]
                logger.info(f"Feature selection completed. Selected features: {top_10_features}")

                return top_10_df

            except Exception as e:
                logger.error(f"Error during feature selection {e}")
                raise CustomException("Failed to select features", e)
        
    def save_data(self,df,path):
        try:
            logger.info(f"Saving processed data to {path}")
            df.to_csv(path, index=False)
            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error while saving data to {path}: {e}")
            raise CustomException("Failed to save processed data", e)
        
    def process(self):
        try:
            logger.info("Starting data processing pipeline")

            logger.info("Loading data from RAW Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            processed_train_df = self.preprocess_data(train_df)
            processed_test_df = self.preprocess_data(test_df)

            balanced_train_df = self.balance_data(processed_train_df)
            

            final_train_df = self.select_features(balanced_train_df)
            final_test_df = processed_test_df[final_train_df.columns]

            self.save_data(final_train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(final_test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
            raise

        finally:
            logger.info("Data processing finished")

if __name__ == "__main__":
    data_processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_processor.process()