from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *


if __name__ == "__main__":
    ### 1. Data Ingestion

    data_ingetion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingetion.run()

    ### 2. Data Processing
    data_processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_processor.process()

    ### 3. Model Training
    trainer = ModelTraining(
        proessed_train_path=PROCESSED_TRAIN_DATA_PATH,
        processed_test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()