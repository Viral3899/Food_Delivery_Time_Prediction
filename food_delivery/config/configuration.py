
import sys
import os

from food_delivery.exception import CustomException
from food_delivery.logger import logging
from food_delivery.constant import *

from food_delivery.util import read_yaml_file
from food_delivery.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig,\
    ModelEvaluationConfig, ModelPusherConfig, ModelTrainerConfig, TrainingPipelineConfig


class Configuration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH,
                 current_time_stamp=get_current_time_stamp()) -> None:

        try:
            self.config_info = read_yaml_file(file_path=config_file_path)

            self.training_pipeline_config = self.get_training_pipeline_config()

            self.time_stamp = current_time_stamp

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """
        It takes a config file, and returns a TrainingPipelineConfig object
        :return: The training pipeline config is being returned.
        """

        try:

            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]

            artifact_dir = os.path.join(
                ROOT_DIR, training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )

            training_pipeline_config = TrainingPipelineConfig(
                artifact_dir=artifact_dir)

            logging.info(
                f"Training Pipeline Config: {training_pipeline_config}")

            return training_pipeline_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        It takes the config_info dictionary and returns a DataIngestionConfig object
        :return: DataIngestionConfig object
        """

        try:
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]

            dataset_download_url = data_ingestion_config_info[DATA_INGESTION_DOWNLOAD_URL_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_ingestion_artifact_dir = os.path.join(
                artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR_NAME,
                self.time_stamp
            )

            raw_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
            )

            ingested_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_DIR_KEY]
            )

            tgz_download_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_config_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY]
            )

            ingested_train_dir = os.path.join(
                ingested_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_DIR_KEY]
            )

            ingested_test_dir = os.path.join(
                ingested_dir,
                data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_DIR_KEY]
            )

            data_ingestion_config = DataIngestionConfig(
                dataset_download_url=dataset_download_url,
                tgz_download_dir=tgz_download_dir,
                raw_data_dir=raw_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir
            )

            logging.info(f"Data Ingestion Config: {data_ingestion_config}")

            return data_ingestion_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        It takes the config_info dictionary from the config file and returns a DataValidationConfig
        object
        :return: DataValidationConfig
        """
        try:
            data_validation_config_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_artifact_dir = os.path.join(
                artifact_dir, DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.time_stamp
            )

            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_config_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            )

            report_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )

            report_page_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_config_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path
            )

            return data_validation_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        It takes the config file and returns a DataTransformationConfig object
        :return: DataTransformationConfig
        """
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transformation_artifact_dir = os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR_KEY,
                self.time_stamp
            )

            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            add_order_time_features = data_transformation_config_info[
                DATA_TRANSFORMATION_ADD_ORDER_TIME_FEATURES]

            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )

            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY]
            )

            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY]

            )

            data_transformation_config = DataTransformationConfig(
                add_order_time_features=add_order_time_features,
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir
            )

            logging.info(
                f"Data transformation config: {data_transformation_config}")
            return data_transformation_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        It takes the config file and returns the model trainer config object
        :return: The model trainer config is being returned.
        """
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_trainer_artifact_dir = os.path.join(artifact_dir,
                                                      MODEL_TRAINER_ARTIFACT_DIR,
                                                      self.time_stamp
                                                      )

            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
                                                   model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                                                   model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
                                                   )

            model_config_file_path = os.path.join(model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                                                  model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
                                                  )

            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]

            model_trainer_config = ModelTrainerConfig(trained_model_file_path=trained_model_file_path,
                                                      base_accuracy=base_accuracy,
                                                      model_config_file_path=model_config_file_path)

            logging.info(f"Model Trainer Config {model_trainer_config}")

            return model_trainer_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        This function retrieves the model evaluation configuration from a specified file path and
        returns it as a ModelEvaluationConfig object.
        :return: an instance of the `ModelEvaluationConfig` class with the `model_evaluation_file_path`
        and `time_stamp` attributes set based on the values obtained from the `config_info` dictionary
        and `training_pipeline_config` object. The function also logs the returned
        `ModelEvaluationConfig` instance.
        """

        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR, )

            model_evaluation_file_path = os.path.join(artifact_dir,
                                                      model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                             time_stamp=self.time_stamp)

            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_model_pusher_config(self) -> ModelPusherConfig:
        """
        This function returns a ModelPusherConfig object with an export directory path based on the time
        stamp and configuration information.
        :return: an instance of the `ModelPusherConfig` class with the `export_dir_path` attribute set
        to a directory path created using the `ROOT_DIR` constant, `MODEL_PUSHER_MODEL_EXPORT_DIR_KEY`
        key from the `config_info` dictionary, and the `time_stamp` attribute of the class instance. If
        an exception occurs, the function raises a `food_delivery
        """
        try:
            time_stamp = str(self.time_stamp).replace('-', '')
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(
                ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY], time_stamp)
            model_pusher_config = ModelPusherConfig(
                export_dir_path=export_dir_path)

            logging.info(f'Model Pusher config {model_pusher_config}')
            return model_pusher_config
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)
