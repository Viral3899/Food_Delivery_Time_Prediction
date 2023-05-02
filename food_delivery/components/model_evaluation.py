import os
import sys
import pandas as pd
import numpy as np


from food_delivery.config.configuration import Configuration
from food_delivery.exception import CustomException
from food_delivery.logger import logging
from food_delivery.constant import *
from food_delivery.util import read_yaml_file, load_data, load_numpy_array_data, load_object, write_yaml_file
from food_delivery.entity.config_entity import ModelEvaluationConfig
from food_delivery.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from food_delivery.entity.model_factory import evaluate_regression_model


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact) -> None:
        """
        This is the constructor function for a class that takes in several artifacts and a configuration
        object and initializes them as class attributes.

        :param model_evaluation_config: An object of the class ModelEvaluationConfig, which contains the
        configuration settings for model evaluation
        :type model_evaluation_config: ModelEvaluationConfig
        :param data_ingestion_artifact: It is an object of the class DataIngestionArtifact, which
        contains information and artifacts related to the data ingestion process of the machine learning
        model. This could include data sources, data preprocessing steps, and data cleaning techniques
        :type data_ingestion_artifact: DataIngestionArtifact
        :param data_validation_artifact: DataValidationArtifact is an object that contains information
        about the data validation process for the model. It may include details such as the validation
        metrics used, the validation dataset, and any preprocessing steps applied to the data before
        validation. This parameter is passed to the constructor of a class that is responsible for
        evaluating the
        :type data_validation_artifact: DataValidationArtifact
        :param model_trainer_artifact: This parameter is an instance of the class ModelTrainerArtifact,
        which contains the artifacts generated during the model training process. These artifacts may
        include the trained model, feature transformers, and other necessary objects for making
        predictions. The ModelTrainerArtifact is used in the model evaluation process to load the
        trained model
        :type model_trainer_artifact: ModelTrainerArtifact
        """

        try:
            logging.info(
                f"\n\n{'='*20} Model Evaluation log Started {'='*20}\n\n")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_best_model(self,):
        """
        This function retrieves the best model from a saved model evaluation file.
        :return: a trained machine learning model that has been saved as a file, or None if the file
        does not exist or if the best model has not been evaluated yet. If an error occurs, a
        CustomException is raised.
        """
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)

                return model
            model_evaluation_file_content = read_yaml_file(
                file_path=model_evaluation_file_path)

            model_evaluation_file_content = dict(
            ) if model_evaluation_file_content is None else model_evaluation_file_content

            if BEST_MODEL_KEY not in model_evaluation_file_content:
                return model

            model = load_object(
                file_path=model_evaluation_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        """
        This function updates a model evaluation report with the results of a new model evaluation.

        :param model_evaluation_artifact: The parameter `model_evaluation_artifact` is an object of the
        class `ModelEvaluationArtifact` which contains information about the evaluated model such as the
        path to the model file
        :type model_evaluation_artifact: ModelEvaluationArtifact
        """
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f'Previous Eval Results : {model_eval_content}')

            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }

            if previous_best_model is not None:
                model_history = {
                    self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f'update Eval Results : {model_eval_content}')
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function evaluates a trained model and compares it with an existing model, and returns a
        ModelEvaluationArtifact indicating whether the trained model is accepted or not.
        :return: a ModelEvaluationArtifact object.
        """
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(
                file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info('Loading data for model evaluation')
            train_dataframe = load_data(file_path=train_file_path,
                                        schema_file_path=schema_file_path
                                        )
            test_dataframe = load_data(file_path=test_file_path,
                                       schema_file_path=schema_file_path
                                       )

            schema_content = read_yaml_file(file_path=schema_file_path)

            target_column = schema_content[TARGET_COLUMN_KEY]
            logging.info("Splitting Data into target and features")
            train_target = np.array(train_dataframe[target_column])
            test_target = np.array(test_dataframe[target_column])

            train_dataframe = train_dataframe.drop(target_column, axis=1)
            test_dataframe = test_dataframe.drop(target_column, axis=1)

            logging.info("All set for Evaluation")

            model = self.get_best_model()

            if model is None:
                logging.info(
                    "No Existing Model Found Hence Accepting Trained Model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(
                    f"Model accepted model evaluation artifact{model_evaluation_artifact} created successfully")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                             X_train=train_dataframe,
                                                             y_train=train_target,
                                                             X_test=test_dataframe,
                                                             y_test=test_target,
                                                             base_accuracy=self.model_trainer_artifact.model_accuracy
                                                             )
            print(metric_info_artifact)
            logging.info(
                f'model Evaluation Completed model metric info artifact : {metric_info_artifact}')

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)

                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True
                                                                    )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(
                    f'Model Accepted , Model Evaluation Artifact {model_evaluation_artifact}')

            else:
                logging.info(
                    'Trained Model is not Better Than Existing Model hence Not Accepting Trained Model')
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False
                                                                    )
            return model_evaluation_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def __del__(self):
        logging.info(
            f"\n\n{'='*20} Model Evaluation Log Completed {'='*20} \n\n")
