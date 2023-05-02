import os
import sys
import shutil

from food_delivery.exception import CustomException
from food_delivery.logger import logging
from food_delivery.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from food_delivery.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact) -> None:
        """
        This is a constructor function that initializes the model pusher configuration and model
        evaluation artifact.

        :param model_pusher_config: It is an object of the class ModelPusherConfig, which contains the
        configuration settings for the model pusher
        :type model_pusher_config: ModelPusherConfig
        :param model_evaluation_artifact: This parameter is an instance of the class
        ModelEvaluationArtifact, which contains information about the evaluation of a machine learning
        model. It may include metrics such as accuracy, precision, recall, and F1 score, as well as any
        other relevant information about the model's performance
        :type model_evaluation_artifact: ModelEvaluationArtifact
        """
        try:
            logging.info(f"\n\n{'=' * 30}Model Pusher log started.{'=' * 30}\n\n")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def export_model(self) -> ModelPusherArtifact:
        """
        This function exports a trained model to a specified directory and returns a ModelPusherArtifact
        object.
        :return: a `ModelPusherArtifact` object.
        """
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = os.path.join(export_dir, model_file_name)
            logging.info(f'Exporting Model File :[{export_model_file_path}]')
            os.makedirs(export_dir, exist_ok=True)

            shutil.copy(src=evaluated_model_file_path,
                        dst=export_model_file_path)

            logging.info(
                f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")

            return model_pusher_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        This function attempts to export a model and returns a ModelPusherArtifact, but raises a
        CustomException if there is an exception.
        :return: an instance of the `ModelPusherArtifact` class.
        """
        try:
            return self.export_model()
        except Exception as e:
            raise CustomException(e, sys) from e

    def __del__(self):
        logging.info(f"\n\n{'==' * 20}Model Pusher log completed.{'==' * 20}\n\n")
