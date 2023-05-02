import os
import sys
import pandas as pd
from typing import List


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from food_delivery.logger import logging
from food_delivery.exception import CustomException
from food_delivery.util import read_yaml_file, save_object, load_numpy_array_data, load_object
from food_delivery.config.configuration import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig
from food_delivery.entity.config_entity import DataValidationConfig, DataIngestionConfig, DataTransformationConfig
from food_delivery.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from food_delivery.entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel, evaluate_regression_model


class food_deliveryEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object) -> None:
        """
        This function takes in a preprocessing object and a trained model object and assigns them to the
        class variables preprocessing_object and trained_model_object.

        :param preprocessing_object: This is the object of the class Preprocessing
        :param trained_model_object: This is the object of the class that contains the trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):

        try:    
            transformed_features = self.preprocessing_object.transform(X)
            predictions = self.trained_model_object.predict(
                transformed_features)
            return predictions
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact) -> None:
        """
        The function takes in two arguments, a ModelTrainerConfig object and a DataTransformationArtifact
        object. 

        The ModelTrainerConfig object is a class that contains the configuration parameters for the model
        trainer. 

        The DataTransformationArtifact object is a class that contains the data transformation artifact. 

        The function then initializes the model trainer config and data transformation artifact. 

        The function also logs the start of the model trainer log. 

        The function then raises an exception if an error occurs.

        :param model_trainer_config: This is the configuration object that contains all the parameters
        that are required to train the model
        :type model_trainer_config: ModelTrainerConfig
        :param data_transformation_artifact: This is the artifact that is created by the
        DataTransformation class
        :type data_transformation_artifact: DataTransformationArtifact
        """
        try:
            logging.info(
                f"\n\n{'='*20} Model Trainer log Started {'='*20}\n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        It takes the transformed training and testing data, splits it into input and target columns,
        extracts the model config file, finds the best model using the model factory class, initializes
        the model selection operation, extracts the trained model list, evaluates all trained models on
        the training and testing datasets, finds the best model on both the training and testing
        datasets, saves the model, and returns the model trainer artifact
        :return: ModelTrainerArtifact
        """
        try:
            logging.info('loading Transformed Training and Testing Data...')

            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(transformed_train_file_path)

            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(transformed_test_file_path)

            logging.info(
                f'Splitting Training and Testing Data... into Input and Target Column')

            X_train, y_train, X_test, y_test = train_array[:, :-
                                                           1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]

            logging.info('Extracting Model Config File')
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info('Start finding Best Model using Model Factory Class')
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected(Base) accuracy Should be {base_accuracy}")

            model_factory = ModelFactory(
                model_config_path=model_config_file_path)

            logging.info(f'Initializing Model Selection operation')
            best_model = model_factory.get_best_model(
                X=X_train, y=y_train, base_accuracy=0.6)

            logging.info(f"Best model found on training dataset: {best_model}")

            logging.info(f"Extracting trained model list.")

            grid_searched_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list = [
                model.best_model for model in grid_searched_model_list]

            logging.info(
                f'Evaluation all trained model on Training and Testing both Dataset')
            metric_info: MetricInfoArtifact = evaluate_regression_model(model_list=model_list,
                                                                        X_train=X_train, y_train=y_train,
                                                                        X_test=X_test, y_test=y_test,
                                                                        base_accuracy=base_accuracy
                                                                        )

            logging.info(
                f"Best found model on both training and testing dataset.")

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            food_delivery_model = food_deliveryEstimatorModel(preprocessing_object=preprocessing_obj,
                                                  trained_model_object=model_object
                                                  )
            logging.info(f"Saving Model at path : {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=food_delivery_model)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                          message='Model Trained Successfully',
                                                          trained_model_file_path=trained_model_file_path,
                                                          train_rmse=metric_info.train_rmse,
                                                          test_rmse=metric_info.test_rmse,
                                                          train_accuracy=metric_info.train_accuracy,
                                                          test_accuracy=metric_info.test_accuracy,
                                                          model_accuracy=metric_info.model_accuracy
                                                          )
            logging.info(f'Model Trainer Artifact: {model_trainer_artifact}')
            return model_trainer_artifact
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    def __del__(self):
        logging.info(
            f"\n\n{'='*20} Model Trainer Log Completed {'='*20} \n\n")
