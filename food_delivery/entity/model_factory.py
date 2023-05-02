import os
import sys
import yaml
from collections import namedtuple
from typing import List
import importlib
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error


from food_delivery.util import read_yaml_file
from food_delivery.logger import logging
from food_delivery.exception import CustomException

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail", ["model_serial_number",
                                                               "model",
                                                               "param_grid_search",
                                                               "model_name"
                                                               ])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score"
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score"
                                     ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact", ["model_name",
                                                       "model_object",
                                                       "train_rmse",
                                                       "test_rmse",
                                                       "train_accuracy",
                                                       "test_accuracy",
                                                       "model_accuracy",
                                                       "index_number"
                                                       ])


class ModelFactory:

    def __init__(self, model_config_path: str = None):
        """
        The function reads a config file and initializes a model object.

        :param model_config_path: str = None
        :type model_config_path: str
        """

        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_cv_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_cv_property_data: dict = dict(
                self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.model_initialization_config: dict = dict(
                self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    @staticmethod
    def read_params(config_path: str) -> dict:
        """
        It reads a yaml file and returns a dictionary

        :param config_path: str = "config.yaml"
        :type config_path: str
        :return: a dictionary.
        """
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        """
        It takes an instance of a class and a dictionary of properties and values and updates the
        instance with the values in the dictionary

        :param instance_ref: object
        :type instance_ref: object
        :type property_data: dict
        :return: The instance of the class with the updated values
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception(
                    "property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        """
        It takes a string of the module name and a string of the class name and returns the class object

        :param module_name: The name of the module you want to import
        :type module_name: str
        :param class_name: The name of the class you want to instantiate
        :type class_name: str
        :return: The class reference
        """
        try:
            # load the Module ,IF Module there is not will raise ImportError
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(
                f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6) -> BestModel:
        """
        It takes a list of GridSearchedBestModel objects and returns the best model from the list

        :param grid_searched_best_model_list: List[GridSearchedBestModel]
        :type grid_searched_best_model_list: List[GridSearchedBestModel]
        :param base_accuracy: The minimum accuracy that we want to achieve
        :return: BestModel
        """
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(
                        f'Acceptable Model Found {grid_searched_best_model}')
                    base_accuracy = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model

            if not best_model:
                raise Exception(
                    f'None of the Model has Base Accuracy : {base_accuracy}')
            logging.info(f'best Model {best_model}')

            return best_model

        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail,
                                      input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        The function takes in an initialized model, input feature and output feature and returns a grid
        searched best model

        :param initialized_model: InitializedModelDetail
        :type initialized_model: InitializedModelDetail
        :param input_feature: The input feature dataframe
        :param output_feature: The target variable
        :return: A GridSearchedBestModel object
        """
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_cv_class_name
                                                             )
            grid_search_cv_model = grid_search_cv_ref(estimator=initialized_model.model,
                                                      param_grid=initialized_model.param_grid_search
                                                      )
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv_model,
                                                                   self.grid_search_cv_property_data
                                                                   )
            message = f'{"$$"* 30} f"Training {type(initialized_model.model).__name__} Started." {"$$"*30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{"##"* 30} f"Training {type(initialized_model.model).__name__}" completed {"##"*30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            return grid_searched_best_model

        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function will return a list of model details.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.model_initialization_config.keys():

                model_initialization_config = self.model_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY]
                                                            )
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(
                        model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
        """
        It takes a list of initialized models and returns a list of grid searched best models

        :param initialized_model_list: List[InitializedModelDetail]
        :type initialized_model_list: List[InitializedModelDetail]
        :param input_feature: The input feature is a list of features that are used to predict the
        output feature
        :param output_feature: The target variable
        :return: a list of GridSearchedBestModel objects.
        """

        try:
            self.grid_searched_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_search_best_model = self.initiate_best_parameter_search_for_initialized_model(initialized_model=initialized_model,
                                                                                                   input_feature=input_feature,
                                                                                                   output_feature=output_feature
                                                                                                   )
                self.grid_searched_best_model_list.append(
                    grid_search_best_model)
            return self.grid_searched_best_model_list

        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

    def get_best_model(self, X, y, base_accuracy):
        try:
            logging.info('Started Initializing model from config File')
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f'Initialized Models {initialized_model_list}')
            grid_search_best_model = self.initiate_best_parameter_search_for_initialized_models(initialized_model_list=initialized_model_list,
                                                                                                input_feature=X,
                                                                                                output_feature=y)
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_search_best_model,
                                                                                  base_accuracy=base_accuracy)

        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)


def evaluate_regression_model(model_list: list,
                              X_train: np.ndarray,y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              base_accuracy: float = 0.6) -> MetricInfoArtifact:
    """
    This function evaluates a list of regression models based on their accuracy and root mean squared
    error, and returns the best model that meets certain criteria.
    
    :param model_list: A list of regression models to be evaluated
    :type model_list: list
    :param X_train: A numpy array containing the features of the training dataset
    :type X_train: np.ndarray
    :param y_train: The target variable values for the training dataset
    :type y_train: np.ndarray
    :param X_test: The input features for the testing dataset
    :type X_test: np.ndarray
    :param y_test: y_test is a numpy array containing the true target values for the testing dataset
    :type y_test: np.ndarray
    :param base_accuracy: The minimum acceptable accuracy for a model to be considered as a good model.
    It has a default value of 0.6
    :type base_accuracy: float
    :return: a MetricInfoArtifact object.
    """
    

    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)
            logging.info(
                f'{"+"*20} Started Evaluating model : [{type(model).__name__}] {"+"*20}')

            # getting Prediction fro training And testing Dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating r2 score on Training  and Testing Dataset
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)

            # calculating root Mean Squared Error for Training ANd Testing Dataset
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Calculating Harmonic mean of Training and Testing accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / \
                (train_acc + test_acc)
            diff_train_test_acc = abs(train_acc - test_acc)

            logging.info(f"{'-+'*10} Scores {'-+'*10}")
            logging.info(f"""
                         Train Score   : -->>>> {train_acc}
                         Test Score    : -->>>> {train_acc}
                         Average Score : -->>>> {model_accuracy}
                         """)

            logging.info(f"{'-+'*10} Losses {'-+'*10}")
            logging.info(f"""
                         Train Root Mean Squared Error         : -->>>> {train_rmse}
                         Test Root Mean Squared Error          : -->>>> {test_rmse}
                         Difference between train and test acc : -->>>> {diff_train_test_acc}
                         """)

            if round(model_accuracy,3) >= round(base_accuracy,3) and diff_train_test_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                          model_object=model,
                                                          train_rmse=train_rmse,
                                                          test_rmse=test_rmse,
                                                          train_accuracy=train_acc,
                                                          test_accuracy=test_acc,
                                                          model_accuracy=model_accuracy,
                                                          index_number=index_number
                                                          )
                logging.info(f"Acceptable Model Found {metric_info_artifact}.")

            print(metric_info_artifact)
            index_number += 1
        if metric_info_artifact is None:
            logging.info(
                f"No model Found with Higher accuracy than BAse Accuracy")
        return metric_info_artifact

    except Exception as e:
        logging.info(f'Error Occurred at {CustomException(e,sys)}')
        raise CustomException(e, sys)
