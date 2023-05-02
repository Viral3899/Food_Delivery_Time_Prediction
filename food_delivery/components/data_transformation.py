import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


from food_delivery.constant import *
from food_delivery.exception import CustomException
from food_delivery.logger import logging
from food_delivery.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from food_delivery.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig
from food_delivery.config.configuration import Configuration
from food_delivery.util import read_yaml_file, load_data, save_numpy_array_data, save_object

class FeatureGenerator(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_order_time_features=True, columns=None):
        self.add_order_time_features = add_order_time_features
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.add_order_time_features:
            X['Order_Date'] = pd.to_datetime(X['Order_Date'])
            X['Month'] = X['Order_Date'].dt.month
            X['Day'] = X['Order_Date'].dt.day
            X['Year'] = X['Order_Date'].dt.year
            X['Hour_Orderd'] = pd.to_datetime(X['Time_Orderd']).dt.hour
            X['Minute_Orderd'] = pd.to_datetime(X['Time_Orderd']).dt.minute
            X['Hour_Picked'] = pd.to_datetime(X['Time_Order_picked']).dt.hour
            X['Minute_Picked'] = pd.to_datetime(X['Time_Order_picked']).dt.minute
        
        # Calculate the distance between the restaurant and delivery location using geopy
        X['distance'] = X.apply(lambda row: geodesic((row['Restaurant_latitude'], row['Restaurant_longitude']), 
                                                     (row['Delivery_location_latitude'], row['Delivery_location_longitude'])).km, axis=1)
        
        if self.columns is not None:
            X = X[self.columns]
        
        return X
class FeatureTransformer:
    def __init__(self, feature_gen_pipe, preprocessing_pipe) -> None:
        """
        This function takes in a preprocessing object and a trained model object and assigns them to the
        class variables preprocessing_object and trained_model_object.

        :param preprocessing_object: This is the object of the class Preprocessing
        :param trained_model_object: This is the object of the class that contains the trained model
        """
        self.feature_gen_pipe = feature_gen_pipe
        self.preprocessing_pipe = preprocessing_pipe
        # self.fit = fit

    def fit_transform(self, X):

        try:
       
            transformed_features = self.feature_gen_pipe.transform(X)
            # print(transformed_features.columns)
            
            
            transomed_data = self.preprocessing_pipe.fit_transform(
                transformed_features)
            return transomed_data
                
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)
        
    def transform(self, X):

        try:
       
            transformed_features = self.feature_gen_pipe.transform(X)
            transomed_data = self.preprocessing_pipe.transform(
                transformed_features)
            return transomed_data
                
        except Exception as e:
            logging.info(f'Error Occurred at {CustomException(e,sys)}')
            raise CustomException(e, sys)

class DataTransformation:
    
    def __init__(self,data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(
                f"\n\n{'='*20} Data Transformation log Started {'='*20}\n\n")

            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)
        
    
    def get_preprocessor(self) -> ColumnTransformer:
        # Define the columns that will be transformed by the FeatureGenerator
        try:
            
            num_col = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 
                            'Delivery_location_latitude', 'Delivery_location_longitude', 
                            'Day', 'Month', 'Year', 'Hour_Orderd', 'Minute_Orderd', 'Hour_Picked', 'Minute_Picked', 'distance']
            cat_col = ['Road_traffic_density','Weather_conditions', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']

            # Define the pipelines for each column type

            num_pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Define the preprocessing step to apply each pipeline to its respective columns
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_col),
                ('cat', cat_pipeline,cat_col )
            ])

            return preprocessor

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

           
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        It takes the training and testing dataframe, splits the input and target feature, applies the
        preprocessing object on the input feature, and saves the transformed training and testing array
        and preprocessing object.
        :return: DataTransformationArtifact
        """
        try:
            logging.info('Obtaining preprocessing Object')
            date_cols = ['Order_Date', 'Time_Orderd', 'Time_Order_picked']
            preprocessing_obj = self.get_preprocessor()
            fet_gen_pipeline = FeatureGenerator(add_order_time_features=True)
            preprocessing_obj = FeatureTransformer(feature_gen_pipe=fet_gen_pipeline,preprocessing_pipe=preprocessing_obj)
            
            logging.info('Getting Train and Test File Path')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(
                'Loading Training And Testing File As Panda DataFrame')

            train_df = load_data(file_path=train_file_path,
                                schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path,
                                schema_file_path=schema_file_path)

            dataset_schema = read_yaml_file(schema_file_path)
            target_column_name = dataset_schema[TARGET_COLUMN_KEY]

            logging.info(
                f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # fet_gen_pipeline = FeatureGenerator()
            # input_feature_train_df = fet_gen_pipeline.transform(input_feature_train_df)
            # input_feature_test_df = fet_gen_pipeline.transform(input_feature_test_df)
            

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                            np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,
                            np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(
                train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(
                test_file_path).replace(".csv", ".npz")

            transformed_train_file_path = os.path.join(
                transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(
                transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")

            save_numpy_array_data(
                file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(
                file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,
                        obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                    message="Data transformation successful.",
                                                                    transformed_train_file_path=transformed_train_file_path,
                                                                    transformed_test_file_path=transformed_test_file_path,
                                                                    preprocessed_object_file_path=preprocessing_obj_file_path

                                                                    )
            logging.info(
                f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)




