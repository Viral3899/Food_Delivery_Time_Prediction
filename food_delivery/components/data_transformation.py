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