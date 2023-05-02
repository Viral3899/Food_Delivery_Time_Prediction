import os
import sys

from food_delivery.exception import CustomException
from food_delivery.util import load_object

import pandas as pd


class DeliveryData:

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 food_delivery_median_age: float,
                 total_rooms: float,
                 total_bedrooms: float,
                 population: float,
                 households: float,
                 median_income: float,
                 ocean_proximity: str,
                 median_house_value: float = None
                 ):
        try:
            self.longitude = longitude
            self.latitude = latitude
            self.food_delivery_median_age = food_delivery_median_age
            self.total_rooms = total_rooms
            self.total_bedrooms = total_bedrooms
            self.population = population
            self.households = households
            self.median_income = median_income
            self.ocean_proximity = ocean_proximity
            self.median_house_value = median_house_value
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_food_delivery_input_data_frame(self):

        try:
            food_delivery_input_dict = self.get_food_delivery_data_as_dict()
            return pd.DataFrame(food_delivery_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_food_delivery_data_as_dict(self):
        try:
            input_data = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "food_delivery_median_age": [self.food_delivery_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity]}
            return input_data
        except Exception as e:
            raise CustomException(e, sys)


class food_deliveryPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise CustomException(e, sys) from e