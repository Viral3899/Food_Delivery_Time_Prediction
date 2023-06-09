import os
import sys

from food_delivery.exception import CustomException
from food_delivery.util import load_object

import pandas as pd


class DeliveryData:

    def __init__(self,
                 delivery_person_age: float,
                 delivery_person_ratings: float,
                 restaurant_latitude: float,

                 restaurant_longitude: float,

                 delivery_location_latitude: float,

                 delivery_location_longitude: float,
                 order_date : object,
                 time_ordered : object,
                 time_order_picked : object,
                 weather_conditions : object,
                 road_traffic_density  : object,
                 vehicle_condition  : int,
                 type_of_order : object,
                 type_of_vehicle : object,
                 multiple_deliveries :float,
                 festival : object,
                 city : object
                 ):
        try:
            self.delivery_person_age = delivery_person_age
            self.delivery_person_ratings = delivery_person_ratings
            self.restaurant_latitude = restaurant_latitude
            self.restaurant_longitude = restaurant_longitude
            self.delivery_location_latitude = delivery_location_latitude
            self.delivery_location_longitude = delivery_location_longitude
            self.order_date = order_date
            self.time_ordered = time_ordered
            self.time_order_picked = time_order_picked
            self.weather_conditions = weather_conditions
            self.road_traffic_density = road_traffic_density
            self.vehicle_condition = vehicle_condition
            self.type_of_order = type_of_order
            self.type_of_vehicle = type_of_vehicle
            self.multiple_deliveries = multiple_deliveries
            self.festival = festival
            self.city = city
            
        
            
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
                "Delivery_person_Age": [self.delivery_person_age],
                "Delivery_person_Ratings": [self.delivery_person_ratings],
                "Restaurant_latitude": [self.restaurant_latitude],
                "Restaurant_longitude": [self.restaurant_longitude],
                "Delivery_location_latitude": [self.delivery_location_latitude],
                "Delivery_location_longitude": [self.delivery_location_longitude],
                "Order_Date": [self.order_date],
                "Time_Orderd": [self.time_ordered],
                "Time_Order_picked": [self.time_order_picked],
                "Weather_conditions": [self.weather_conditions],
                "Road_traffic_density": [self.road_traffic_density],
                "Type_of_order": [self.type_of_order],
                "Vehicle_condition": [self.vehicle_condition],
                "Type_of_vehicle": [self.type_of_vehicle],
                "multiple_deliveries": [self.multiple_deliveries],
                "Festival": [self.festival],
                "City": [self.city]}
            return input_data
        except Exception as e:
            raise CustomException(e, sys)


class FoodDeliveryTimePredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(
                self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            Delivery_time = model.predict(X)
            return Delivery_time
        except Exception as e:
            raise CustomException(e, sys) from e
