import os
import numpy as np
import threading
from geopy.distance import distance

from flask import Flask, render_template, request
import joblib
from food_delivery.pipeline.pipeline import Pipeline
from food_delivery.entity.delivery_time_predictor import DeliveryData, FoodDeliveryTimePredictor
from food_delivery.config.configuration import Configuration
from food_delivery.constant import *

OOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "fod_delivery"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form data
        delivery_person_age = float(request.form.get('Delivery_person_Age'))
        delivery_person_ratings = float(
            request.form.get('Delivery_person_Ratings'))
        restaurant_latitude = float(request.form.get('Restaurant_latitude'))
        restaurant_longitude = float(request.form.get('Restaurant_longitude'))
        delivery_location_latitude = float(
            request.form.get('Delivery_location_latitude'))
        delivery_location_longitude = float(
            request.form.get('Delivery_location_longitude'))
        order_date = request.form.get('Order_Date')
        time_ordered = request.form.get('Time_Orderd')
        time_order_picked = request.form.get('Time_Order_picked')
        weather_conditions = request.form.get('Weather_conditions')
        road_traffic_density = request.form.get('Road_traffic_density')
        vehicle_condition = int(request.form.get('Vehicle_condition'))
        type_of_order = request.form.get('Type_of_order')
        type_of_vehicle = request.form.get('Type_of_vehicle')
        multiple_deliveries = float(request.form.get('multiple_deliveries'))
        festival = request.form.get('Festival')
        city = request.form.get('City')

        # Perform any required processing on the form data here
        food_delivery_data = DeliveryData(
            delivery_person_age=delivery_person_age,
            delivery_person_ratings=delivery_person_ratings,
            restaurant_latitude=restaurant_latitude,
            restaurant_longitude=restaurant_longitude,
            delivery_location_latitude=delivery_location_latitude,
            delivery_location_longitude=delivery_location_longitude,
            order_date=order_date,
            time_ordered=time_ordered,
            time_order_picked=time_order_picked,
            weather_conditions=weather_conditions,
            road_traffic_density=road_traffic_density,
            vehicle_condition=vehicle_condition,
            type_of_order=type_of_order,
            type_of_vehicle=type_of_vehicle,
            multiple_deliveries=multiple_deliveries,
            festival=festival,
            city=city
        )
        from geopy.distance import distance

        distance = round(distance((restaurant_latitude, restaurant_longitude),
                         (delivery_location_latitude, delivery_location_longitude)).km, 2)
        delivery_df = food_delivery_data.get_food_delivery_input_data_frame()
        time_predictor = FoodDeliveryTimePredictor(model_dir=MODEL_DIR)
        prediction = time_predictor.predict(X=delivery_df)
        # Return a response
        return render_template('result.html',
                               delivery_person_age=delivery_person_age,
                               delivery_person_ratings=delivery_person_ratings,
                               restaurant_latitude=restaurant_latitude,
                               restaurant_longitude=restaurant_longitude,
                               delivery_location_latitude=delivery_location_latitude,
                               delivery_location_longitude=delivery_location_longitude,
                               order_date=order_date,
                               time_ordered=time_ordered,
                               time_order_picked=time_order_picked,
                               weather_conditions=weather_conditions,
                               road_traffic_density=road_traffic_density,
                               vehicle_condition=vehicle_condition,
                               type_of_order=type_of_order,
                               type_of_vehicle=type_of_vehicle,
                               multiple_deliveries=multiple_deliveries,
                               festival=festival,
                               city=city,
                               distance=round(distance),
                               prediction=round(prediction[0])
                               )

    return render_template('form.html')


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    message = ""
    pipeline = Pipeline(config=Configuration(
        current_time_stamp=get_current_time_stamp()))
    if Pipeline.experiment.running_status == False and Pipeline.experiment.stop_time != np.nan:
        message = "Training is completed."
    if request.method == 'POST':
        if not Pipeline.experiment.running_status:
            message = "Re-Training started."
            pipeline.start()
        else:
            message = "Training is already in progress."

    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('retrain.html', context=context)


if __name__ == "__main__":
    app.run(debug=True)
