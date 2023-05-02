import os
import numpy as np
import threading
from flask import Flask, render_template, request
import joblib
from food_delivery.pipeline.pipeline import Pipeline
from food_delivery.entity.delivery_time_predictor import DeliveryData, food_deliveryPredictor
from food_delivery.config.configuration import Configuration
from food_delivery.constant import *

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        # Get the form data
        delivery_person_age = float(request.form.get('Delivery_person_Age'))
        delivery_person_ratings = float(request.form.get('Delivery_person_Ratings'))
        restaurant_latitude = float(request.form.get('Restaurant_latitude'))
        restaurant_longitude = float(request.form.get('Restaurant_longitude'))
        delivery_location_latitude = float(request.form.get('Delivery_location_latitude'))
        delivery_location_longitude = float(request.form.get('Delivery_location_longitude'))
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
        food_delivery_data = DeliveryData(longitude=longitude,
                                   latitude=latitude,
                                   food_delivery_median_age=food_delivery_median_age,
                                   total_rooms=total_rooms,
                                   total_bedrooms=total_bedrooms,
                                   population=population,
                                   households=households,
                                   median_income=median_income,
                                   ocean_proximity=ocean_proximity)
        # Return a response
        return 'Form submitted successfully!'
    
    return render_template('form.html')


@app.route('/train',methods = ['GET','POST'])
def training():
    return "Started Training"


@app.route('/predict',methods = ['GET','POST'])
def predict():
    return "Started Predicting"

from flask import Flask, render_template, request

app = Flask(__name__)


    




if __name__=="__main__":
    app.run(debug=True)
    
    
    
    