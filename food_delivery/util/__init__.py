import os
import sys
import yaml
import pandas as pd
import numpy as np
import dill


from food_delivery.exception import CustomException
from food_delivery.logger import logging
from food_delivery.constant import *


def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file and Returns A Content as Dictionary.
    File path : str
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    """
    file_path : str
    schema_file_path : str
    Validate Data Schema with schema File
    return : Pandas DataFrame
    """
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_message = ""

        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column] = dataframe[column].astype(schema[column])
            else:
                error_message = f'{error_message} \nColumn {column} is not in the schema'
        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str,  array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file=file_obj, arr=array)

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def save_object(file_path: str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(file=file_obj, obj=obj)

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    file_path: str
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise CustomException(e, sys)


def transform_time(data,col1,col2):
    """
    The function transforms time data in a specified column of a pandas dataframe and adds 15 minutes to
    another specified column if it is null.
    
    :param data: a pandas DataFrame
    :param col1: The name of the column in the input data that contains time values in the format of
    'HH:MM'
    :param col2: The name of the column in the input data that represents the end time of a time
    interval
    :return: The function `transform_time` is returning a modified version of the input `data` dataframe
    with two columns `col1` and `col2` transformed into datetime format and with `col2` having a
    15-minute time difference from `col1`.
    """
    data['new'] = np.where(data[col1].str.contains('\.'),data[col1],np.nan)
    data['new'] = data['new'].astype(float)
    data['new'] = data['new'] * 86400
    data['new'] =  pd.to_datetime(data['new'],unit='m')
    data[col1] = pd.to_datetime(data[col1],format = '%H:%M',errors='coerce')
    data[col1] = np.where((data['new'].isna()==False),data['new'],data[col1])
    data.drop('new',axis=1,inplace = True)
    data[col2] = data.apply(lambda row: row[col1] + pd.Timedelta(minutes=15) if pd.isnull(row[col2]) else row[col2], axis=1)
    
    return data
