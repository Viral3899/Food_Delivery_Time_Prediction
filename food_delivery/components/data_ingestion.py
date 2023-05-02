import os
import sys
import requests
from six.moves import urllib
import tarfile
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from food_delivery.logger import logging
from food_delivery.exception import CustomException

from food_delivery.entity.config_entity import DataIngestionConfig
from food_delivery.config.configuration import Configuration
from food_delivery.entity.artifact_entity import DataIngestionArtifact
from food_delivery.constant import *
from food_delivery.components.data_validation import *

from food_delivery.util import *




class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        This is a constructor function that initializes a class object with a data ingestion
        configuration and logs any errors that occur.

        :param data_ingestion_config: The parameter `data_ingestion_config` is an instance of the
        `DataIngestionConfig` class, which contains configuration information for data ingestion. This
        parameter is passed to the constructor of a class that is being initialized
        :type data_ingestion_config: DataIngestionConfig
        """
        try:
            logging.info(
                f"\n\n{'='*20} Data Ingestion log Started {'='*20}\n\n")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def download_food_delivery_data(self,) -> str:
        """
        It downloads the food_delivery data from the remote url and stores it in the local file system
        :return: The file path of the downloaded file.
        """
        try:
            # extraction remote url to download url
            download_url = self.data_ingestion_config.dataset_download_url

            # folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir

            os.makedirs(tgz_download_dir, exist_ok=True)

            food_delivery_file_name = os.path.basename(download_url)

            tgz_file_path = os.path.join(
                tgz_download_dir, food_delivery_file_name)

            logging.info(f"""
             --> Started Downloading......
             --> From URL : [{download_url}]
             --> Into Folder : [{tgz_file_path}]
             """)

            urllib.request.urlretrieve(download_url, tgz_file_path)

            # response = requests.get(download_url)
            # with open(tgz_file_path, 'wb') as f:
            #     f.write(response.content)

            logging.info(
                f"File :[{tgz_file_path}] has been downloaded successfully.")

            return tgz_file_path

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def extract_tgz_file(self, tgz_file_path: str):
        """
        It extracts the contents of a tgz file to a directory

        :param tgz_file_path: The path to the tgz file that you want to extract
        :type tgz_file_path: str
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            logging.info(f"""
            --> Extracting Data from tgz file :  [{tgz_file_path}]
            --> to raw data dir : [{raw_data_dir}]
            """)

            os.makedirs(raw_data_dir, exist_ok=True)

            with tarfile.open(tgz_file_path) as food_delivery_tgz_file_obj:
                food_delivery_tgz_file_obj.extractall(path=raw_data_dir)

            return raw_data_dir

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def modified_split_data_as_train_test_(self,) -> DataIngestionArtifact:
        """
        It reads the data from the raw data directory, splits the data into train and test, and writes
        the train and test data into the ingested train and test directories
        :return: DataIngestionArtifact
        """
        try:

            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            food_delivery_file_path = os.path.join(raw_data_dir, file_name)

            food_delivery_data_frame = pd.read_csv(food_delivery_file_path)

            logging.info(f'Reading file data from [{food_delivery_file_path}]')

            logging.info(f'Splitting Data into Train Test')
            start_train_set = None
            start_test_set = None
            X = food_delivery_data_frame.drop('Time_taken (min)', axis=1)

            y = food_delivery_data_frame['Time_taken (min)']
            start_train_set, start_test_set, start_train_set['Time_taken (min)'], start_test_set['Time_taken (min)'] = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            start_train_set['Order_Date'] = pd.to_datetime(
                start_train_set['Order_Date'])
            
            
            
            
            start_train_set = transform_time(start_train_set,'Time_Orderd','Time_Order_picked')
            start_train_set = transform_time(start_train_set,'Time_Order_picked','Time_Orderd')
            start_test_set = transform_time(start_test_set,'Time_Orderd','Time_Order_picked')
            start_test_set = transform_time(start_test_set,'Time_Order_picked','Time_Orderd')
                       
            # Split the mixed values into separate columns for hours and minutes
           
            train_file_path = os.path.join(
                self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(
                self.data_ingestion_config.ingested_test_dir, file_name)

            if start_train_set is not None:
                os.makedirs(
                    self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(
                    f'Exporting training Dataset into [{train_file_path}]')
                start_train_set.to_csv(train_file_path, index=False)

            if start_test_set is not None:
                os.makedirs(
                    self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(
                    f'Exporting testing Dataset into [{test_file_path}]')
                start_test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data Ingestion Completed Successefully")

            logging.info(
                f'Data Ingestion Artifact : [{data_ingestion_artifact}]')

            return data_ingestion_artifact

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self,) -> DataIngestionArtifact:
        """
        It downloads the food_delivery data, extracts the tgz file, and splits the data into train and test
        sets
        :return: DataIngestionArtifact
        """
        try:
            tgz_file_path = self.download_food_delivery_data()

            self.extract_tgz_file(tgz_file_path=tgz_file_path)

            return self.modified_split_data_as_train_test_()

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)


if __name__ == '__main__':
    a = DataIngestion(Configuration(
        config_file_path='D:\End_to_End_Project\Food_delivery\config\config.yaml').get_data_ingestion_config())
    data_ingestion_artifact =a.initiate_data_ingestion()
    
    data_validation = DataValidation(Configuration(
        config_file_path='D:\End_to_End_Project\Food_delivery\config\config.yaml').get_data_validation_config(
            ), data_ingestion_artifact=data_ingestion_artifact)
    data_validation.initiate_data_validation()
    # print(data_validation)
