import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
import json
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


from food_delivery.logger import logging
from food_delivery.exception import CustomException
from food_delivery.entity.config_entity import DataValidationConfig
from food_delivery.util import read_yaml_file
from food_delivery.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from food_delivery.config.configuration import Configuration


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        """
        This is a constructor function that initializes the data validation configuration and data
        ingestion artifact, and logs any errors that occur.

        :param data_validation_config: This parameter is of type DataValidationConfig and contains the
        configuration settings for data validation
        :type data_validation_config: DataValidationConfig
        :param data_ingestion_artifact: DataIngestionArtifact is an object that contains information
        about the data that has been ingested, such as the file path, file format, and any other
        relevant metadata
        :type data_ingestion_artifact: DataIngestionArtifact
        """
        try:
            logging.info(
                f"\n\n{'='*20} Data Validation log Started {'='*20}\n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_train_and_test_df(self):
        """
        It reads the train and test data from the data ingestion artifact and returns the dataframes.
        :return: The train_df and test_df are being returned.
        """
        try:
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def is_train_test_file_exists(self) -> bool:
        """
        It checks if the training and testing files are present in the specified path
        :return: The return value is a boolean value.
        """
        try:
            logging.info(
                'Checking if Training And Testing File is Available!!!!?')
            is_train_file_exists = False
            is_test_file_exists = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            logging.info(
                f'Training File Available Status [is_train_file_exists = {is_train_file_exists}]')

            is_test_file_exists = os.path.exists(test_file_path)
            logging.info(
                f'Testing File Available Status [is_test_file_exists = {is_test_file_exists}]')

            is_available = is_train_file_exists and is_test_file_exists

            logging.info(
                f'Training and Testing both File Available Status [is_Available = {is_available}]')

            if not is_available:
                message = f'Training File : [{train_file_path}] or\n Testing File : [{test_file_path}] is not present.'
                logging.info(message)
                raise Exception(message)

            return is_available

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def validate_dataset_schema(self) -> bool:
        """
        It checks if the number of columns in the schema file matches the number of columns in the train
        and test dataframes, if the target column is present in both the dataframes, if the datatypes of
        the columns in the schema file match the datatypes of the columns in the train and test
        dataframes, and if the domain values of the categorical columns in the schema file match the
        domain values of the categorical columns in the train and test dataframes
        :return: A boolean value
        """
        try:
            is_schema_validated = False

            schema_file_path = self.data_validation_config.schema_file_path
            schema_config = read_yaml_file(file_path=schema_file_path)

            train_df, test_df = self.get_train_and_test_df()
            train_df_columns = train_df.columns

            test_df_columns = test_df.columns

            logging.info('Validating Number of Columns')
            number_of_cols_validate = False
            if (len((schema_config['columns'].keys())) == (len(train_df_columns))) & (len((schema_config['columns'].keys())) == (len(test_df_columns))):
                number_of_cols_validate = True
                logging.info('Number of cols are Validated')
            else:
                raise Exception('Number of cols are not matching')

            logging.info(f'Checking weather target Columns is Available]')
            is_target_column_available = False
            if (schema_config['target_column'] in train_df_columns) and (schema_config['target_column'] in test_df_columns):
                is_target_column_available = True
                logging.info(
                    f'Target Column is Available [{schema_config["target_column"]}]')
            else:
                raise Exception('target col is not available')

            logging.info(
                f'Validating Datatype of Columns with Given Schema at [{schema_file_path}]')
            data_type_of_cols_validate = False
            if (Counter(schema_config['numerical_columns'])) == (Counter([col for col in train_df.columns if train_df[col].dtype != 'O' and col != schema_config['target_column']])) \
                    and (Counter(schema_config['numerical_columns'])) == (Counter([col for col in test_df.columns if test_df[col].dtype != 'O' and col != schema_config['target_column']])) \
                    and (Counter(schema_config['categorical_columns'])) == (Counter([col for col in train_df.columns if test_df[col].dtype == 'O'])) \
                    and (Counter(schema_config['categorical_columns'])) == (Counter([col for col in test_df.columns if test_df[col].dtype == 'O'])):
                data_type_of_cols_validate = True
                logging.info(
                    'DataType of Columns are passed Successfully for both Data')

            else:
                raise Exception('DataType of Columns is not matching')

            logging.info(
                f'Validating Domain Values of Columns with Given Schema at [{schema_file_path}]')
            is_domain_value_validate = False
            if (Counter((schema_config['domain_value']['City']) + [np.nan]) == Counter(train_df['City'].unique()) == Counter(test_df['City'].unique())) and \
                (Counter(schema_config['domain_value']['Weather_conditions'] + [np.nan])) == Counter(train_df['Weather_conditions'].unique()) == Counter(test_df['Weather_conditions'].unique()) and \
                (Counter((schema_config['domain_value']['Road_traffic_density']) + [np.nan]) == Counter(train_df['Road_traffic_density'].unique()) == Counter(test_df['Road_traffic_density'].unique())) and \
                    (Counter((schema_config['domain_value']['Type_of_order'])) == Counter(train_df['Type_of_order'].unique()) == Counter(test_df['Type_of_order'].unique())) and \
                    (Counter((schema_config['domain_value']['Type_of_vehicle'])) == Counter(train_df['Type_of_vehicle'].unique()) == Counter(test_df['Type_of_vehicle'].unique())):
                is_domain_value_validate = True
                logging.info(
                    f'Domain values of Columns [{list(schema_config["domain_value"].keys())}] are passed Successfully for both Data')
            else:
                raise Exception('domain value prob')

            is_schema_validated = number_of_cols_validate and is_domain_value_validate and is_target_column_available and data_type_of_cols_validate

            if not is_schema_validated:
                message = f"Either Number of Columns or Datatype of Columns or Domain VAlues of Column  is Not Matched "
                logging.info(message)
                raise Exception(message)

            return is_schema_validated

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def get_and_save_data_drift_report(self):
        """
        It takes the train and test dataframes, calculates the data drift report and saves it to a file
        :return: The report is being returned.
        """
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df, test_df = self.get_train_and_test_df()
            profile.calculate(train_df, test_df)
            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(self.data_validation_config.report_file_path, "w") as report_file:
                json.dump(report, report_file)

            return report

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def save_data_drift_report_page(self):
        """
        It takes a dataframe, splits it into train and test, and then saves the report page to a file
        """
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)

            dashboard.save(self.data_validation_config.report_page_file_path)

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def is_data_drift_found(self) -> bool:
        """
        It checks if there is a data drift in the model and if there is, it saves the data drift report
        page
        """
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()

            return True
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        It checks if the train and test files exist, validates the schema of the dataset, and checks if
        there is any data drift
        :return: DataValidationArtifact
        """
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(schema_file_path=self.data_validation_config.schema_file_path,
                                                              report_file_path=self.data_validation_config.report_file_path,
                                                              report_page_file_path=self.data_validation_config.report_page_file_path,
                                                              is_validated=True,
                                                              message='Data validation performed successfully')
            logging.info(
                f"Data VAlidation Artifact: [{data_validation_artifact}]")
            return data_validation_artifact
        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)

    # def __del__(self):

    #     logging.info(
    #         f"\n\n{'='*20} Data Validation Log Completed {'='*20} \n\n")
