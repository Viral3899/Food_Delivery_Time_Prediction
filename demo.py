from food_delivery.pipeline.pipeline import Pipeline
from food_delivery.logger import logging
from food_delivery.exception import CustomException
from food_delivery.config.configuration import Configuration
import sys

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    try:
        pipe = Pipeline()
        pipe.start()

    except Exception as e:
        logging.error(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
