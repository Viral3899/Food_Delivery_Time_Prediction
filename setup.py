from setuptools import setup, find_packages
from typing import List


PROJECT_NAME = 'Food_Delivery'
VERSION = '0.0.1'
AUTHOR_NAME = 'Viral Sherathiya'
AUTHOR_EMAIL = 'viralsherathiay1008@gmail.com'
HYPHEN_E_DOT = '-e .'


def get_requirements():
    """
    This Function returns the list of Requirements from requirements.txt & all the Packages.
    """

    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements()
)
