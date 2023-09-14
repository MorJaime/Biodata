import os
from setuptools import setup, find_namespace_packages, find_packages

setup(
    name='utilsnamespace',
    version = "0.0.1",
    author = "Jaime Morales",
    author_email = "j.morzav@gmail.com",
    description = ("utility modules for biodata ML"),
    #packages=find_namespace_packages(include=['codes.*',])
    packages=find_packages()
)