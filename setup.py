from setuptools import setup, find_namespace_packages

setup(
    name='utilsnamespace',
    
    packages=find_namespace_packages(include=['utils.*'])
)