from setuptools import setup, find_packages

setup(
    name='morph',
    version="0.0.1",
    packages=find_packages(include=['morph', 'morph.*']),
)