from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='breast_cancer_pred',
    packages=find_packages(),
    version='0.1.0',
    description='Breast Cancer Prediction',
    author='Alhas',
    license='MIT',
    install_requires=requirements,
)