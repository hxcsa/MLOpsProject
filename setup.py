from setuptools import find_packages, setup

with open('requierments.txt') as f:
    requierments = f.read().splitlines()

setup(
    name='breast_cancer_pred',
    packages=find_packages(),
    version='0.1.0',
    description='Breast Cancer Prediction',
    author='Alhas',
    license='MIT',
    install_requires=requierments,
)