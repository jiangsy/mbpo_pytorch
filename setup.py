from setuptools import find_packages
from setuptools import setup

setup(
    name='slbo_pytorch',
    auther='Shengyi Jiang',
    author_email='shengyi.jiang@outlook.com',
    packages=find_packages(),
    package_data={},
    install_requires=[
        'torch>=1.4.0',
        'gym>=0.17.0',
        'numpy',
        'pyglib',
        'scipy',
        'munch',
        'pyyaml',
        'colorama',
        'tensorboard>=1.15.0'
    ])
