from setuptools import find_packages
from setuptools import setup

setup(
    name='mbpo_pytorch',
    auther='Shengyi Jiang',
    author_email='shengyi.jiang@outlook.com',
    packages=find_packages(),
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
    ],
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml"],
    }
)
