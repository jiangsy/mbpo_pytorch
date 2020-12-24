from setuptools import find_packages
from setuptools import setup

setup(
    name='mbpo_pytorch',
    auther='Shengyi Jiang',
    author_email='shengyi.jiang@outlook.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.4.0',
        'mujoco-py',
        'scipy',
        'numpy',
        'gym>=0.17.0',
        'pyglib',
        'munch',
        'pyyaml',
        'colorama',
        'tensorboard>=1.15.0',
        'pandas'
    ],
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml"],
    }
)
