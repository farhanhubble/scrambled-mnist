import os
import kaggle
from config import config

def download_mnist_train():
    os.makedirs(config.data_dir + "/train/raw", exist_ok=True)
    kaggle.api.dataset_download_files(config.kaggle_url, path=config.data_dir + "/train/raw", unzip=True)

def download_mnist_test():
    os.makedirs(config.data_dir + "/test/raw", exist_ok=True)
    kaggle.api.dataset_download_files(config.kaggle_url, path=config.data_dir + "/test/raw", unzip=True)
