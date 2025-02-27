from config import config
import os
import json
import kagglehub
import shutil


def set_auth():
    with open(config.kaggle_token_file, "r") as f:
        kaggle_auth = json.load(f)
        os.environ["KAGGLE_USERNAME"] = kaggle_auth["username"]
        os.environ["KAGGLE_KEY"] = kaggle_auth["key"]


def download_mnist_train():
    os.makedirs(config.data_dir + "/train/raw", exist_ok=True)
    local_path = kagglehub.dataset_download(config.kaggle_mnist_handle, path=config.kaggle_mnist_train_file)
    shutil.copy(local_path, config.data_dir + "/train/raw")

def download_mnist_test():
    os.makedirs(config.data_dir + "/test/raw", exist_ok=True)
    local_path = kagglehub.dataset_download(config.kaggle_mnist_handle, path=config.kaggle_mnist_test_file)
    shutil.copy(local_path, config.data_dir + "/test/raw")