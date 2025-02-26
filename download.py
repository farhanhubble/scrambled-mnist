import kagglehub
from config import CONFIG
import os

def download_mnist_train():
    os.makedirs(CONFIG.train_raw_dir, exist_ok=True)
    kagglehub.login(CONFIG.kaggle_key_path)
    kagglehub.dataset_download("digit-recognizer", path=CONFIG.train_raw_dir, subset="train.csv")

def download_mnist_test():
    os.makedirs(CONFIG.test_raw_dir, exist_ok=True)
    kagglehub.login(CONFIG.kaggle_key_path)
    kagglehub.dataset_download("digit-recognizer", path=CONFIG.test_raw_dir, subset="test.csv")