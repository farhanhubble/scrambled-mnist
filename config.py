from pydantic import BaseModel
from pathlib import Path
import json

class ProjectConfig(BaseModel):
    kaggle_dataset_url: str = "https://www.kaggle.com/c/digit-recognizer/data"
    kaggle_key_path: str = "./kaggle.json"
    data_dir: str = "./data"
    train_raw_dir: str = "./data/train/raw"
    test_raw_dir: str = "./data/test/raw"
    scrambled_dir: str = "./data/train/scrambled"
    augmented_dir: str = "./data/train/augmented"
    model_save_path: str = "./models/model.pth"

def load_config(config_path: str = "config.json") -> ProjectConfig:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    return ProjectConfig(**config_data)

CONFIG = load_config()