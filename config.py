from pydantic import BaseModel
import json
from pydantic import FilePath


class Config(BaseModel):
    kaggle_mnist_handle: str
    kaggle_mnist_train_file: str
    kaggle_mnist_test_file: str
    kaggle_token_file: FilePath
    data_dir: str
    seed: int
    scrambled_copies: int
    scramble_fraction: float


def load_config(path="config.json") -> Config:
    with open(path, "r") as f:
        return Config(**json.load(f))


config = load_config()
