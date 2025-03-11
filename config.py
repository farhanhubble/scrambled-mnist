from pydantic import BaseModel
import json
import os


class Config(BaseModel):
    kaggle_mnist_handle: str
    kaggle_mnist_train_file: str
    kaggle_mnist_test_file: str
    kaggle_token_file: FilePath
    data_dir: str
    seed: int
    scrambled_copies: int
    scramble_fraction: float
    saved_model_dir: str
    model_name: str
    report_file: str


def load_config(path="config.json") -> Config:
    with open(path, "r") as f:
        return Config(**json.load(f))
    
def set_env(config: Config):
    for key, value in config.model_dump().items():
        os.environ[key] = str(value)


config = load_config()
set_env(config)

if __name__ == "__main__":
    print(config.model_dump_json(indent=2))