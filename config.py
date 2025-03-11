from pydantic import BaseModel, Field, FilePath
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
    mnist_c_main_dataset: str = Field(..., description="URL for Google MNIST-C main datset")
    mnist_c_aux_dataset: str = Field(..., description="URL for Google MNIST-C aux dataset")
    tmp_dir: str = Field(..., description="Temporary directory for external data")
    external_data_subdir: str = Field(..., description="Subdirectory for external data, relative to data_dir")



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