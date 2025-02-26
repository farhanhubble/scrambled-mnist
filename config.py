from pydantic import BaseModel
import json

class Config(BaseModel):
    kaggle_url: str
    kaggle_token: str
    data_dir: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    seed: int

def load_config(path="config.json") -> Config:
    with open(path, "r") as f:
        return Config(**json.load(f))

config = load_config()
