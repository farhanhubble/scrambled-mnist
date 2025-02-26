import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 30
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"

class DataConfig(BaseModel):
    kaggle_dataset: str = "zalando-research/fashionmnist"
    kaggle_credentials_path: Path = Field(default=Path.home() / ".kaggle/kaggle.json")
    data_dir: Path = Field(default=Path("./data"))
    train_raw_dir: Path = Field(default=Path("./data/train/raw"))
    test_raw_dir: Path = Field(default=Path("./data/test/raw"))
    train_scrambled_dir: Path = Field(default=Path("./data/train/scrambled"))
    train_augmented_dir: Path = Field(default=Path("./data/train/augmented"))
    num_scrambled_copies: int = 3  # Generate 1-3 scrambled copies per image

class ModelConfig(BaseModel):
    model_dir: Path = Field(default=Path("./models"))
    checkpoint_dir: Path = Field(default=Path("./checkpoints"))
    input_shape: List[int] = [28, 28, 1]
    num_classes: int = 10

class ProjectConfig(BaseModel):
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    random_seed: int = 42

def load_config(config_path: str = "hyper.json") -> ProjectConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return ProjectConfig(**config_data)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return ProjectConfig()
    except json.JSONDecodeError:
        print(f"Error parsing {config_path}. Using default configuration.")
        return ProjectConfig()

# Create required directories
def create_directories(config: ProjectConfig):
    directories = [
        config.data.data_dir,
        config.data.train_raw_dir,
        config.data.test_raw_dir,
        config.data.train_scrambled_dir,
        config.data.train_augmented_dir,
        config.model.model_dir,
        config.model.checkpoint_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)