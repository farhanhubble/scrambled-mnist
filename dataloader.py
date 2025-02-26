import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from config import CONFIG
import json

class MNISTDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.images[idx], self.labels[idx]
        return self.images[idx]

def load_data(hyperparams_path="hyper.json"):
    with open(hyperparams_path, "r") as f:
        hp = json.load(f)
    
    # Load augmented train data
    images = np.load(f"{CONFIG.augmented_dir}/images.npy")
    labels = np.load(f"{CONFIG.augmented_dir}/labels.npy")
    
    # Split into train/val
    train_idx, val_idx = train_test_split(
        range(len(images)), train_size=hp["train_split"], random_state=42
    )
    train_dataset = MNISTDataset(images[train_idx], labels[train_idx])
    val_dataset = MNISTDataset(images[val_idx], labels[val_idx])
    
    # Load test data (no labels in Kaggle MNIST test set)
    test_df = pd.read_csv(f"{CONFIG.test_raw_dir}/test.csv")
    test_images = test_df.values.reshape(-1, 28, 28)
    test_dataset = MNISTDataset(test_images)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hp["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader