import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

class2idx = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "99": 10  # Use 99 for scrambled digits to ensure pytorch assigns it the last index
}

idx2class = {v: k for k, v in class2idx.items()}

def load_hyperparameters():
    with open("hypers.json", "r") as f:
        return json.load(f)

hyperparams = load_hyperparameters()

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    train_set = datasets.ImageFolder(root="data/train/augmented", transform=transform)
    test_set = datasets.ImageFolder(root="data/test/converted", transform=transform)

    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False)

    return train_loader, test_loader
