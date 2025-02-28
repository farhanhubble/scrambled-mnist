import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

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
