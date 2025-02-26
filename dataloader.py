import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config

def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set = datasets.ImageFolder(root=config.data_dir + "/train/augmented", transform=transform)
    test_set = datasets.ImageFolder(root=config.data_dir + "/test/raw", transform=transform)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader
