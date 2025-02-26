import torch
from network import CNN
from dataloader import get_dataloaders
from config import config

def evaluate():
    _, test_loader = get_dataloaders()
    model = CNN()
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {100 * correct / total:.2f}%")
