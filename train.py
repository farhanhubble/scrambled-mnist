import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import get_dataloaders
from network import CNN
import json

def load_hyperparameters():
    with open("hyper.json", "r") as f:
        return json.load(f)

hyperparams = load_hyperparameters()

def train():
    train_loader, _ = get_dataloaders()
    
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(hyperparams["num_epochs"]):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
