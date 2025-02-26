import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import get_dataloaders
from network import CNN
from config import config

def train():
    train_loader, _ = get_dataloaders()
    
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
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
