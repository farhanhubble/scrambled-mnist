import torch
import torch.nn as nn
import torch.optim as optim
from network import get_model
from dataloader import load_data
from metrics import accuracy
from config import CONFIG
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(hyperparams_path="hyper.json"):
    with open(hyperparams_path, "r") as f:
        hp = json.load(f)
    
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=hp["learning_rate"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"]
    )
    
    train_loader, val_loader, _ = load_data(hyperparams_path)
    
    for epoch in range(hp["epochs"]):
        model.train()
        train_loss, train_acc = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(outputs, labels)
        
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_acc += accuracy(outputs, labels)
        
        print(f"Epoch {epoch+1}/{hp['epochs']}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc/len(train_loader):.4f}, "
              f"Val Acc: {val_acc/len(val_loader):.4f}")
    
    torch.save(model.state_dict(), CONFIG.model_save_path)