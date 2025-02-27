import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import get_dataloaders
from network import CNN
import json
from tqdm import tqdm
from config import config
import os


def load_hyperparameters():
    with open("hypers.json", "r") as f:
        return json.load(f)


hyperparams = load_hyperparameters()


def train():
    train_loader, _ = get_dataloaders()

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    progress = tqdm(
        range(hyperparams["num_epochs"]),
        desc="Training",
        position=0,
        unit="epoch",
    )
    for epoch in progress:
        model.train()
        total_loss = 0

        for images, labels in tqdm(
            train_loader, desc="Training batch", leave=False, position=1, unit="batch"
        ):
            progress.set_postfix(training_loss=total_loss)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    os.makedirs(config.saved_model_dir, exist_ok=True)
    torch.save(model.state_dict(), config.saved_model_dir + "/" + config.model_name)
