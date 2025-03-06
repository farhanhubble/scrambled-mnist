import torch
from network import CNN
from config import config
from dataloader import get_dataloaders
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd


def create_confusion_matrix():
    data = np.zeros((11, 11), dtype=int)
    columns = [str(i) for i in range(10)] + ["scrambled"]
    indices = [str(i) for i in range(10)] + ["scrambled"]
    return pd.DataFrame(data, columns=columns, index=indices)

def update_confusion_matrix(confusion_matrix, predicted, labels):
    for i in range(len(predicted)):
        confusion_matrix[labels[i]][predicted[i]] += 1



def evaluate():
    _, test_loader = get_dataloaders()
    model = CNN()
    model.load_state_dict(torch.load(config.saved_model_dir + "/" + config.model_name))
    model.eval()

    confusion_matrix = np.zeros((11, 11), dtype=int)

    correct, total = 0, 0
    with open(config.report_file, "a") as f:
        f.write(f"[{datetime.now()}] Testing started\n")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            update_confusion_matrix(confusion_matrix, predicted, labels)

    test_accuracy = 100 * correct / total
    print(f"Accuracy: {test_accuracy:.2f}%")
    with open(config.report_file, "a") as f:
        f.write(f"[{datetime.now()}] Test accuracy: {test_accuracy}\n")
