from datetime import datetime
import numpy as np
import numpy as np
from datetime import datetime
import torch

import pandas as pd
from tqdm import tqdm

from config import config
from dataloader import get_dataloaders, idx2class
from network import CNN


def create_confusion_matrix():
    data = np.zeros((11, 11), dtype=int)
    columns = [str(i) for i in range(10)] + ["99"]
    indices = [str(i) for i in range(10)] + ["99"]
    return pd.DataFrame(data, columns=columns, index=indices)

def update_confusion_matrix(confusion_matrix, predicted, labels):
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    for i in range(len(predicted)):
        confusion_matrix.loc[idx2class[labels[i]], idx2class[predicted[i]]] += 1


def evaluate():
    _, test_loader = get_dataloaders()
    model = CNN()
    model.load_state_dict(torch.load(config.saved_model_dir + "/" + config.model_name))
    model.eval()

    confusion_matrix = create_confusion_matrix()

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
        f.write(f"[{datetime.now()}] Confusion matrix:\n")
        confusion_matrix.to_markdown(f, 'a')
