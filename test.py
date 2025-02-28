import torch
from network import CNN
from config import config
from dataloader import get_dataloaders
from tqdm import tqdm
from datetime import datetime

def evaluate():
    _, test_loader = get_dataloaders()
    model = CNN()
    model.load_state_dict(torch.load(config.saved_model_dir + "/" + config.model_name))
    model.eval()

    correct, total = 0, 0
    with open(config.report_file, "w") as f:
        f.write(f"Testting started at {datetime.now()}\n")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    print(f"Accuracy: {test_accuracy:.2f}%")
    with open(config.report_file, "w") as f:
        f.write(f"[{datetime.datetime.now()}] Test accuracy: {test_accuracy}\n")
