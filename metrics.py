import torch

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def robustness_to_adversarial(model, loader, epsilon=0.1):
    model.eval()
    total_acc = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images_adv = images + epsilon * torch.sign(torch.randn_like(images))
        outputs = model(images_adv)
        total_acc += accuracy(outputs, labels)
    return total_acc / len(loader)