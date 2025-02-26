import torch
from network import get_model
from dataloader import load_data
from metrics import accuracy, robustness_to_adversarial
from config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    model = get_model().to(device)
    model.load_state_dict(torch.load(CONFIG.model_save_path))
    model.eval()
    
    _, _, test_loader = load_data()
    
    # Standard accuracy (note: no labels in Kaggle test set, so this is for adversarial testing)
    robustness = robustness_to_adversarial(model, test_loader)
    print(f"Adversarial Robustness (epsilon=0.1): {robustness:.4f}")