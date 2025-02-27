from config import config
import numpy as np
import torch

def seed():
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)