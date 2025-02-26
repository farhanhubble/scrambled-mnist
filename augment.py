import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from config import config

def scramble(image: torch.Tensor) -> torch.Tensor:
    img_np = np.array(image)
    np.random.shuffle(img_np)
    return torch.tensor(img_np)

def augment():
    transform = transforms.ToTensor()
    raw_path = config.data_dir + "/train/raw"
    scrambled_path = config.data_dir + "/train/scrambled"
    augmented_path = config.data_dir + "/train/augmented"

    os.makedirs(scrambled_path, exist_ok=True)
    os.makedirs(augmented_path, exist_ok=True)

    for img_name in os.listdir(raw_path):
        img = Image.open(os.path.join(raw_path, img_name))
        img_tensor = transform(img)

        # Save scrambled copies
        for i in range(5):
            scrambled_img = scramble(img_tensor)
            transforms.ToPILImage()(scrambled_img).save(f"{scrambled_path}/{img_name[:-4]}_scrambled{i}.png")

        # Save augmented images
        img.save(f"{augmented_path}/{img_name}")
