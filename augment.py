import os
import torch
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


def scramble(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Scrambles the pixels of a given PyTorch tensor representing an image.

    Args:
        image_tensor (torch.Tensor): 1x28x28 grayscale image tensor.

    Returns:
        torch.Tensor: Scrambled image tensor.
    """
    img_np = image_tensor.numpy().flatten()  # Convert to NumPy and flatten
    np.random.shuffle(img_np)  # Shuffle pixels randomly
    scrambled_tensor = torch.tensor(img_np.reshape(1, 28, 28))  # Reshape back to 28x28
    return scrambled_tensor


def augment(input_dir: str, output_dir: str, num_scrambled: int = 5):
    """
    Augments the dataset by creating scrambled versions of images.

    Args:
        input_dir (str): Directory containing original images (structured by label).
        output_dir (str): Directory to save augmented images.
        num_scrambled (int): Number of scrambled copies to generate per image.
    """
    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    os.makedirs(output_dir, exist_ok=True)

    # Create a directory for scrambled images labeled as "69"
    scrambled_label_dir = os.path.join(output_dir, "69")
    os.makedirs(scrambled_label_dir, exist_ok=True)

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue  # Skip non-directory files

        # Create corresponding output directory for original images
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(label_dir), desc=f"Processing label {label}"):
            img_path = os.path.join(label_dir, img_name)
            img = Image.open(img_path).convert("L")  # Ensure grayscale
            img_tensor = transform(img)  # Convert to PyTorch tensor

            # Save original image in its label directory
            img.save(os.path.join(output_label_dir, img_name))

            # Generate scrambled copies and save them in label "69"
            for i in range(num_scrambled):
                scrambled_img = scramble(img_tensor)
                to_pil(scrambled_img).save(
                    os.path.join(
                        scrambled_label_dir, f"{img_name[:-4]}_scrambled{i}.png"
                    )
                )
