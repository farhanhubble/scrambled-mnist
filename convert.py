"""Convert MNIST CSV data to PNG images."""
import os
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def _skip_header(reader):
    # Skip the header
    next(reader)


def csv_to_images(csv_path: str, output_dir: str):
    """
    Converts MNIST CSV data (785 columns per row: label + 784 pixels)
    into PNG images suitable for PyTorch's ImageFolder dataset.

    Args:
        csv_path (str): Path to the MNIST CSV file.
        output_dir (str): Directory where images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        _skip_header(reader)
        data = list(reader)

    for i, row in tqdm(enumerate(data), total=len(data)):
        label = row[0]  # First column is the label
        pixels = list(map(int, row[1:]))  # Convert remaining columns to integers
        img = Image.new("L", (28, 28))  # Create grayscale image (L mode)
        img.putdata(pixels)  # Fill image with pixel values

        # Create label directory
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Save image
        img.save(os.path.join(label_dir, f"{i}.png"))
