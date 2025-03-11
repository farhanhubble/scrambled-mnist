from config import config
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def _convert_mnist_c(data_root: str, output_dir: str):
    subdirs = os.listdir(data_root)
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_root, subdir)):
            train_data_path = os.path.join(data_root, subdir, "train_images.npy")
            train_label_path = os.path.join(data_root, subdir, "train_labels.npy")
            _npy_to_images(
                train_data_path,
                train_label_path,
                os.path.join(output_dir, subdir, "train"),
            )

            test_data_path = os.path.join(data_root, subdir, "test_images.npy")
            test_label_path = os.path.join(data_root, subdir, "test_labels.npy")
            _npy_to_images(
                test_data_path,
                test_label_path,
                os.path.join(output_dir, subdir, "test"),
            )


def _npy_to_images(data_path: str, label_path: str, output_dir: str):
    """
    Converts MNIST-C NPY data (2 files: data and labels) into PNG images
    suitable for PyTorch's ImageFolder dataset.

    Args:
        data_path (str): Path to the NPY data file.
        label_path (str): Path to the NPY label file.
        output_dir (str): Directory where images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(data_path)
    labels = np.load(label_path)

    print(f"Converting {data_path} to PNG images in {output_dir}...")
    for i, (pixels, label) in tqdm(
        enumerate(zip(data, labels)), total=len(data), leave=False
    ):
        img = Image.new("L", (28, 28))  # Create grayscale image (L mode)
        img.putdata(pixels.flatten())  # Fill image with pixel values

        # Create label directory
        label_dir = os.path.join(output_dir, str(label.item()))
        os.makedirs(label_dir, exist_ok=True)

        # Save image
        img.save(os.path.join(label_dir, f"{i}.png"))
    print("Conversion finished!")

def convert_mnist_c():
    _convert_mnist_c(
        config.data_dir + "/external/raw/mnist_c",
        config.data_dir + "/external/converted/mnist_c",
    )
    _convert_mnist_c(
        config.data_dir + "/external/raw/mnist_c_leftovers",
        config.data_dir + "/external/converted/mnist_c_leftovers",
    )
