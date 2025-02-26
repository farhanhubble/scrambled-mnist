import os
import numpy as np
from pathlib import Path
import random
import idx2numpy
import shutil
from PIL import Image
from config import load_config, ProjectConfig
import tensorflow as tf

def load_mnist_data(file_path):
    """Load MNIST data from IDX file"""
    return idx2numpy.convert_from_file(str(file_path))

def scramble_image(image, scramble_intensity=0.3):
    """
    Scramble an image by randomly permuting blocks of pixels
    
    Args:
        image: The input image (28x28)
        scramble_intensity: Controls how many blocks to scramble (0.0-1.0)
        
    Returns:
        Scrambled version of the image
    """
    h, w = image.shape
    
    # Define block size
    block_size = 4  # 4x4 blocks
    
    # Create copy of the image
    scrambled = image.copy()
    
    # Calculate number of blocks
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    # Determine how many blocks to scramble based on intensity
    num_blocks_to_scramble = int(total_blocks * scramble_intensity)
    
    # Randomly select blocks to scramble
    blocks_to_scramble = random.sample(range(total_blocks), num_blocks_to_scramble)
    
    # Create a list of block positions
    block_positions = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_positions.append((i, j))
    
    # Scramble selected blocks
    for block_idx in blocks_to_scramble:
        # Get random target position for this block
        target_idx = random.choice(range(total_blocks))
        
        # Get source and target block positions
        src_i, src_j = block_positions[block_idx]
        tgt_i, tgt_j = block_positions[target_idx]
        
        # Swap blocks
        src_slice = (
            slice(src_i * block_size, (src_i + 1) * block_size),
            slice(src_j * block_size, (src_j + 1) * block_size)
        )
        
        tgt_slice = (
            slice(tgt_i * block_size, (tgt_i + 1) * block_size),
            slice(tgt_j * block_size, (tgt_j + 1) * block_size)
        )
        
        temp = scrambled[src_slice].copy()
        scrambled[src_slice] = scrambled[tgt_slice]
        scrambled[tgt_slice] = temp
    
    return scrambled

def scramble(config: ProjectConfig = None):
    """Generate 1-N scrambled copies of the original data"""
    if config is None:
        config = load_config()
    
    # Ensure directories exist
    config.data.train_scrambled_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous scrambled data
    if config.data.train_scrambled_dir.exists():
        shutil.rmtree(config.data.train_scrambled_dir)
        config.data.train_scrambled_dir.mkdir(parents=True)
    
    # Load original MNIST training data
    images_path = config.data.train_raw_dir / "train-images-idx3-ubyte"
    labels_path = config.data.train_raw_dir / "train-labels-idx1-ubyte"
    
    images = load_mnist_data(images_path)
    labels = load_mnist_data(labels_path)
    
    # Generate and save scrambled versions
    print(f"Generating scrambled copies of training data...")
    
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        
        # Determine number of scrambled copies for this image (1 to max_copies)
        num_copies = random.randint(1, config.data.num_scrambled_copies)
        
        for copy_idx in range(num_copies):
            # Generate scrambled version with varying intensity
            intensity = random.uniform(0.2, 0.8)
            scrambled_img = scramble_image(image, scramble_intensity=intensity)
            
            # Save as PNG
            img_filename = f"scrambled_{i}_{copy_idx}.png"
            label_filename = f"scrambled_{i}_{copy_idx}.txt"
            
            # Save image
            pil_img = Image.fromarray(scrambled_img.astype(np.uint8))
            pil_img.save(config.data.train_scrambled_dir / img_filename)
            
            # Save label
            with open(config.data.train_scrambled_dir / label_filename, 'w') as f:
                f.write(str(label))
    
    print(f"Created scrambled dataset at {config.data.train_scrambled_dir}")

def augment(config: ProjectConfig = None):
    """Combine original and scrambled data into augmented dataset"""
    if config is None:
        config = load_config()
    
    # Ensure directory exists
    config.data.train_augmented_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous augmented data
    if config.data.train_augmented_dir.exists():
        shutil.rmtree(config.data.train_augmented_dir)
        config.data.train_augmented_dir.mkdir(parents=True)
    
    # Copy original data to augmented directory
    orig_images_path = config.data.train_raw_dir / "train-images-idx3-ubyte"
    orig_labels_path = config.data.train_raw_dir / "train-labels-idx1-ubyte"
    
    shutil.copy(orig_images_path, config.data.train_augmented_dir / "original_images.idx")
    shutil.copy(orig_labels_path, config.data.train_augmented_dir / "original_labels.idx")
    
    # Get scrambled data
    scrambled_files = sorted(list(config.data.train_scrambled_dir.glob("*.png")))
    
    # Prepare arrays for combined dataset
    original_images = load_mnist_data(orig_images_path)
    original_labels = load_mnist_data(orig_labels_path)
    
    # Count scrambled images
    num_scrambled = len(scrambled_files)
    
    # Create TFRecord files for efficient data loading
    tfrecord_path = config.data.train_augmented_dir / "augmented_data.tfrecord"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        # First add all original images
        for i in range(len(original_images)):
            image = original_images[i].astype(np.uint8)
            label = int(original_labels[i])
            
            # Convert to TFRecord
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
                    'is_augmented': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                }
            ))
            writer.write(example.SerializeToString())
        
        # Then add all scrambled images
        for img_path in scrambled_files:
            # Get corresponding label file
            label_path = config.data.train_scrambled_dir / (img_path.stem + ".txt")
            
            if not label_path.exists():
                continue
                
            # Read image and label
            image = np.array(Image.open(img_path))
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
            
            # Convert to TFRecord
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
                    'is_augmented': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                }
            ))
            writer.write(example.SerializeToString())
    
    print(f"Created augmented dataset at {config.data.train_augmented_dir}")
    print(f"Total images: {len(original_images) + num_scrambled} (Original: {len(original_images)}, Scrambled: {num_scrambled})")

if __name__ == "__main__":
    config = load_config()
    scramble(config)
    augment(config)