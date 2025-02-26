import tensorflow as tf
import numpy as np
import idx2numpy
from pathlib import Path
from config import load_config, ProjectConfig
import random

def parse_tfrecord_fn(example):
    """Parse a TFRecord example"""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'is_augmented': tf.io.FixedLenFeature([], tf.int64),
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    # Parse image and reshape
    image = tf.io.decode_raw(example['image'], tf.uint8)
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    image = tf.reshape(image, [height, width])
    
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # Add channel dimension
    
    label = tf.cast(example['label'], tf.int32)
    
    return image, label

def load_augmented_dataset(config: ProjectConfig = None):
    """Load the augmented dataset (original + scrambled images)"""
    if config is None:
        config = load_config()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Define path to TFRecord
    tfrecord_path = config.data.train_augmented_dir / "augmented_data.tfrecord"
    
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"Augmented dataset not found at {tfrecord_path}. Run augmentation first.")
    
    # Create dataset from TFRecord
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def load_test_dataset(config: ProjectConfig = None):
    """Load the MNIST test dataset"""
    if config is None:
        config = load_config()
    
    # Define path to test data
    test_images_path = config.data.test_raw_dir / "t10k-images-idx3-ubyte"
    test_labels_path = config.data.test_raw_dir / "t10k-labels-idx1-ubyte"
    
    if not test_images_path.exists() or not test_labels_path.exists():
        raise FileNotFoundError(f"Test data not found. Run download first.")
    
    # Load test data
    test_images = idx2numpy.convert_from_file(str(test_images_path))
    test_labels = idx2numpy.convert_from_file(str(test_labels_path))
    
    # Normalize images
    test_images = test_images.astype(np.float32) / 255.0
    test_images = np.expand_dims(test_images, -1)  # Add channel dimension
    
    return tf.data.Dataset.from_tensor_slices((test_images, test_labels))

def create_train_val_datasets(config: ProjectConfig = None):
    """Create training and validation datasets"""
    if config is None:
        config = load_config()
    
    # Load augmented dataset
    dataset = load_augmented_dataset(config)
    
    # Shuffle dataset
    dataset = dataset.shuffle(buffer_size=10000, seed=config.random_seed)
    
    # Calculate dataset size
    dataset_size = sum(1 for _ in dataset)
    
    # Calculate split sizes
    val_size = int(dataset_size * config.training.validation_split)
    train_size = dataset_size - val_size
    
    # Split into training and validation
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    # Batch and prefetch
    train_ds = train_ds.batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def create_test_dataset(config: ProjectConfig = None):
    """Create test dataset"""
    if config is None:
        config = load_config()
    
    # Load test dataset
    test_ds = load_test_dataset(config)
    
    # Batch and prefetch
    test_ds = test_ds.batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return test_ds

def generate_adversarial_examples(model, dataset, epsilon=0.1, config: ProjectConfig = None):
    """
    Generate adversarial examples using Fast Gradient Sign Method (FGSM)
    
    Args:
        model: Trained model
        dataset: Test dataset
        epsilon: Perturbation magnitude
    
    Returns:
        Adversarial examples dataset
    """
    if config is None:
        config = load_config()
    
    adversarial_images = []
    labels = []
    
    for images, batch_labels in dataset:
        # Convert to TensorFlow variables
        images = tf.convert_to_tensor(images)
        batch_labels = tf.convert_to_tensor(batch_labels)
        
        with tf.GradientTape() as tape:
            # Watch input images
            tape.watch(images)
            # Get predictions
            predictions = model(images)
            # Calculate loss
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(batch_labels, predictions)
        
        # Get gradients
        gradients = tape.gradient(loss, images)
        
        # Create adversarial examples
        signed_grad = tf.sign(gradients)
        adversarial_image = images + epsilon * signed_grad
        
        # Clip to ensure valid pixel values
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        
        adversarial_images.append(adversarial_image)
        labels.append(batch_labels)
    
    # Concatenate results
    adversarial_images = tf.concat(adversarial_images, axis=0)
    labels = tf.concat(labels, axis=0)
    
    return tf.data.Dataset.from_tensor_slices((adversarial_images, labels)).batch(config.training.batch_size)

if __name__ == "__main__":
    config = load_config()
    train_ds, val_ds = create_train_val_datasets(config)
    test_ds = create_test_dataset(config)
    
    print(f"Training dataset batches: {len(list(train_ds))}")
    print(f"Validation dataset batches: {len(list(val_ds))}")
    print(f"Test dataset batches: {len(list(test_ds))}")