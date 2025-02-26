import pandas as pd
import numpy as np
import os
from config import CONFIG

def scramble(image: np.ndarray, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    flat = image.flatten()
    np.random.shuffle(flat)
    return flat.reshape(image.shape)

def augment():
    os.makedirs(CONFIG.scrambled_dir, exist_ok=True)
    os.makedirs(CONFIG.augmented_dir, exist_ok=True)
    
    # Load original train data
    df = pd.read_csv(f"{CONFIG.train_raw_dir}/train.csv")
    labels = df["label"].values
    images = df.drop("label", axis=1).values.reshape(-1, 28, 28)  # 28x28 images
    
    augmented_images = [images]
    augmented_labels = [labels]
    
    # Generate 1-5 scrambled copies
    for i in range(np.random.randint(1, 6)):
        scrambled = np.array([scramble(img, seed=i) for img in images])
        augmented_images.append(scrambled)
        augmented_labels.append(labels)
        np.save(f"{CONFIG.scrambled_dir}/scrambled_{i}.npy", scrambled)
    
    # Concatenate original + scrambled
    final_images = np.concatenate(augmented_images, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    # Save augmented dataset
    np.save(f"{CONFIG.augmented_dir}/images.npy", final_images)
    np.save(f"{CONFIG.augmented_dir}/labels.npy", final_labels)

if __name__ == "__main__":
    augment()