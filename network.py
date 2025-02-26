import tensorflow as tf
from tensorflow.keras import layers, models
from config import load_config, ProjectConfig

def create_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple Convolutional Neural Network architecture for MNIST
    
    This network has a simple but effective architecture for MNIST:
    - 2 convolutional layers with max pooling
    - Dropout for regularization
    - Dense layers for classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_adversarial_robust_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a CNN with architectural features that help with adversarial robustness
    
    Features that help with robustness:
    - Larger filter sizes
    - More filters
    - BatchNormalization
    - Lipschitz constrained weights (using kernel constraint)
    """
    # Simple spectral norm constraint to improve Lipschitz constant
    spectral_norm = tf.keras.constraints.MaxNorm(max_value=1.0)
    
    model = models.Sequential([
        # First Convolutional Block - larger filters
        layers.Conv2D(64, (5, 5), activation='relu', padding='same', 
                     input_shape=input_shape, kernel_constraint=spectral_norm),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(128, (5, 5), activation='relu', padding='same', 
                     kernel_constraint=spectral_norm),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block for more capacity
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                     kernel_constraint=spectral_norm),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_constraint=spectral_norm),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_model(model_type="simple", config: ProjectConfig = None):
    """Get a neural network model based on the specified type"""
    if config is None:
        config = load_config()
    
    input_shape = tuple(config.model.input_shape)
    num_classes = config.model.num_classes
    
    if model_type == "simple":
        return create_simple_cnn(input_shape, num_classes)
    elif model_type == "robust":
        return create_adversarial_robust_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    config = load_config()
    model = get_model("robust", config)
    model.summary()