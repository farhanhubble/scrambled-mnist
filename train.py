import tensorflow as tf
import time
import os
from pathlib import Path
from config import load_config, ProjectConfig
from network import get_model
from dataloader import create_train_val_datasets
from metrics import MetricsTracker

class TimeHistory(tf.keras.callbacks.Callback):
    """Callback to track epoch times"""
    def on_train_begin(self, logs={}):
        self.times = []
        self.start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)
    
    def on_train_end(self, logs={}):
        self.total_time = time.time() - self.start_time

def train(model_type="robust", experiment_name="default", config: ProjectConfig = None):
    """Train a model on the augmented dataset"""
    if config is None:
        config = load_config()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(config.random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Create directories
    checkpoint_dir = config.model.checkpoint_dir / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = config.model.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = get_model(model_type, config)
    
    # Initialize optimizer
    if config.training.optimizer.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    elif config.training.optimizer.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.training.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=config.training.loss,
        metrics=['accuracy']
    )
    
    # Load datasets
    train_ds, val_ds = create_train_val_datasets(config)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)
    
    # Create callbacks
    time_history = TimeHistory()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.training.early_stopping_patience,
        restore_best_weights=True
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'model_{epoch:02d}.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    class MetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            metrics_tracker.update_train_metrics(epoch, logs)
    
    # Train model
    print(f"Starting training for {config.training.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs,
        callbacks=[early_stopping, checkpoint_callback, time_history, MetricsCallback()]
    )
    
    # Save epoch times
    for epoch_time in time_history.times:
        metrics_tracker.update_epoch_time(epoch_time)
    
    # Save total training time
    metrics_tracker.set_training_time(time_history.total_time)
    
    # Analyze convergence
    convergence_epoch = metrics_tracker.analyze_convergence()
    print(f"Model converged at epoch {convergence_epoch}")
    
    # Save model
    model_path = model_dir / f"{experiment_name}.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_tracker.save_metrics(experiment_name)
    metrics_tracker.plot_training_curves(experiment_name)
    
    return model, metrics_tracker

if __name__ == "__main__":
    config = load_config()
    
    # Train on original data only
    train(model_type="robust", experiment_name="original", config=config)
    
    # Train on augmented data
    train(model_type="robust", experiment_name="augmented", config=config)