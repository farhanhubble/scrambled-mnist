import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import load_config, ProjectConfig
from dataloader import generate_adversarial_examples

class MetricsTracker:
    def __init__(self, config: ProjectConfig = None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        
        # Create metrics directory
        self.metrics_dir = Path("./metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Define metrics
        self.metrics = {
            "train_accuracy": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_loss": [],
            "test_accuracy": None,
            "test_loss": None,
            "adversarial_accuracy": None,
            "convergence_epoch": None,
            "confusion_matrix": None,
            "classification_report": None,
            "training_time": None,
            "epoch_times": []
        }
    
    def update_train_metrics(self, epoch, logs):
        """Update training metrics after each epoch"""
        self.metrics["train_accuracy"].append(logs.get("accuracy", logs.get("sparse_categorical_accuracy", 0)))
        self.metrics["train_loss"].append(logs.get("loss", 0))
        self.metrics["val_accuracy"].append(logs.get("val_accuracy", logs.get("val_sparse_categorical_accuracy", 0)))
        self.metrics["val_loss"].append(logs.get("val_loss", 0))
    
    def update_epoch_time(self, epoch_time):
        """Update epoch execution time"""
        self.metrics["epoch_times"].append(epoch_time)
    
    def set_training_time(self, training_time):
        """Set total training time"""
        self.metrics["training_time"] = training_time
    
    def set_convergence_epoch(self, epoch):
        """Set the epoch at which convergence was achieved"""
        self.metrics["convergence_epoch"] = epoch
    
    def evaluate_test_performance(self, model, test_dataset):
        """Evaluate model on test dataset"""
        test_loss, test_accuracy = model.evaluate(test_dataset)
        self.metrics["test_accuracy"] = float(test_accuracy)
        self.metrics["test_loss"] = float(test_loss)
        
        return test_loss, test_accuracy
    
    def evaluate_adversarial_robustness(self, model, test_dataset, epsilon=0.1):
        """Evaluate model on adversarial examples"""
        # Generate adversarial examples
        adversarial_dataset = generate_adversarial_examples(model, test_dataset, epsilon, self.config)
        
        # Evaluate on adversarial examples
        adv_loss, adv_accuracy = model.evaluate(adversarial_dataset)
        self.metrics["adversarial_accuracy"] = float(adv_accuracy)
        
        return adv_loss, adv_accuracy
    
    def compute_confusion_matrix(self, model, test_dataset):
        """Compute confusion matrix"""
        y_true = []
        y_pred = []
        
        for images, labels in test_dataset:
            predictions = model.predict(images)
            pred_labels = np.argmax(predictions, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(pred_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics["confusion_matrix"] = cm.tolist()  # Convert to list for JSON serialization
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        self.metrics["classification_report"] = report
        
        return cm, report
    
    def save_metrics(self, experiment_name="default"):
        """Save metrics to JSON file"""
        metrics_file = self.metrics_dir / f"{experiment_name}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")
    
    def plot_training_curves(self, experiment_name="default"):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["train_accuracy"], label="Training Accuracy")
        plt.plot(self.metrics["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics["train_loss"], label="Training Loss")
        plt.plot(self.metrics["val_loss"], label="Validation Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        fig_path = self.metrics_dir / f"{experiment_name}_training_curves.png"
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Training curves saved to {fig_path}")
    
    def plot_confusion_matrix(self, experiment_name="default"):
        """Plot confusion matrix"""
        if self.metrics["confusion_matrix"] is None:
            print("Confusion matrix not computed yet")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.metrics["confusion_matrix"], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10)
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        
        # Save figure
        fig_path = self.metrics_dir / f"{experiment_name}_confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Confusion matrix saved to {fig_path}")
    
    def analyze_convergence(self):
        """Analyze convergence speed"""
        # Determine when the model reached 95% of its maximum validation accuracy
        val_acc = np.array(self.metrics["val_accuracy"])
        max_acc = np.max(val_acc)
        threshold = 0.95 * max_acc
        
        # Find the first epoch where accuracy exceeded the threshold
        for epoch, acc in enumerate(val_acc):
            if acc >= threshold:
                self.metrics["convergence_epoch"] = epoch
                break
        
        return self.metrics["convergence_epoch"]
    
    def get_summary(self):
        """Get a summary of key metrics"""
        convergence_epoch = self.analyze_convergence() if self.metrics["convergence_epoch"] is None else self.metrics["convergence_epoch"]
        
        summary = {
            "test_accuracy": self.metrics["test_accuracy"],
            "adversarial_accuracy": self.metrics["adversarial_accuracy"],
            "convergence_epoch": convergence_epoch,
            "training_time": self.metrics["training_time"],
            "avg_epoch_time": np.mean(self.metrics["epoch_times"]) if self.metrics["epoch_times"] else None
        }
        
        return summary

def evaluate_out_of_distribution(model, config: ProjectConfig = None):
    """
    Evaluate model on out-of-distribution data
    
    For simplicity, we'll rotate MNIST images as a form of OOD data
    """
    if config is None:
        config = load_config()
    
    from dataloader import load_test_dataset
    import tensorflow_addons as tfa
    
    # Load test dataset
    test_ds = load_test_dataset(config)
    
    # Create rotated versions
    angles = [30, 45, 60, 90]
    ood_accuracies = {}
    
    for angle in angles:
        rotated_images = []
        labels = []
        
        for images, batch_labels in test_ds.unbatch():
            # Expand dimensions for rotation
            image = tf.expand_dims(images, 0)
            # Rotate image
            rotated = tfa.image.rotate(image, angle * np.pi / 180)
            rotated = tf.squeeze(rotated, 0)
            
            rotated_images.append(rotated)
            labels.append(batch_labels)
        
        # Create dataset from rotated images
        ood_ds = tf.data.Dataset.from_tensor_slices((rotated_images, labels))
        ood_ds = ood_ds.batch(config.training.batch_size)
        
        # Evaluate
        _, accuracy = model.evaluate(ood_ds)
        ood_accuracies[f"rotation_{angle}"] = float(accuracy)
    
    return ood_accuracies