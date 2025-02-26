import typer
from pathlib import Path
import time
import json
from typing import Optional

from config import load_config, create_directories
from download import download_mnist_train, download_mnist_test
from augment import scramble, augment
from train import train
from test import evaluate, compare_experiments

app = typer.Typer()

@app.command()
def setup(config_path: str = "hyper.json"):
    """Setup project directories and configurations"""
    config = load_config(config_path)
    create_directories(config)
    typer.echo(f"Project setup completed. Configuration loaded from {config_path}")
    
    # Save a copy of the config for reference
    with open("config_used.json", "w") as f:
        json.dump(config.dict(), f, indent=2)

@app.command()
def download(config_path: str = "hyper.json"):
    """Download MNIST dataset"""
    config = load_config(config_path)
    typer.echo("Downloading MNIST dataset...")
    
    start_time = time.time()
    download_mnist_train(config)
    download_mnist_test(config)
    duration = time.time() - start_time
    
    typer.echo(f"Download completed in {duration:.2f} seconds")

@app.command()
def create_scrambled(config_path: str = "hyper.json"):
    """Create scrambled copies of MNIST images"""
    config = load_config(config_path)
    typer.echo("Creating scrambled copies of MNIST images...")
    
    start_time = time.time()
    scramble(config)
    duration = time.time() - start_time
    
    typer.echo(f"Scrambling completed in {duration:.2f} seconds")

@app.command()
def create_augmented(config_path: str = "hyper.json"):
    """Create augmented dataset by combining original and scrambled data"""
    config = load_config(config_path)
    typer.echo("Creating augmented dataset...")
    
    start_time = time.time()
    augment(config)
    duration = time.time() - start_time
    
    typer.echo(f"Augmentation completed in {duration:.2f} seconds")

@app.command()
def train_model(
    model_type: str = "robust",
    experiment_name: str = "default",
    config_path: str = "hyper.json"
):
    """Train model on specified dataset"""
    config = load_config(config_path)
    typer.echo(f"Training {model_type} model for experiment '{experiment_name}'...")
    
    start_time = time.time()
    model, metrics = train(model_type, experiment_name, config)
    duration = time.time() - start_time
    
    typer.echo(f"Training completed in {duration:.2f} seconds")
    typer.echo(f"Final validation accuracy: {metrics.metrics['val_accuracy'][-1]:.4f}")

@app.command()
def evaluate_model(
    model_path: str,
    experiment_name: str = "default",
    config_path: str = "hyper.json"
):
    """Evaluate trained model on test set"""
    config = load_config(config_path)
    model_path = Path(model_path)
    
    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}")
        raise typer.Exit(1)
    
    typer.echo(f"Evaluating model for experiment '{experiment_name}'...")
    
    start_time = time.time()
    results = evaluate(model_path, experiment_name, config)
    duration = time.time() - start_time
    
    typer.echo(f"Evaluation completed in {duration:.2f} seconds")
    typer.echo(f"Test accuracy: {results['test_accuracy']:.4f}")
    typer.echo(f"Adversarial accuracy (ε=0.1): {results['adversarial_robustness']['epsilon_0.1']['accuracy']:.4f}")

@app.command()
def compare(
    experiments: list[str] = ["original", "augmented"],
    config_path: str = "hyper.json"
):
    """Compare results from multiple experiments"""
    config = load_config(config_path)
    typer.echo(f"Comparing experiments: {', '.join(experiments)}")
    
    results = compare_experiments(experiments, config)
    
    typer.echo("\nComparison Summary:")
    typer.echo(f"Fastest convergence: {results['convergence_speed']['fastest']} (epoch {results['convergence_speed']['epochs']})")
    typer.echo(f"Most robust to adversarial examples: {results['adversarial_robustness']['most_robust']} (accuracy {results['adversarial_robustness']['accuracy']:.4f})")
    typer.echo(f"Best OOD performance: {results['ood_performance']['best']} (accuracy {results['ood_performance']['accuracy']:.4f})")
    typer.echo(f"Best test accuracy: {results['test_accuracy']['best']} (accuracy {results['test_accuracy']['accuracy']:.4f})")

@app.command()
def run_all(config_path: str = "hyper.json"):
    """Run the full experiment pipeline"""
    config = load_config(config_path)
    
    # Setup
    typer.echo("Setting up project...")
    create_directories(config)
    
    # Download data
    typer.echo("Downloading MNIST dataset...")
    download_mnist_train(config)
    download_mnist_test(config)
    
    # Create scrambled data
    typer.echo("Creating scrambled copies...")
    scramble(config)
    
    # Create augmented dataset
    typer.echo("Creating augmented dataset...")
    augment(config)
    
    # Train original model
    typer.echo("Training model on original data...")
    train("robust", "original", config)
    
    # Train augmented model
    typer.echo("Training model on augmented data...")
    train("robust", "augmented", config)
    
    # Evaluate models
    typer.echo("Evaluating models...")
    original_model_path = config.model.model_dir / "original.h5"
    augmented_model_path = config.model.model_dir / "augmented.h5"
    
    evaluate(original_model_path, "original", config)
    evaluate(augmented_model_path, "augmented", config)
    
    # Compare results
    typer.echo("Comparing experiments...")
    compare_experiments(["original", "augmented"], config)
    
    typer.echo("Full experiment pipeline completed")

if __name__ == "__main__":
    app()