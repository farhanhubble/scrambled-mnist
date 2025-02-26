import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
from config import load_config, ProjectConfig
from dataloader import create_test_dataset, generate_adversarial_examples
from metrics import MetricsTracker, evaluate_out_of_distribution

def evaluate(model_path, experiment_name="default", config: ProjectConfig = None):
    """Evaluate a trained model on the test dataset"""
    if config is None:
        config = load_config()
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test dataset
    test_ds = create_test_dataset(config)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)
    
    # Evaluate on test set
    print(f"Evaluating model on test set...")
    start_time = time.time()
    test_loss, test_accuracy = metrics_tracker.evaluate_test_performance(model, test_ds)
    evaluation_time = time.time() - start_time
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    # Evaluate adversarial robustness
    print(f"Evaluating adversarial robustness...")
    epsilon_values = [0.05, 0.1, 0.2]
    adv_results = {}
    
    for epsilon in epsilon_values:
        adv_loss, adv_accuracy = metrics_tracker.evaluate_adversarial_robustness(model, test_ds, epsilon)
        adv_results[f"epsilon_{epsilon}"] = {
            "accuracy": float(adv_accuracy),
            "loss": float(adv_loss)
        }
        print(f"Adversarial accuracy (ε={epsilon}): {adv_accuracy:.4f}")
    
    # Evaluate out-of-distribution performance
    print(f"Evaluating out-of-distribution performance...")
    ood_results = evaluate_out_of_distribution(model, config)
    for angle, accuracy in ood_results.items():
        print(f"OOD accuracy ({angle}): {accuracy:.4f}")
    
    # Compute confusion matrix
    metrics_tracker.compute_confusion_matrix(model, test_ds)
    metrics_tracker.plot_confusion_matrix(experiment_name)
    
    # Save all metrics
    metrics_tracker.save_metrics(experiment_name)
    
    # Create a comprehensive results summary
    results_summary = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "evaluation_time": evaluation_time,
        "adversarial_robustness": adv_results,
        "out_of_distribution": ood_results,
        "convergence_epoch": metrics_tracker.metrics["convergence_epoch"],
        "training_time": metrics_tracker.metrics["training_time"]
    }
    
    # Save results summary
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    summary_path = results_dir / f"{experiment_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results summary saved to {summary_path}")
    
    return results_summary

def compare_experiments(experiment_names, config: ProjectConfig = None):
    """Compare results from multiple experiments"""
    if config is None:
        config = load_config()
    
    results_dir = Path("./results")
    
    # Load results for each experiment
    results = {}
    for experiment in experiment_names:
        summary_path = results_dir / f"{experiment}_summary.json"
        
        if not summary_path.exists():
            print(f"Results for {experiment} not found")
            continue
        
        with open(summary_path, 'r') as f:
            results[experiment] = json.load(f)
    
    # Compare convergence speed
    convergence_epochs = {exp: data["convergence_epoch"] for exp, data in results.items() if "convergence_epoch" in data}
    fastest_convergence = min(convergence_epochs.items(), key=lambda x: x[1]) if convergence_epochs else None
    
    # Compare adversarial robustness
    adversarial_robustness = {}
    for exp, data in results.items():
        if "adversarial_robustness" in data:
            # Use epsilon=0.1 for comparison
            if "epsilon_0.1" in data["adversarial_robustness"]:
                adversarial_robustness[exp] = data["adversarial_robustness"]["epsilon_0.1"]["accuracy"]
    
    most_robust = max(adversarial_robustness.items(), key=lambda x: x[1]) if adversarial_robustness else None
    
    # Compare OOD performance
    ood_performance = {}
    for exp, data in results.items():
        if "out_of_distribution" in data:
            # Average over all rotations
            ood_acc = np.mean([acc for rot, acc in data["out_of_distribution"].items()])
            ood_performance[exp] = ood_acc
    
    best_ood = max(ood_performance.items(), key=lambda x: x[1]) if ood_performance else None
    
    # Compare test accuracy
    test_accuracy = {exp: data["test_accuracy"] for exp, data in results.items() if "test_accuracy" in data}
    best_accuracy = max(test_accuracy.items(), key=lambda x: x[1]) if test_accuracy else None
    
    # Create comparison report
    comparison = {
        "convergence_speed": {
            "fastest": fastest_convergence[0] if fastest_convergence else None,
            "epochs": fastest_convergence[1] if fastest_convergence else None,
            "all_values": convergence_epochs
        },
        "adversarial_robustness": {
            "most_robust": most_robust[0] if most_robust else None,
            "accuracy": most_robust[1] if most_robust else None,
            "all_values": adversarial_robustness
        },
        "ood_performance": {
            "best": best_ood[0] if best_ood else None,
            "accuracy": best_ood[1] if best_ood else None,
            "all_values": ood_performance
        },
        "test_accuracy": {
            "best": best_accuracy[0] if best_accuracy else None,
            "accuracy": best_accuracy[1] if best_accuracy else None,
            "all_values": test_accuracy
        }
    }
    
    # Save comparison
    comparison_path = results_dir / "experiments_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Experiments comparison saved to {comparison_path}")
    
    # Print summary
    print("\n--- Experiments Comparison ---")
    print(f"Fastest convergence: {fastest_convergence[0]} (epoch {fastest_convergence[1]})") if fastest_convergence else print("No convergence data available")
    print(f"Most robust to adversarial examples: {most_robust[0]} (accuracy {most_robust[1]:.4f})") if most_robust else print("No adversarial robustness data available")
    print(f"Best OOD performance: {best_ood[0]} (accuracy {best_ood[1]:.4f})") if best_ood else print("No OOD performance data available")
    print(f"Best test accuracy: {best_accuracy[0]} (accuracy {best_accuracy[1]:.4f})") if best_accuracy else print("No test accuracy data available")
    
    return comparison

if __name__ == "__main__":
    config = load_config()
    
    # Evaluate baseline model
    baseline_model_path = config.model.model_dir / "original.h5"
    if baseline_model_path.exists():
        evaluate(baseline_model_path, "original", config)
    
    # Evaluate augmented model
    augmented_model_path = config.model.model_dir / "augmented.h5"
    if augmented_model_path.exists():
        evaluate(augmented_model_path, "augmented", config)
    
    # Compare experiments
    compare_experiments(["original", "augmented"], config)