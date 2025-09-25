#!/usr/bin/env python3
"""
Simple Cosine Annealing Experiments
Run individual experiments to test different cosine annealing configurations
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from pathlib import Path

# Import our modules
from configs.moe_config import MoEModelConfig
from training.trainer import train_moe_model
from data.loader import load_and_cache_data
from torch.utils.data import DataLoader, random_split

def create_lr_schedule_plot(output_dir: Path):
    """Create visualization of different learning rate schedules"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    max_steps = 200
    steps = np.arange(max_steps)
    
    # Define different cosine annealing configurations
    configs = [
        {"name": "Baseline (5% warmup, 10% min)", "warmup": 0.05, "min_lr": 0.1, "color": "#E74C3C"},
        {"name": "Conservative (10% warmup, 5% min)", "warmup": 0.1, "min_lr": 0.05, "color": "#2ECC71"},
        {"name": "Aggressive (2% warmup, 20% min)", "warmup": 0.02, "min_lr": 0.2, "color": "#F39C12"},
        {"name": "Long Warmup (20% warmup)", "warmup": 0.2, "min_lr": 0.1, "color": "#9B59B6"},
        {"name": "No Warmup (0% warmup)", "warmup": 0.0, "min_lr": 0.1, "color": "#34495E"},
    ]
    
    for config in configs:
        warmup_steps = int(max_steps * config["warmup"])
        lrs = []
        
        for step in steps:
            if step < warmup_steps:
                # Warmup phase
                lr = step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                # Cosine annealing phase
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr = config["min_lr"] + (1 - config["min_lr"]) * cosine_factor
            
            lrs.append(lr)
        
        ax.plot(steps, lrs, label=config["name"], linewidth=2, alpha=0.8, color=config["color"])
    
    ax.set_xlabel('Training Steps', fontsize=12, weight='bold')
    ax.set_ylabel('Learning Rate (normalized)', fontsize=12, weight='bold')
    ax.set_title('Cosine Annealing Learning Rate Schedules', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_annealing_schedules.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 LR schedule plot saved: {output_dir / 'cosine_annealing_schedules.png'}")

def run_single_experiment(config: MoEModelConfig, experiment_name: str, train_loader: DataLoader, val_loader: DataLoader):
    """Run a single experiment"""
    print(f"\n🧪 Running experiment: {experiment_name}")
    
    start_time = time.time()
    
    try:
        model, final_metrics = train_moe_model(config, train_loader, val_loader)
        training_time = time.time() - start_time
        
        print(f"✅ {experiment_name} completed:")
        print(f"   Loss: {final_metrics.get('val_loss', 0):.3f}")
        print(f"   Accuracy: {final_metrics.get('val_accuracy', 0):.3f}")
        print(f"   Perplexity: {final_metrics.get('val_perplexity', 0):.0f}")
        print(f"   Time: {training_time:.1f}s")
        
        return {
            'experiment_name': experiment_name,
            'final_metrics': final_metrics,
            'training_time': training_time,
            'status': 'completed'
        }
    except Exception as e:
        print(f"❌ {experiment_name} failed: {e}")
        return {
            'experiment_name': experiment_name,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Run cosine annealing experiments"""
    print("🔬 Cosine Annealing Learning Rate Schedule Experiments")
    print("=" * 60)
    
    # Create experiments directory
    output_dir = Path("experiments/cosine_annealing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📦 Loading data...")
    config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(config)
    
    # Create data loaders
    dataset_size = len(tokens)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_tokens, val_tokens = random_split(tokens, [train_size, val_size])
    
    train_loader = DataLoader(train_tokens, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_tokens, batch_size=24, shuffle=False)
    
    print(f"✅ Data loaded: {len(train_tokens)} train, {len(val_tokens)} val samples")
    print(f"✅ Vocab size: {config.vocab_size}")
    
    # Create LR schedule visualization
    create_lr_schedule_plot(output_dir)
    
    # Configure for longer training
    config.max_steps = 200
    config.eval_every = 20
    
    # Run experiments
    results = []
    
    # Experiment 1: Baseline (current implementation)
    results.append(run_single_experiment(config, "Baseline_Current", train_loader, val_loader))
    
    # For now, let's just run the baseline to see if it works
    # We can add more experiments once we confirm the basic setup works
    
    # Save results
    with open(output_dir / 'cosine_annealing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 {len([r for r in results if r['status'] == 'completed'])} experiments completed successfully")

if __name__ == "__main__":
    main()
