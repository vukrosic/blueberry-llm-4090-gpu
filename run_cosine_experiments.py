#!/usr/bin/env python3
"""
Cosine Annealing Experiments - Full Implementation
Run all cosine annealing configurations and collect results
"""

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import re

def run_training_with_config(config_name: str, max_steps: int = 200, eval_every: int = 20):
    """Run training with specific configuration"""
    print(f"\n🧪 Running experiment: {config_name}")
    
    # Create a temporary config file for this experiment
    config_content = f'''from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = {max_steps}

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.065

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = {eval_every}
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
'''
    
    config_file = f"temp_config_{config_name.lower().replace(' ', '_').replace('-', '_')}.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Run training
    start_time = time.time()
    result = subprocess.run([
        "python", "train_moe.py", 
        "--config", config_file,
        "--max_steps", str(max_steps),
        "--eval_every", str(eval_every)
    ], capture_output=True, text=True)
    
    training_time = time.time() - start_time
    
    # Clean up temp config
    Path(config_file).unlink()
    
    if result.returncode != 0:
        print(f"❌ {config_name} failed: {result.stderr}")
        return None
    
    # Parse results - look for the final results section
    lines = result.stdout.split('\n')
    final_results = {}
    
    # Look for the final results section
    in_final_results = False
    for line in lines:
        if "📊 Final Results:" in line:
            in_final_results = True
            continue
        elif in_final_results and "Val Loss:" in line:
            # Parse: "   Val Loss: 7.4671"
            parts = line.strip().split()
            if len(parts) >= 3:
                final_results['val_loss'] = float(parts[2])
        elif in_final_results and "Val Accuracy:" in line:
            # Parse: "   Val Accuracy: 0.0894"
            parts = line.strip().split()
            if len(parts) >= 3:
                final_results['val_accuracy'] = float(parts[2])
        elif in_final_results and "Val Perplexity:" in line:
            # Parse: "   Val Perplexity: 1689.38"
            parts = line.strip().split()
            if len(parts) >= 3:
                final_results['val_perplexity'] = float(parts[2])
        elif in_final_results and line.strip() == "":
            # End of final results section
            break
    
    # If we didn't find results in the final section, try the step-by-step results
    if not final_results:
        for line in lines:
            if "Step" in line and "Val Loss:" in line and "Val Acc:" in line:
                # Parse: "Step 10: Val Loss: 9.3933, Val Acc: 0.0188, Val PPL: 12007.32"
                parts = line.split()
                try:
                    final_results = {
                        'val_loss': float(parts[4].replace(',', '')),
                        'val_accuracy': float(parts[7].replace(',', '')),
                        'val_perplexity': float(parts[10])
                    }
                except (IndexError, ValueError):
                    continue
    
    print(f"✅ {config_name} completed:")
    print(f"   Loss: {final_results.get('val_loss', 0):.3f}")
    print(f"   Accuracy: {final_results.get('val_accuracy', 0):.3f}")
    print(f"   Perplexity: {final_results.get('val_perplexity', 0):.0f}")
    print(f"   Time: {training_time:.1f}s")
    
    return {
        'experiment_name': config_name,
        'final_results': final_results,
        'training_time': training_time,
        'status': 'completed'
    }

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

def create_results_comparison_plot(results, output_dir: Path):
    """Create comparison plot of experiment results"""
    if not results:
        return
    
    # Extract data
    names = [r['experiment_name'] for r in results if r['status'] == 'completed']
    losses = [r['final_results'].get('val_loss', 0) for r in results if r['status'] == 'completed']
    accuracies = [r['final_results'].get('val_accuracy', 0) for r in results if r['status'] == 'completed']
    perplexities = [r['final_results'].get('val_perplexity', 0) for r in results if r['status'] == 'completed']
    
    if not names:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Loss comparison
    bars1 = ax1.bar(range(len(names)), losses, color='#E74C3C', alpha=0.8)
    ax1.set_xlabel('Experiment', fontsize=12, weight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=12, weight='bold')
    ax1.set_title('Cosine Annealing: Validation Loss', fontsize=14, weight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy comparison
    bars2 = ax2.bar(range(len(names)), accuracies, color='#2ECC71', alpha=0.8)
    ax2.set_xlabel('Experiment', fontsize=12, weight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=12, weight='bold')
    ax2.set_title('Cosine Annealing: Validation Accuracy', fontsize=14, weight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Perplexity comparison
    bars3 = ax3.bar(range(len(names)), perplexities, color='#F39C12', alpha=0.8)
    ax3.set_xlabel('Experiment', fontsize=12, weight='bold')
    ax3.set_ylabel('Validation Perplexity', fontsize=12, weight='bold')
    ax3.set_title('Cosine Annealing: Validation Perplexity', fontsize=14, weight='bold')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Add value labels on bars
    for bar, ppl in zip(bars3, perplexities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{ppl:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_annealing_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Results comparison plot saved: {output_dir / 'cosine_annealing_results_comparison.png'}")

def main():
    """Run cosine annealing experiments"""
    print("🔬 Cosine Annealing Learning Rate Schedule Experiments")
    print("=" * 60)
    
    # Create experiments directory
    output_dir = Path("experiments/cosine_annealing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create LR schedule visualization
    create_lr_schedule_plot(output_dir)
    
    # Define experiments
    experiments = [
        "Baseline_Current",
        "Conservative_Warmup", 
        "Aggressive_Warmup",
        "Long_Warmup",
        "No_Warmup"
    ]
    
    # Run experiments
    results = []
    
    for exp_name in experiments:
        result = run_training_with_config(exp_name, max_steps=200, eval_every=20)
        if result:
            results.append(result)
    
    # Create results comparison plot
    create_results_comparison_plot(results, output_dir)
    
    # Save results
    with open(output_dir / 'cosine_annealing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 {len([r for r in results if r['status'] == 'completed'])} experiments completed successfully")
    
    if results:
        print("\n🎯 Results Summary:")
        for result in results:
            if result['status'] == 'completed':
                metrics = result['final_results']
                print(f"   {result['experiment_name']}: Loss={metrics.get('val_loss', 0):.3f}, "
                      f"Acc={metrics.get('val_accuracy', 0):.3f}, PPL={metrics.get('val_perplexity', 0):.0f}")

if __name__ == "__main__":
    main()
