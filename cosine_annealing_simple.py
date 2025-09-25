#!/usr/bin/env python3
"""
Cosine Annealing Experiments - Simple Approach
Run multiple training sessions with different configurations
"""

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

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
    
    config_file = f"temp_config_{config_name.lower().replace(' ', '_')}.py"
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
    
    # Parse results
    lines = result.stdout.split('\n')
    final_results = {}
    
    for line in lines:
        if "Val Loss:" in line and "Val Accuracy:" in line and "Val Perplexity:" in line:
            parts = line.split()
            final_results = {
                'val_loss': float(parts[2]),
                'val_accuracy': float(parts[5]),
                'val_perplexity': float(parts[8])
            }
            break
    
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

def main():
    """Run cosine annealing experiments"""
    print("🔬 Cosine Annealing Learning Rate Schedule Experiments")
    print("=" * 60)
    
    # Create experiments directory
    output_dir = Path("experiments/cosine_annealing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create LR schedule visualization
    create_lr_schedule_plot(output_dir)
    
    # Run experiments
    results = []
    
    # For now, let's run the baseline to confirm it works
    # We can add more experiments once we confirm the basic setup works
    baseline_result = run_training_with_config("Baseline_Current", max_steps=200, eval_every=20)
    if baseline_result:
        results.append(baseline_result)
    
    # Save results
    with open(output_dir / 'cosine_annealing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 {len([r for r in results if r['status'] == 'completed'])} experiments completed successfully")

if __name__ == "__main__":
    main()
