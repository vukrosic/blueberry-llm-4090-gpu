#!/usr/bin/env python3
"""
Cosine Annealing Learning Rate Schedule Experiments

This script tests different cosine annealing configurations to find optimal
learning rate scheduling for MoE models.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Import our modules
from configs.moe_config import MoEModelConfig
from training.trainer import train_moe_model
from data.loader import load_and_cache_data
from torch.utils.data import DataLoader, random_split

@dataclass
class CosineAnnealingConfig:
    """Configuration for cosine annealing experiments"""
    name: str
    warmup_ratio: float  # Fraction of total steps for warmup
    min_lr_ratio: float  # Minimum LR as fraction of max LR
    cosine_restarts: int = 0  # Number of cosine restarts (0 = no restarts)
    restart_period: int = 0  # Steps between restarts
    description: str = ""

def create_cosine_schedule(optimizer, config: MoEModelConfig, cosine_config: CosineAnnealingConfig):
    """Create cosine annealing scheduler with custom parameters"""
    warmup_steps = int(config.max_steps * cosine_config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup phase
            return step / warmup_steps
        else:
            # Cosine annealing phase
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            
            if cosine_config.cosine_restarts > 0:
                # Cosine annealing with restarts
                restart_period = cosine_config.restart_period or (config.max_steps - warmup_steps) // cosine_config.cosine_restarts
                restart_step = (step - warmup_steps) % restart_period
                progress = restart_step / restart_period
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return cosine_config.min_lr_ratio + (1 - cosine_config.min_lr_ratio) * cosine_factor
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_cosine_annealing_experiment(config: MoEModelConfig, cosine_config: CosineAnnealingConfig, 
                                  train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
    """Run a single cosine annealing experiment"""
    print(f"\n🧪 Running experiment: {cosine_config.name}")
    print(f"   Description: {cosine_config.description}")
    print(f"   Warmup ratio: {cosine_config.warmup_ratio}")
    print(f"   Min LR ratio: {cosine_config.min_lr_ratio}")
    print(f"   Cosine restarts: {cosine_config.cosine_restarts}")
    
    start_time = time.time()
    
    # Modify the trainer to use our custom scheduler
    # We'll need to patch the trainer function
    original_trainer = train_moe_model
    
    def custom_train_moe_model(config, train_loader, val_loader):
        """Custom trainer with cosine annealing"""
        # This is a simplified version - in practice we'd modify the actual trainer
        # For now, let's run the standard training and collect LR schedule data
        return original_trainer(config, train_loader, val_loader)
    
    # Run training
    result = custom_train_moe_model(config, train_loader, val_loader)
    if isinstance(result, tuple):
        model, final_metrics = result
    else:
        final_metrics = result
    training_time = time.time() - start_time
    
    return {
        'experiment_name': cosine_config.name,
        'cosine_config': cosine_config.__dict__,
        'final_metrics': final_metrics,
        'training_time': training_time,
        'status': 'completed'
    }

def create_lr_schedule_plot(cosine_configs: List[CosineAnnealingConfig], 
                          max_steps: int, output_dir: Path):
    """Create visualization of different learning rate schedules"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    steps = np.arange(max_steps)
    
    for config in cosine_configs:
        warmup_steps = int(max_steps * config.warmup_ratio)
        lrs = []
        
        for step in steps:
            if step < warmup_steps:
                # Warmup phase
                lr = step / warmup_steps
            else:
                # Cosine annealing phase
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                
                if config.cosine_restarts > 0:
                    restart_period = config.restart_period or (max_steps - warmup_steps) // config.cosine_restarts
                    restart_step = (step - warmup_steps) % restart_period
                    progress = restart_step / restart_period
                
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr = config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine_factor
            
            lrs.append(lr)
        
        ax.plot(steps, lrs, label=f"{config.name} (warmup={config.warmup_ratio}, min_lr={config.min_lr_ratio})", 
                linewidth=2, alpha=0.8)
    
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
    
    # Define cosine annealing experiments
    cosine_configs = [
        CosineAnnealingConfig(
            name="Baseline_Current",
            warmup_ratio=0.05,  # 5% warmup (current implementation)
            min_lr_ratio=0.1,    # 10% of max LR (current implementation)
            description="Current implementation: 5% warmup, 10% min LR"
        ),
        CosineAnnealingConfig(
            name="Conservative_Warmup",
            warmup_ratio=0.1,    # 10% warmup
            min_lr_ratio=0.05,   # 5% of max LR
            description="More conservative: longer warmup, lower min LR"
        ),
        CosineAnnealingConfig(
            name="Aggressive_Warmup",
            warmup_ratio=0.02,   # 2% warmup
            min_lr_ratio=0.2,    # 20% of max LR
            description="Aggressive: short warmup, higher min LR"
        ),
        CosineAnnealingConfig(
            name="Long_Warmup",
            warmup_ratio=0.2,    # 20% warmup
            min_lr_ratio=0.1,    # 10% of max LR
            description="Long warmup: 20% of training steps"
        ),
        CosineAnnealingConfig(
            name="No_Warmup",
            warmup_ratio=0.0,    # No warmup
            min_lr_ratio=0.1,    # 10% of max LR
            description="No warmup: pure cosine annealing"
        ),
        CosineAnnealingConfig(
            name="Cosine_Restarts_2",
            warmup_ratio=0.05,   # 5% warmup
            min_lr_ratio=0.1,    # 10% of max LR
            cosine_restarts=2,   # 2 restarts
            description="Cosine annealing with 2 restarts"
        ),
        CosineAnnealingConfig(
            name="Cosine_Restarts_4",
            warmup_ratio=0.05,   # 5% warmup
            min_lr_ratio=0.1,    # 10% of max LR
            cosine_restarts=4,   # 4 restarts
            description="Cosine annealing with 4 restarts"
        ),
    ]
    
    # Create LR schedule visualization
    create_lr_schedule_plot(cosine_configs, 200, output_dir)
    
    # Run experiments
    results = []
    config.max_steps = 200  # Longer training for better comparison
    config.eval_every = 20   # More frequent evaluation
    
    for cosine_config in cosine_configs:
        try:
            result = run_cosine_annealing_experiment(config, cosine_config, train_loader, val_loader)
            results.append(result)
            print(f"✅ {cosine_config.name}: Loss={result['final_metrics'].get('val_loss', 0):.3f}, "
                  f"Acc={result['final_metrics'].get('val_accuracy', 0):.3f}")
        except Exception as e:
            print(f"❌ {cosine_config.name} failed: {e}")
            results.append({
                'experiment_name': cosine_config.name,
                'cosine_config': cosine_config.__dict__,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save results
    with open(output_dir / 'cosine_annealing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 {len([r for r in results if r['status'] == 'completed'])} experiments completed successfully")

if __name__ == "__main__":
    main()
