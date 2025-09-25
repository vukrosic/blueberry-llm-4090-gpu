"""
Quick Ablation Test - Run a subset of experiments to validate the framework
"""

import torch
import time
import json
import os
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.ablation_study.ablation_configs import get_baseline_config, get_ablation_experiments


def run_quick_test():
    """Run a quick test with 3 key experiments"""
    print("🧪 Quick Ablation Test - Testing Framework")
    print("="*50)
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n📊 Loading data...")
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create dataset
    dataset = TextTokenDataset(tokens, temp_config.max_seq_len)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=temp_config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=temp_config.batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Get baseline and a few key experiments
    baseline_config = get_baseline_config(vocab_size)
    all_experiments = get_ablation_experiments(vocab_size)
    
    # Select 3 key experiments for quick test
    test_experiments = [
        baseline_config,
        all_experiments[0].config,  # grad_accum_2
        all_experiments[2].config,  # moe_top1
        all_experiments[6].config,  # dropout_005
    ]
    
    test_names = ["baseline", "grad_accum_2", "moe_top1", "dropout_005"]
    
    results = []
    
    for i, (config, name) in enumerate(zip(test_experiments, test_names)):
        print(f"\n{'='*60}")
        print(f"🧪 Running Test {i+1}/4: {name}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Train model
        start_time = time.time()
        try:
            model, final_metrics = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            result = {
                'experiment_name': name,
                'val_loss': final_metrics['val_loss'],
                'val_accuracy': final_metrics['val_accuracy'],
                'val_perplexity': final_metrics['val_perplexity'],
                'training_time_minutes': training_time / 60,
                'success': True
            }
            
            print(f"\n✅ Test {name} completed successfully!")
            print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Test {name} failed: {str(e)}")
            result = {
                'experiment_name': name,
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'val_perplexity': float('inf'),
                'training_time_minutes': 0,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
        
        # Brief pause between experiments
        time.sleep(1)
    
    # Analyze results
    print(f"\n📊 QUICK TEST RESULTS")
    print(f"{'='*60}")
    print(f"{'Experiment':<15} {'Val Loss':<10} {'Val Acc':<10} {'Val PPL':<10} {'Time (min)':<10}")
    print(f"{'-'*60}")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: x['val_loss'])
        
        for result in successful_results:
            print(f"{result['experiment_name']:<15} {result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} {result['val_perplexity']:<10.2f} {result['training_time_minutes']:<10.1f}")
        
        best_result = successful_results[0]
        print(f"\n🏆 BEST TEST: {best_result['experiment_name']}")
        print(f"   Validation Loss: {best_result['val_loss']:.4f}")
        print(f"   Improvement over baseline: {((successful_results[-1]['val_loss'] - best_result['val_loss']) / successful_results[-1]['val_loss'] * 100):.1f}%")
    
    # Save results
    os.makedirs("experiments/ablation_study/results", exist_ok=True)
    results_file = "experiments/ablation_study/results/quick_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n🎉 Quick test completed! Framework is ready for full ablation study.")


if __name__ == "__main__":
    run_quick_test()
