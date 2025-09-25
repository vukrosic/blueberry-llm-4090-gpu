#!/usr/bin/env python3
"""
Extended Learning Rate Research Script

This script continues the learning rate research by exploring additional
learning rates around the optimal range and testing different configurations.
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed


def run_extended_lr_research():
    """Run extended learning rate research around optimal range"""
    
    # Define extended learning rate combinations around the optimal range (0.1)
    lr_combinations = [
        # Fine-grained around optimal
        {"muon_lr": 0.05, "name": "optimal_low"},
        {"muon_lr": 0.07, "name": "optimal_med_low"},
        {"muon_lr": 0.08, "name": "optimal_med"},
        {"muon_lr": 0.12, "name": "optimal_high"},
        {"muon_lr": 0.15, "name": "optimal_very_high"},
        
        # Test different momentum values with optimal LR
        {"muon_lr": 0.1, "momentum": 0.9, "name": "optimal_lr_mom_09"},
        {"muon_lr": 0.1, "momentum": 0.99, "name": "optimal_lr_mom_99"},
        
        # Test different AdamW ratios
        {"muon_lr": 0.1, "adamw_ratio": 0.05, "name": "optimal_lr_adamw_005"},
        {"muon_lr": 0.1, "adamw_ratio": 0.2, "name": "optimal_lr_adamw_02"},
        
        # Test different weight decay
        {"muon_lr": 0.1, "weight_decay": 0.05, "name": "optimal_lr_wd_005"},
        {"muon_lr": 0.1, "weight_decay": 0.2, "name": "optimal_lr_wd_02"},
    ]
    
    print("🔬 Extended Learning Rate Research")
    print(f"📊 Testing {len(lr_combinations)} extended configurations")
    
    # Load data once
    print("\n📊 Loading dataset...")
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ablation_results/extended_lr_research_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, lr_config in enumerate(lr_combinations):
        print(f"\n{'='*60}")
        print(f"🔬 Experiment {i+1}/{len(lr_combinations)}: {lr_config['name']}")
        print(f"📋 Muon LR: {lr_config['muon_lr']:.3f}")
        if 'momentum' in lr_config:
            print(f"📋 Momentum: {lr_config['momentum']:.2f}")
        if 'adamw_ratio' in lr_config:
            print(f"📋 AdamW Ratio: {lr_config['adamw_ratio']:.2f}")
        if 'weight_decay' in lr_config:
            print(f"📋 Weight Decay: {lr_config['weight_decay']:.2f}")
        print(f"{'='*60}")
        
        try:
            # Create config with specific parameters
            config_params = {
                'vocab_size': vocab_size,
                'muon_lr': lr_config['muon_lr'],
                'max_steps': 100,  # More steps for better evaluation
                'eval_every': 20,
                'batch_size': 16
            }
            
            # Add custom parameters if specified
            if 'momentum' in lr_config:
                config_params['momentum'] = lr_config['momentum']
            if 'adamw_ratio' in lr_config:
                config_params['adamw_ratio'] = lr_config['adamw_ratio']
            if 'weight_decay' in lr_config:
                config_params['weight_decay'] = lr_config['weight_decay']
            
            config = MoEModelConfig(**config_params)
            
            # Create dataset
            dataset = TextTokenDataset(tokens, config.max_seq_len)
            
            # Train/val split
            val_size = len(dataset) // 10
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
            
            # Train model
            start_time = time.time()
            model, training_results = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Store results
            result = {
                'experiment_name': lr_config['name'],
                'muon_lr': lr_config['muon_lr'],
                'momentum': lr_config.get('momentum', 0.95),
                'adamw_ratio': lr_config.get('adamw_ratio', 0.1),
                'weight_decay': lr_config.get('weight_decay', 0.1),
                'training_time': training_time,
                'final_loss': training_results.get('val_loss', 0),
                'final_accuracy': training_results.get('val_accuracy', 0),
                'final_perplexity': training_results.get('val_perplexity', 0),
                'status': 'completed'
            }
            results.append(result)
            
            print(f"✅ Experiment {lr_config['name']} completed successfully")
            print(f"⏱️  Training time: {training_time:.1f} seconds")
            print(f"📊 Final Accuracy: {result['final_accuracy']:.4f}")
            print(f"📉 Final Loss: {result['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Experiment {lr_config['name']} failed: {str(e)}")
            error_result = {
                'experiment_name': lr_config['name'],
                'muon_lr': lr_config['muon_lr'],
                'momentum': lr_config.get('momentum', 0.95),
                'adamw_ratio': lr_config.get('adamw_ratio', 0.1),
                'weight_decay': lr_config.get('weight_decay', 0.1),
                'error': str(e),
                'status': 'failed'
            }
            results.append(error_result)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    successful_results = [r for r in results if r.get('status') == 'completed']
    
    if successful_results:
        print(f"\n📊 Extended Research Summary:")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(results) - len(successful_results)}")
        
        # Find best performing experiment
        best_result = max(successful_results, key=lambda x: x['final_accuracy'])
        print(f"\n🎯 Best Performance:")
        print(f"   Learning Rate: {best_result['muon_lr']:.3f}")
        print(f"   Momentum: {best_result['momentum']:.2f}")
        print(f"   AdamW Ratio: {best_result['adamw_ratio']:.2f}")
        print(f"   Weight Decay: {best_result['weight_decay']:.2f}")
        print(f"   Final Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"   Final Loss: {best_result['final_loss']:.4f}")
        
        # Generate comprehensive report
        report = {
            'summary': {
                'total_experiments': len(results),
                'successful_experiments': len(successful_results),
                'best_lr': best_result['muon_lr'],
                'best_momentum': best_result['momentum'],
                'best_adamw_ratio': best_result['adamw_ratio'],
                'best_weight_decay': best_result['weight_decay'],
                'best_accuracy': best_result['final_accuracy'],
                'best_loss': best_result['final_loss']
            },
            'all_results': results
        }
        
        report_file = output_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📄 Report: {report_file}")
        
    else:
        print(f"\n❌ No successful experiments completed")
    
    return output_dir


def run_learning_rate_sweep():
    """Run a fine-grained learning rate sweep around the optimal range"""
    
    print("\n🔍 Fine-Grained Learning Rate Sweep")
    
    # Create a fine-grained sweep around 0.1
    lr_values = np.linspace(0.06, 0.14, 9)  # 9 values around optimal
    
    print(f"📊 Testing {len(lr_values)} learning rates: {lr_values}")
    
    # Load data
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ablation_results/lr_sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, lr in enumerate(lr_values):
        print(f"\n{'='*50}")
        print(f"🔍 Sweep {i+1}/{len(lr_values)}: LR = {lr:.3f}")
        print(f"{'='*50}")
        
        try:
            config = MoEModelConfig(
                vocab_size=vocab_size,
                muon_lr=lr,
                max_steps=75,  # Moderate steps for sweep
                eval_every=15,
                batch_size=16
            )
            
            # Create dataset
            dataset = TextTokenDataset(tokens, config.max_seq_len)
            
            # Train/val split
            val_size = len(dataset) // 10
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
            
            # Train model
            start_time = time.time()
            model, training_results = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Store results
            result = {
                'learning_rate': lr,
                'training_time': training_time,
                'final_loss': training_results.get('val_loss', 0),
                'final_accuracy': training_results.get('val_accuracy', 0),
                'final_perplexity': training_results.get('val_perplexity', 0),
                'status': 'completed'
            }
            results.append(result)
            
            print(f"✅ LR {lr:.3f} completed - Acc: {result['final_accuracy']:.4f}, Loss: {result['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ LR {lr:.3f} failed: {str(e)}")
            error_result = {
                'learning_rate': lr,
                'error': str(e),
                'status': 'failed'
            }
            results.append(error_result)
    
    # Save results
    results_file = output_dir / "sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best LR from sweep
    successful_results = [r for r in results if r.get('status') == 'completed']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['final_accuracy'])
        print(f"\n🎯 Best LR from sweep: {best_result['learning_rate']:.3f}")
        print(f"📊 Best Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"📉 Best Loss: {best_result['final_loss']:.4f}")
    
    return output_dir


def main():
    """Main function"""
    try:
        print("🚀 Starting Extended Learning Rate Research")
        
        # Run extended research
        extended_dir = run_extended_lr_research()
        
        # Run fine-grained sweep
        sweep_dir = run_learning_rate_sweep()
        
        print(f"\n🎉 Extended research completed!")
        print(f"📁 Extended results: {extended_dir}")
        print(f"📁 Sweep results: {sweep_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
