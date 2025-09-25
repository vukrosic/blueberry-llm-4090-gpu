#!/usr/bin/env python3
"""
Simple Learning Rate Ablation Study

This script runs a simple ablation study by modifying the existing training script
with different learning rate combinations.
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


def run_learning_rate_ablation():
    """Run ablation study on learning rates"""
    
    # Define learning rate combinations to test
    lr_combinations = [
        {"muon_lr": 0.001, "name": "very_low"},
        {"muon_lr": 0.01, "name": "low"},
        {"muon_lr": 0.03, "name": "medium"},
        {"muon_lr": 0.1, "name": "high"},
        {"muon_lr": 0.3, "name": "very_high"},
    ]
    
    print("🧪 Learning Rate Ablation Study")
    print(f"📊 Testing {len(lr_combinations)} learning rate combinations")
    
    # Load data once
    print("\n📊 Loading dataset...")
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ablation_results/lr_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, lr_config in enumerate(lr_combinations):
        print(f"\n{'='*60}")
        print(f"🧪 Experiment {i+1}/{len(lr_combinations)}: {lr_config['name']}")
        print(f"📋 Muon LR: {lr_config['muon_lr']:.3f}")
        print(f"{'='*60}")
        
        try:
            # Create config with specific learning rate
            config = MoEModelConfig(
                vocab_size=vocab_size,
                muon_lr=lr_config['muon_lr'],
                max_steps=50,  # Reduced for faster ablation
                eval_every=10,
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
            
            # Store results (extract only serializable metrics)
            result = {
                'experiment_name': lr_config['name'],
                'muon_lr': lr_config['muon_lr'],
                'training_time': training_time,
                'final_loss': training_results.get('val_loss', 0),
                'final_accuracy': training_results.get('val_accuracy', 0),
                'final_perplexity': training_results.get('val_perplexity', 0),
                'status': 'completed'
            }
            results.append(result)
            
            print(f"✅ Experiment {lr_config['name']} completed successfully")
            print(f"⏱️  Training time: {training_time:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Experiment {lr_config['name']} failed: {str(e)}")
            error_result = {
                'experiment_name': lr_config['name'],
                'muon_lr': lr_config['muon_lr'],
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
        print(f"\n📊 Ablation Study Summary:")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(results) - len(successful_results)}")
        
        # Find best performing experiment
        best_result = max(successful_results, key=lambda x: x['final_accuracy'])
        print(f"\n🎯 Best Performance:")
        print(f"   Learning Rate: {best_result['muon_lr']:.3f}")
        print(f"   Final Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"   Final Loss: {best_result['final_loss']:.4f}")
        
        # Generate simple report
        report = {
            'summary': {
                'total_experiments': len(results),
                'successful_experiments': len(successful_results),
                'best_lr': best_result['muon_lr'],
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


def main():
    """Main function"""
    try:
        output_dir = run_learning_rate_ablation()
        print(f"\n🎉 Ablation study completed!")
        print(f"📁 Results directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Study interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during study: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
