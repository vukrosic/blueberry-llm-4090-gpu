"""
Validation Script: Demonstrate ablation study improvements
Compare baseline vs optimized configuration side-by-side
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.ablation_study.optimized_config import get_optimized_config


def run_validation_comparison():
    """Run side-by-side comparison of baseline vs optimized configuration"""
    print("🔬 Ablation Study Validation: Baseline vs Optimized")
    print("="*70)
    
    # System info
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data once
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
    
    # Define configurations
    baseline_config = MoEModelConfig(vocab_size=vocab_size)
    optimized_config = get_optimized_config(vocab_size)
    
    results = {}
    
    # Test configurations
    configs = [
        ("Baseline", baseline_config),
        ("Optimized", optimized_config)
    ]
    
    for config_name, config in configs:
        print(f"\n{'='*70}")
        print(f"🧪 Testing {config_name} Configuration")
        print(f"{'='*70}")
        
        # Show key parameters
        print(f"📋 Key Parameters:")
        print(f"   • Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
        print(f"   • Learning Rate: {config.muon_lr}")
        print(f"   • Dropout: {config.dropout}")
        print(f"   • Weight Decay: {config.weight_decay}")
        print(f"   • Number of Experts: {config.num_experts}")
        print(f"   • Expert Top-K: {config.expert_top_k}")
        
        # Set seed for fair comparison
        set_seed(42)
        
        # Train model
        start_time = time.time()
        try:
            model, final_metrics = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            results[config_name] = {
                'val_loss': final_metrics['val_loss'],
                'val_accuracy': final_metrics['val_accuracy'],
                'val_perplexity': final_metrics['val_perplexity'],
                'training_time': training_time,
                'success': True
            }
            
            print(f"\n✅ {config_name} Results:")
            print(f"   📉 Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   📈 Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   📊 Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
            print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ {config_name} failed: {str(e)}")
            results[config_name] = {'success': False, 'error': str(e)}
        
        # Brief pause between runs
        time.sleep(1)
    
    # Compare results
    if all(r['success'] for r in results.values()):
        print(f"\n📊 COMPARISON RESULTS")
        print(f"{'='*70}")
        
        baseline = results['Baseline']
        optimized = results['Optimized']
        
        print(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement'}")
        print(f"{'-'*70}")
        
        # Validation Loss (lower is better)
        loss_improvement = ((baseline['val_loss'] - optimized['val_loss']) / baseline['val_loss']) * 100
        print(f"{'Validation Loss':<20} {baseline['val_loss']:<15.4f} {optimized['val_loss']:<15.4f} {loss_improvement:+.1f}%")
        
        # Validation Accuracy (higher is better)
        acc_improvement = ((optimized['val_accuracy'] - baseline['val_accuracy']) / baseline['val_accuracy']) * 100
        print(f"{'Validation Accuracy':<20} {baseline['val_accuracy']:<15.4f} {optimized['val_accuracy']:<15.4f} {acc_improvement:+.1f}%")
        
        # Perplexity (lower is better)
        ppl_improvement = ((baseline['val_perplexity'] - optimized['val_perplexity']) / baseline['val_perplexity']) * 100
        print(f"{'Validation PPL':<20} {baseline['val_perplexity']:<15.2f} {optimized['val_perplexity']:<15.2f} {ppl_improvement:+.1f}%")
        
        # Training Time
        time_diff = optimized['training_time'] - baseline['training_time']
        print(f"{'Training Time (min)':<20} {baseline['training_time']/60:<15.1f} {optimized['training_time']/60:<15.1f} {time_diff/60:+.1f}")
        
        print(f"\n🎯 SUMMARY")
        print(f"{'='*70}")
        if loss_improvement > 0:
            print(f"✅ SUCCESS: Optimized configuration achieves {loss_improvement:.1f}% better validation loss")
            print(f"✅ BONUS: {acc_improvement:.1f}% improvement in validation accuracy")
            print(f"✅ EFFICIENCY: {ppl_improvement:.1f}% reduction in perplexity")
            
            if abs(time_diff) < 10:  # Less than 10 seconds difference
                print(f"✅ SPEED: Training time unchanged (within measurement error)")
            else:
                print(f"⏱️ TIME: Training time change: {time_diff/60:+.1f} minutes")
        else:
            print(f"❌ UNEXPECTED: Optimized configuration shows {abs(loss_improvement):.1f}% worse validation loss")
            print(f"   This may indicate experimental variation or configuration issues")
    
    else:
        print(f"\n❌ COMPARISON FAILED: One or more configurations failed to run")
        for name, result in results.items():
            if not result['success']:
                print(f"   • {name}: {result.get('error', 'Unknown error')}")
    
    print(f"\n🔬 Ablation study validation complete!")


if __name__ == "__main__":
    run_validation_comparison()
