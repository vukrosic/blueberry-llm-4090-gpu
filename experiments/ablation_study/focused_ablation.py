"""
Focused Ablation Study - Run the most promising experiments
Based on quick test results, focusing on gradient accumulation and promising combinations
"""

import torch
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed


def get_focused_experiments(vocab_size: int):
    """Get focused set of experiments based on quick test insights"""
    experiments = []
    
    # Baseline
    baseline = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=4, muon_lr=0.065, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.1,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=8, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("baseline", baseline, "Current optimized baseline"))
    
    # Best from quick test: gradient accumulation = 2
    grad_accum_2 = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=2, muon_lr=0.065, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.1,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=8, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("grad_accum_2", grad_accum_2, "Lower gradient accumulation for faster updates"))
    
    # Combine best grad_accum with lower dropout
    grad_accum_2_dropout_005 = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=2, muon_lr=0.065, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.05,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=8, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("combo_grad_dropout", grad_accum_2_dropout_005, "Combine best grad_accum with lower dropout"))
    
    # Combine with higher learning rate
    grad_accum_2_lr_008 = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=2, muon_lr=0.08, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.1,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=8, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("combo_grad_lr", grad_accum_2_lr_008, "Combine best grad_accum with higher learning rate"))
    
    # Test with more experts
    grad_accum_2_more_experts = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=2, muon_lr=0.065, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.1,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=12, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("combo_grad_experts", grad_accum_2_more_experts, "Combine best grad_accum with more experts"))
    
    # Ultimate combination - all best parameters
    ultimate_combo = MoEModelConfig(
        d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24, max_steps=20,
        gradient_accumulation_steps=2, muon_lr=0.08, max_seq_len=512, num_documents=2000,
        max_tokens=500000, eval_every=10, eval_steps=100, weight_decay=0.1, dropout=0.05,
        grad_clip=1.0, use_amp=True, vocab_size=vocab_size, num_experts=12, expert_top_k=2,
        load_balancing_weight=0.01
    )
    experiments.append(("ultimate_combo", ultimate_combo, "Ultimate combination of all best parameters"))
    
    return experiments


def run_focused_ablation():
    """Run focused ablation study"""
    print("🎯 Focused Ablation Study - Optimizing Based on Initial Results")
    print("="*70)
    
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
    
    # Get experiments
    experiments = get_focused_experiments(vocab_size)
    print(f"\n🧪 Running {len(experiments)} focused experiments...")
    
    results = []
    
    for i, (name, config, description) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Experiment {i}/{len(experiments)}: {name}")
        print(f"📝 {description}")
        print(f"{'='*70}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Train model
        start_time = time.time()
        try:
            model, final_metrics = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            result = {
                'experiment_name': name,
                'description': description,
                'val_loss': final_metrics['val_loss'],
                'val_accuracy': final_metrics['val_accuracy'],
                'val_perplexity': final_metrics['val_perplexity'],
                'training_time_minutes': training_time / 60,
                'config': {
                    'gradient_accumulation_steps': config.gradient_accumulation_steps,
                    'muon_lr': config.muon_lr,
                    'dropout': config.dropout,
                    'weight_decay': config.weight_decay,
                    'num_experts': config.num_experts,
                    'expert_top_k': config.expert_top_k,
                },
                'success': True
            }
            
            print(f"\n✅ Experiment {name} completed successfully!")
            print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Experiment {name} failed: {str(e)}")
            result = {
                'experiment_name': name,
                'description': description,
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'val_perplexity': float('inf'),
                'training_time_minutes': 0,
                'config': {},
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
        
        # Brief pause between experiments
        time.sleep(1)
    
    # Analyze results
    analyze_focused_results(results)
    
    return results


def analyze_focused_results(results):
    """Analyze and visualize focused results"""
    print(f"\n📊 FOCUSED ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    
    # Filter successful experiments
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("❌ No successful experiments to analyze")
        return
    
    # Sort by validation loss (lower is better)
    successful_results.sort(key=lambda x: x['val_loss'])
    
    # Find baseline for comparison
    baseline_result = next((r for r in successful_results if r['experiment_name'] == 'baseline'), None)
    baseline_loss = baseline_result['val_loss'] if baseline_result else None
    
    print(f"{'Rank':<4} {'Experiment':<20} {'Val Loss':<10} {'Val Acc':<10} {'Improvement':<12} {'Time (min)':<10}")
    print(f"{'-'*80}")
    
    for i, result in enumerate(successful_results, 1):
        improvement = ""
        if baseline_loss and result['experiment_name'] != 'baseline':
            improvement_pct = ((baseline_loss - result['val_loss']) / baseline_loss * 100)
            improvement = f"{improvement_pct:+.1f}%"
        elif result['experiment_name'] == 'baseline':
            improvement = "baseline"
        
        print(f"{i:<4} {result['experiment_name']:<20} {result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} {improvement:<12} {result['training_time_minutes']:<10.1f}")
    
    # Best experiment
    best_result = successful_results[0]
    print(f"\n🏆 BEST EXPERIMENT: {best_result['experiment_name']}")
    print(f"   Description: {best_result['description']}")
    print(f"   Validation Loss: {best_result['val_loss']:.4f}")
    print(f"   Validation Accuracy: {best_result['val_accuracy']:.4f}")
    if baseline_loss:
        improvement = ((baseline_loss - best_result['val_loss']) / baseline_loss * 100)
        print(f"   Improvement over baseline: {improvement:.1f}%")
    
    # Save results
    os.makedirs("experiments/ablation_study/results", exist_ok=True)
    results_file = "experiments/ablation_study/results/focused_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create visualization
    create_focused_visualization(successful_results)


def create_focused_visualization(results):
    """Create focused visualization"""
    # Extract data for plotting
    names = [r['experiment_name'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    val_accuracies = [r['val_accuracy'] for r in results]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Validation Loss
    bars1 = ax1.bar(range(len(names)), val_losses, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss by Experiment')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Highlight best result
    bars1[0].set_color('darkgreen')
    bars1[0].set_alpha(1.0)
    
    # Validation Accuracy
    bars2 = ax2.bar(range(len(names)), val_accuracies, color='lightblue', alpha=0.7)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy by Experiment')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # Highlight best result
    bars2[0].set_color('darkblue')
    bars2[0].set_alpha(1.0)
    
    plt.tight_layout()
    plot_file = "experiments/ablation_study/results/focused_ablation_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 Visualization saved to: {plot_file}")


if __name__ == "__main__":
    results = run_focused_ablation()
    print(f"\n🎉 Focused ablation study completed!")
