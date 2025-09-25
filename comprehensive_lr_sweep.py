#!/usr/bin/env python3
"""
Comprehensive Learning Rate Sweep with Visualization

This script runs an extensive learning rate sweep and creates:
1. Individual loss vs time plots for each learning rate
2. Combined plot showing all learning rates on the same graph
3. Comprehensive analysis and comparison
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_lr_sweep_range():
    """Create comprehensive learning rate range for testing"""
    
    # Create multiple ranges for comprehensive testing
    lr_ranges = []
    
    # Very low range (0.001 - 0.01)
    lr_ranges.extend(np.logspace(-3, -2, 5))  # 0.001, 0.002, 0.003, 0.006, 0.01
    
    # Low range (0.01 - 0.1)
    lr_ranges.extend(np.linspace(0.015, 0.095, 9))  # 0.015, 0.025, 0.035, ..., 0.095
    
    # Medium range (0.1 - 0.5) - more granular
    lr_ranges.extend(np.linspace(0.1, 0.5, 9))  # 0.1, 0.15, 0.2, ..., 0.5
    
    # High range (0.5 - 1.0) - sparse sampling
    lr_ranges.extend([0.6, 0.8, 1.0])
    
    # Remove duplicates and sort
    lr_ranges = sorted(list(set(np.round(lr_ranges, 4))))
    
    return lr_ranges

def run_single_lr_experiment(lr: float, config_base: Dict, tokens: List[int], output_dir: Path):
    """Run a single learning rate experiment and save training data"""
    
    print(f"\n🔍 Testing LR = {lr:.4f}")
    
    try:
        # Create config
        config = MoEModelConfig(
            vocab_size=config_base['vocab_size'],
            muon_lr=lr,
            max_steps=80,  # Enough steps to see convergence patterns
            eval_every=8,   # More frequent evaluation
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
        
        print(f"✅ LR {lr:.4f} - Acc: {result['final_accuracy']:.4f}, Loss: {result['final_loss']:.4f}")
        return result
        
    except Exception as e:
        print(f"❌ LR {lr:.4f} failed: {str(e)}")
        return {
            'learning_rate': lr,
            'error': str(e),
            'status': 'failed'
        }

def modify_trainer_for_data_collection():
    """Modify the trainer to collect training data for plotting"""
    
    # Read the current trainer
    with open('/root/blueberry-llm-4090-gpu/training/trainer.py', 'r') as f:
        trainer_content = f.read()
    
    # Create a modified version that saves evaluation data
    modified_trainer = trainer_content.replace(
        'return model, final_eval',
        '''# Save evaluation data for plotting
    eval_data = {
        'eval_steps': eval_steps,
        'eval_losses': eval_losses,
        'eval_times': eval_times,
        'config_lr': config.muon_lr
    }
    
    # Save to file for later plotting
    import pickle
    eval_file = f"temp_eval_data_lr_{config.muon_lr:.4f}.pkl"
    with open(eval_file, 'wb') as f:
        pickle.dump(eval_data, f)
    
    return model, final_eval'''
    )
    
    # Write the modified trainer
    with open('/root/blueberry-llm-4090-gpu/training/trainer_data_collection.py', 'w') as f:
        f.write(modified_trainer)

def create_individual_plots(lr_results: List[Dict], output_dir: Path):
    """Create individual loss vs time plots for each learning rate"""
    
    plots_dir = output_dir / "individual_plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("📊 Creating individual plots...")
    
    for result in lr_results:
        if result['status'] != 'completed':
            continue
            
        lr = result['learning_rate']
        
        # Try to load evaluation data
        eval_file = f"temp_eval_data_lr_{lr:.4f}.pkl"
        if os.path.exists(eval_file):
            try:
                import pickle
                with open(eval_file, 'rb') as f:
                    eval_data = pickle.load(f)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert steps to relative time (assuming roughly equal step time)
                steps = eval_data['eval_steps']
                losses = eval_data['eval_losses']
                
                ax.plot(steps, losses, 'b-o', linewidth=2, markersize=6, alpha=0.8)
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Validation Loss')
                ax.set_title(f'Learning Rate {lr:.4f} - Validation Loss vs Steps\nFinal: Loss={result["final_loss"]:.3f}, Acc={result["final_accuracy"]:.3f}')
                ax.grid(True, alpha=0.3)
                
                # Add final metrics as text
                textstr = f'Final Accuracy: {result["final_accuracy"]:.4f}\nFinal Loss: {result["final_loss"]:.4f}\nFinal Perplexity: {result["final_perplexity"]:.1f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f'lr_{lr:.4f}_loss_vs_steps.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Clean up temp file
                os.remove(eval_file)
                
            except Exception as e:
                print(f"⚠️  Could not create plot for LR {lr:.4f}: {e}")

def create_combined_plot(lr_results: List[Dict], output_dir: Path):
    """Create combined plot showing all learning rates on the same graph"""
    
    print("📊 Creating combined plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Comprehensive Learning Rate Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different learning rate ranges
    colors = plt.cm.viridis(np.linspace(0, 1, len([r for r in lr_results if r['status'] == 'completed'])))
    
    successful_results = [r for r in lr_results if r['status'] == 'completed']
    
    # Plot 1: Final metrics vs learning rate
    lrs = [r['learning_rate'] for r in successful_results]
    losses = [r['final_loss'] for r in successful_results]
    accuracies = [r['final_accuracy'] for r in successful_results]
    
    ax1.semilogx(lrs, losses, 'o-', linewidth=2, markersize=8, color='red', alpha=0.7, label='Final Loss')
    ax1_twin = ax1.twinx()
    ax1_twin.semilogx(lrs, accuracies, 's-', linewidth=2, markersize=8, color='green', alpha=0.7, label='Final Accuracy')
    
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Final Validation Loss', color='red')
    ax1_twin.set_ylabel('Final Validation Accuracy', color='green')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Final Performance vs Learning Rate')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance heatmap
    # Create a matrix showing performance across different metrics
    metrics_matrix = []
    lr_labels = []
    
    for result in successful_results:
        lr_labels.append(f"{result['learning_rate']:.4f}")
        metrics_matrix.append([
            result['final_loss'],
            result['final_accuracy'],
            result['final_perplexity'] / 1000,  # Scale down perplexity
            result['training_time'] / 60  # Convert to minutes
        ])
    
    metrics_matrix = np.array(metrics_matrix)
    
    # Normalize each column to 0-1 for better visualization
    for i in range(metrics_matrix.shape[1]):
        col = metrics_matrix[:, i]
        metrics_matrix[:, i] = (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else col
    
    im = ax2.imshow(metrics_matrix.T, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(lr_labels)))
    ax2.set_xticklabels(lr_labels, rotation=45, ha='right')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['Loss', 'Accuracy', 'Perplexity (k)', 'Time (min)'])
    ax2.set_title('Performance Heatmap (Normalized)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Normalized Performance (0=best, 1=worst)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_lr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis(lr_results: List[Dict], output_dir: Path):
    """Create detailed analysis plots and statistics"""
    
    successful_results = [r for r in lr_results if r['status'] == 'completed']
    
    if len(successful_results) < 3:
        print("⚠️  Not enough successful results for detailed analysis")
        return
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Learning Rate Analysis', fontsize=16, fontweight='bold')
    
    lrs = np.array([r['learning_rate'] for r in successful_results])
    losses = np.array([r['final_loss'] for r in successful_results])
    accuracies = np.array([r['final_accuracy'] for r in successful_results])
    perplexities = np.array([r['final_perplexity'] for r in successful_results])
    times = np.array([r['training_time'] for r in successful_results])
    
    # 1. Learning rate vs accuracy (log scale)
    axes[0, 0].semilogx(lrs, accuracies, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 0].set_xlabel('Learning Rate (log scale)')
    axes[0, 0].set_ylabel('Final Validation Accuracy')
    axes[0, 0].set_title('Accuracy vs Learning Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight best performance
    best_idx = np.argmax(accuracies)
    axes[0, 0].scatter(lrs[best_idx], accuracies[best_idx], color='red', s=100, zorder=5)
    axes[0, 0].annotate(f'Best: {lrs[best_idx]:.4f}', 
                       (lrs[best_idx], accuracies[best_idx]),
                       xytext=(5, 5), textcoords='offset points')
    
    # 2. Learning rate vs loss (log scale)
    axes[0, 1].semilogx(lrs, losses, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Learning Rate (log scale)')
    axes[0, 1].set_ylabel('Final Validation Loss')
    axes[0, 1].set_title('Loss vs Learning Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Highlight best performance
    best_loss_idx = np.argmin(losses)
    axes[0, 1].scatter(lrs[best_loss_idx], losses[best_loss_idx], color='green', s=100, zorder=5)
    axes[0, 1].annotate(f'Best: {lrs[best_loss_idx]:.4f}', 
                       (lrs[best_loss_idx], losses[best_loss_idx]),
                       xytext=(5, 5), textcoords='offset points')
    
    # 3. Efficiency analysis (accuracy per unit time)
    efficiency = accuracies / (times / 60)  # accuracy per minute
    axes[1, 0].semilogx(lrs, efficiency, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel('Learning Rate (log scale)')
    axes[1, 0].set_ylabel('Efficiency (Accuracy/Minute)')
    axes[1, 0].set_title('Training Efficiency vs Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning rate distribution of top performers
    # Find top 20% performers
    top_20_percent = int(np.ceil(len(successful_results) * 0.2))
    top_indices = np.argsort(accuracies)[-top_20_percent:]
    top_lrs = lrs[top_indices]
    
    axes[1, 1].hist(top_lrs, bins=10, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Distribution of Top {top_20_percent} Learning Rates')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_lr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate insights
    insights = {
        'best_lr': float(lrs[best_idx]),
        'best_accuracy': float(accuracies[best_idx]),
        'best_loss': float(losses[best_loss_idx]),
        'best_loss_lr': float(lrs[best_loss_idx]),
        'top_20_percent_lr_range': [float(top_lrs.min()), float(top_lrs.max())],
        'total_experiments': len(lr_results),
        'successful_experiments': len(successful_results),
        'lr_range_tested': [float(lrs.min()), float(lrs.max())]
    }
    
    # Save insights
    with open(output_dir / 'comprehensive_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    return insights

def run_comprehensive_lr_sweep():
    """Run comprehensive learning rate sweep"""
    
    print("🚀 Starting Comprehensive Learning Rate Sweep")
    
    # Get learning rates to test
    lr_values = create_lr_sweep_range()
    print(f"📊 Testing {len(lr_values)} learning rates")
    print(f"🔍 Range: {lr_values[0]:.4f} to {lr_values[-1]:.4f}")
    
    # Load data once
    print("\n📊 Loading dataset...")
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    config_base = {'vocab_size': vocab_size}
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"comprehensive_lr_sweep_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Modify trainer to collect data
    modify_trainer_for_data_collection()
    
    # Run experiments
    results = []
    for i, lr in enumerate(lr_values):
        print(f"\n{'='*60}")
        print(f"🔍 Experiment {i+1}/{len(lr_values)}: LR = {lr:.4f}")
        print(f"{'='*60}")
        
        result = run_single_lr_experiment(lr, config_base, tokens, output_dir)
        results.append(result)
        
        # Save intermediate results
        with open(output_dir / 'intermediate_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final results
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_individual_plots(results, output_dir)
    create_combined_plot(results, output_dir)
    insights = create_detailed_analysis(results, output_dir)
    
    # Print summary
    successful_results = [r for r in results if r['status'] == 'completed']
    
    print(f"\n{'='*60}")
    print("🎉 COMPREHENSIVE LEARNING RATE SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total experiments: {len(results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['final_accuracy'])
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Learning Rate: {best_result['learning_rate']:.4f}")
        print(f"   Final Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"   Final Loss: {best_result['final_loss']:.4f}")
        print(f"   Final Perplexity: {best_result['final_perplexity']:.2f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📊 Visualizations created:")
        print(f"   - individual_plots/ (individual LR plots)")
        print(f"   - combined_lr_analysis.png")
        print(f"   - detailed_lr_analysis.png")
        
    return output_dir

def main():
    """Main function"""
    try:
        output_dir = run_comprehensive_lr_sweep()
        print(f"\n🎉 Comprehensive sweep completed!")
        print(f"📁 All results and plots saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Sweep interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during sweep: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
