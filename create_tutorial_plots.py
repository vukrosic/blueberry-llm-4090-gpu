#!/usr/bin/env python3
"""
Create individual plots for the tutorial article
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load the comprehensive results"""
    results_file = Path("comprehensive_lr_sweep_20250925_060116/final_results.json")
    with open(results_file, 'r') as f:
        return json.load(f)

def create_accuracy_vs_lr_plot(results, output_dir):
    """Create learning rate vs accuracy plot"""
    lrs = [r['learning_rate'] for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the curve
    ax.semilogx(lrs, accuracies, 'o-', linewidth=3, markersize=10, color='#2E86AB', alpha=0.8)
    
    # Highlight best performance
    best_idx = np.argmax(accuracies)
    ax.scatter(lrs[best_idx], accuracies[best_idx], color='#F24236', s=200, zorder=5, 
               edgecolors='white', linewidth=2)
    
    # Add annotation for best performance
    ax.annotate(f'Optimal: LR {lrs[best_idx]:.3f}\nAccuracy: {accuracies[best_idx]:.3f}', 
               (lrs[best_idx], accuracies[best_idx]),
               xytext=(20, 20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='#F24236', alpha=0.8, edgecolor='white'),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#F24236'),
               fontsize=12, color='white', weight='bold')
    
    # Add shaded optimal range
    optimal_range = [0.035, 0.085]
    ax.axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', 
               label='Optimal Range (0.035-0.085)')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=14, weight='bold')
    ax.set_ylabel('Final Validation Accuracy', fontsize=14, weight='bold')
    ax.set_title('Finding the Sweet Spot: Learning Rate vs Model Performance', 
                 fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_vs_accuracy_tutorial.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_loss_vs_lr_plot(results, output_dir):
    """Create learning rate vs loss plot"""
    lrs = [r['learning_rate'] for r in results]
    losses = [r['final_loss'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the curve
    ax.semilogx(lrs, losses, 'o-', linewidth=3, markersize=10, color='#A23B72', alpha=0.8)
    
    # Highlight best performance
    best_idx = np.argmin(losses)
    ax.scatter(lrs[best_idx], losses[best_idx], color='#F18F01', s=200, zorder=5,
               edgecolors='white', linewidth=2)
    
    # Add annotation for best performance
    ax.annotate(f'Best Loss: LR {lrs[best_idx]:.3f}\nLoss: {losses[best_idx]:.3f}', 
               (lrs[best_idx], losses[best_idx]),
               xytext=(20, -30), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='#F18F01', alpha=0.8, edgecolor='white'),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#F18F01'),
               fontsize=12, color='white', weight='bold')
    
    # Add shaded optimal range
    optimal_range = [0.035, 0.085]
    ax.axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', 
               label='Optimal Range (0.035-0.085)')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=14, weight='bold')
    ax.set_ylabel('Final Validation Loss', fontsize=14, weight='bold')
    ax.set_title('Lower is Better: Learning Rate vs Training Loss', 
                 fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_vs_loss_tutorial.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_efficiency_plot(results, output_dir):
    """Create training efficiency plot"""
    lrs = [r['learning_rate'] for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    times = [r['training_time'] for r in results]
    efficiency = [acc / (time / 60) for acc, time in zip(accuracies, times)]  # accuracy per minute
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the curve
    ax.semilogx(lrs, efficiency, 'o-', linewidth=3, markersize=10, color='#C73E1D', alpha=0.8)
    
    # Highlight best efficiency
    best_idx = np.argmax(efficiency)
    ax.scatter(lrs[best_idx], efficiency[best_idx], color='#F4E04D', s=200, zorder=5,
               edgecolors='white', linewidth=2)
    
    # Add annotation for best efficiency
    ax.annotate(f'Most Efficient: LR {lrs[best_idx]:.3f}\nEfficiency: {efficiency[best_idx]:.4f}', 
               (lrs[best_idx], efficiency[best_idx]),
               xytext=(20, 20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='#F4E04D', alpha=0.8, edgecolor='white'),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#F4E04D'),
               fontsize=12, color='black', weight='bold')
    
    # Add shaded optimal range
    optimal_range = [0.035, 0.085]
    ax.axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', 
               label='Optimal Range (0.035-0.085)')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=14, weight='bold')
    ax.set_ylabel('Training Efficiency (Accuracy/Minute)', fontsize=14, weight='bold')
    ax.set_title('Time is Money: Learning Rate vs Training Efficiency', 
                 fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_vs_efficiency_tutorial.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all tutorial plots"""
    results = load_results()
    output_dir = Path("tutorial_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("📊 Creating tutorial plots...")
    create_accuracy_vs_lr_plot(results, output_dir)
    print("✅ Created accuracy vs LR plot")
    
    create_loss_vs_lr_plot(results, output_dir)
    print("✅ Created loss vs LR plot")
    
    create_efficiency_plot(results, output_dir)
    print("✅ Created efficiency vs LR plot")
    
    print(f"📁 All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
