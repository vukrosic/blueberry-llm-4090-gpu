#!/usr/bin/env python3
"""
Learning Rate Analysis and Visualization Script

This script analyzes the results from learning rate ablation studies and creates
comprehensive visualizations to understand the relationship between learning rates
and model performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Any

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load results from ablation study"""
    results_file = Path(results_dir) / "results.json"
    with open(results_file, 'r') as f:
        return json.load(f)

def create_performance_analysis(results: List[Dict[str, Any]], output_dir: Path):
    """Create comprehensive performance analysis plots"""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learning Rate Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Learning Rate vs Validation Loss
    axes[0, 0].semilogx(df['muon_lr'], df['final_loss'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Muon Learning Rate')
    axes[0, 0].set_ylabel('Final Validation Loss')
    axes[0, 0].set_title('Learning Rate vs Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        axes[0, 0].annotate(f"{row['experiment_name']}\n{row['final_loss']:.3f}", 
                          (row['muon_lr'], row['final_loss']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # 2. Learning Rate vs Validation Accuracy
    axes[0, 1].semilogx(df['muon_lr'], df['final_accuracy'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Muon Learning Rate')
    axes[0, 1].set_ylabel('Final Validation Accuracy')
    axes[0, 1].set_title('Learning Rate vs Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        axes[0, 1].annotate(f"{row['experiment_name']}\n{row['final_accuracy']:.3f}", 
                          (row['muon_lr'], row['final_accuracy']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # 3. Learning Rate vs Perplexity
    axes[1, 0].semilogx(df['muon_lr'], df['final_perplexity'], 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Muon Learning Rate')
    axes[1, 0].set_ylabel('Final Perplexity')
    axes[1, 0].set_title('Learning Rate vs Perplexity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        axes[1, 0].annotate(f"{row['experiment_name']}\n{row['final_perplexity']:.0f}", 
                          (row['muon_lr'], row['final_perplexity']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # 4. Training Time vs Performance (Accuracy)
    scatter = axes[1, 1].scatter(df['training_time'], df['final_accuracy'], 
                                c=df['muon_lr'], s=100, alpha=0.7, cmap='viridis')
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('Final Validation Accuracy')
    axes[1, 1].set_title('Training Time vs Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Learning Rate')
    
    # Add annotations for each point
    for i, row in df.iterrows():
        axes[1, 1].annotate(f"{row['experiment_name']}", 
                          (row['training_time'], row['final_accuracy']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """Create detailed comparison plots"""
    
    df = pd.DataFrame(results)
    
    # Create a comprehensive comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detailed Learning Rate Comparison', fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics Comparison
    x = np.arange(len(df))
    width = 0.25
    
    # Normalize metrics for comparison (0-1 scale)
    loss_norm = 1 - (df['final_loss'] - df['final_loss'].min()) / (df['final_loss'].max() - df['final_loss'].min())
    acc_norm = df['final_accuracy'] / df['final_accuracy'].max()
    ppl_norm = 1 - (df['final_perplexity'] - df['final_perplexity'].min()) / (df['final_perplexity'].max() - df['final_perplexity'].min())
    
    axes[0].bar(x - width, loss_norm, width, label='Loss (inverted)', alpha=0.8)
    axes[0].bar(x, acc_norm, width, label='Accuracy', alpha=0.8)
    axes[0].bar(x + width, ppl_norm, width, label='Perplexity (inverted)', alpha=0.8)
    
    axes[0].set_xlabel('Learning Rate Experiments')
    axes[0].set_ylabel('Normalized Performance (0-1)')
    axes[0].set_title('Normalized Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lr:.3f}" for lr in df['muon_lr']], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Learning Rate Efficiency (Performance per unit time)
    efficiency = df['final_accuracy'] / df['training_time']
    bars = axes[1].bar(range(len(df)), efficiency, alpha=0.7, color='orange')
    axes[1].set_xlabel('Learning Rate Experiments')
    axes[1].set_ylabel('Efficiency (Accuracy/Time)')
    axes[1].set_title('Training Efficiency')
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels([f"{lr:.3f}" for lr in df['muon_lr']], rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{efficiency.iloc[i]:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Learning Rate vs All Metrics (scatter matrix style)
    metrics = ['final_loss', 'final_accuracy', 'final_perplexity']
    colors = ['red', 'green', 'blue']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[2].semilogx(df['muon_lr'], df[metric], 'o-', 
                        linewidth=2, markersize=8, color=color, 
                        label=metric.replace('final_', '').replace('_', ' ').title())
    
    axes[2].set_xlabel('Muon Learning Rate')
    axes[2].set_ylabel('Metric Value')
    axes[2].set_title('All Metrics vs Learning Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(results: List[Dict[str, Any]], output_dir: Path):
    """Create a summary table of results"""
    
    df = pd.DataFrame(results)
    
    # Create summary statistics
    summary_stats = {
        'Learning Rate': df['muon_lr'].values,
        'Experiment': df['experiment_name'].values,
        'Final Loss': df['final_loss'].round(4).values,
        'Final Accuracy': df['final_accuracy'].round(4).values,
        'Final Perplexity': df['final_perplexity'].round(2).values,
        'Training Time (s)': df['training_time'].round(2).values,
        'Efficiency': (df['final_accuracy'] / df['training_time']).round(6).values
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save as CSV
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    # Create a formatted table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            f"{row['Learning Rate']:.3f}",
            row['Experiment'],
            f"{row['Final Loss']:.4f}",
            f"{row['Final Accuracy']:.4f}",
            f"{row['Final Perplexity']:.2f}",
            f"{row['Training Time (s)']:.2f}",
            f"{row['Efficiency']:.6f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['LR', 'Experiment', 'Loss', 'Accuracy', 'Perplexity', 'Time (s)', 'Efficiency'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Highlight best performance
    best_idx = df['final_accuracy'].idxmax()
    for i in range(len(table_data[0])):
        table[(best_idx + 1, i)].set_facecolor('#90EE90')  # Light green
    
    plt.title('Learning Rate Ablation Study Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

def generate_insights(results: List[Dict[str, Any]], output_dir: Path):
    """Generate insights and recommendations"""
    
    df = pd.DataFrame(results)
    
    # Find best performing experiment
    best_idx = df['final_accuracy'].idxmax()
    best_experiment = df.iloc[best_idx]
    
    # Calculate correlations
    correlations = {
        'lr_loss_corr': np.corrcoef(df['muon_lr'], df['final_loss'])[0, 1],
        'lr_acc_corr': np.corrcoef(df['muon_lr'], df['final_accuracy'])[0, 1],
        'lr_ppl_corr': np.corrcoef(df['muon_lr'], df['final_perplexity'])[0, 1],
        'time_acc_corr': np.corrcoef(df['training_time'], df['final_accuracy'])[0, 1]
    }
    
    # Generate insights
    insights = {
        'best_learning_rate': best_experiment['muon_lr'],
        'best_experiment_name': best_experiment['experiment_name'],
        'best_accuracy': best_experiment['final_accuracy'],
        'best_loss': best_experiment['final_loss'],
        'best_perplexity': best_experiment['final_perplexity'],
        'correlations': correlations,
        'recommendations': []
    }
    
    # Add recommendations based on analysis
    if correlations['lr_acc_corr'] > 0.5:
        insights['recommendations'].append("Higher learning rates generally lead to better accuracy")
    elif correlations['lr_acc_corr'] < -0.5:
        insights['recommendations'].append("Lower learning rates generally lead to better accuracy")
    else:
        insights['recommendations'].append("Learning rate has moderate impact on accuracy")
    
    if best_experiment['muon_lr'] > 0.05:
        insights['recommendations'].append("Optimal learning rate is in the higher range (>0.05)")
    elif best_experiment['muon_lr'] < 0.01:
        insights['recommendations'].append("Optimal learning rate is in the lower range (<0.01)")
    else:
        insights['recommendations'].append("Optimal learning rate is in the medium range (0.01-0.05)")
    
    # Check for overfitting (very high LR)
    very_high_lr = df[df['muon_lr'] == df['muon_lr'].max()]
    if very_high_lr['final_loss'].iloc[0] > df['final_loss'].mean() * 1.5:
        insights['recommendations'].append("Very high learning rates (>0.1) show signs of instability")
    
    # Save insights
    with open(output_dir / 'insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    # Print insights
    print("\n" + "="*60)
    print("🎯 LEARNING RATE ANALYSIS INSIGHTS")
    print("="*60)
    print(f"🏆 Best Learning Rate: {insights['best_learning_rate']:.3f} ({insights['best_experiment_name']})")
    print(f"📊 Best Accuracy: {insights['best_accuracy']:.4f}")
    print(f"📉 Best Loss: {insights['best_loss']:.4f}")
    print(f"🎲 Best Perplexity: {insights['best_perplexity']:.2f}")
    
    print(f"\n📈 Correlations:")
    print(f"   LR vs Loss: {correlations['lr_loss_corr']:.3f}")
    print(f"   LR vs Accuracy: {correlations['lr_acc_corr']:.3f}")
    print(f"   LR vs Perplexity: {correlations['lr_ppl_corr']:.3f}")
    print(f"   Time vs Accuracy: {correlations['time_acc_corr']:.3f}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("="*60)
    
    return insights

def main():
    """Main analysis function"""
    
    # Find the most recent results directory
    results_dirs = list(Path('ablation_results').glob('lr_ablation_*'))
    if not results_dirs:
        print("❌ No ablation results found!")
        return
    
    # Use the most recent results
    latest_results_dir = max(results_dirs, key=lambda x: x.name)
    print(f"📊 Analyzing results from: {latest_results_dir}")
    
    # Load results
    results = load_results(latest_results_dir)
    
    # Create output directory for analysis
    analysis_dir = latest_results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate analysis
    print("📈 Creating performance analysis plots...")
    create_performance_analysis(results, analysis_dir)
    
    print("📊 Creating detailed comparison plots...")
    create_detailed_comparison(results, analysis_dir)
    
    print("📋 Creating summary table...")
    summary_df = create_summary_table(results, analysis_dir)
    
    print("💡 Generating insights...")
    insights = generate_insights(results, analysis_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")
    print(f"📁 Generated files:")
    print(f"   - learning_rate_analysis.png")
    print(f"   - detailed_comparison.png") 
    print(f"   - summary_table.png")
    print(f"   - summary_table.csv")
    print(f"   - insights.json")

if __name__ == "__main__":
    main()
