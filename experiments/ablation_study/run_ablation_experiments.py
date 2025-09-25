"""
Ablation Study Runner for Blueberry LLM
Runs systematic experiments to improve validation loss
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
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from ablation_configs import get_baseline_config, get_ablation_experiments, get_best_combination_experiments


class AblationExperimentRunner:
    def __init__(self, results_dir: str = "experiments/ablation_study/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
        
    def run_experiment(self, experiment, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Run a single ablation experiment"""
        print(f"\n{'='*80}")
        print(f"🧪 Running Experiment: {experiment.name}")
        print(f"📝 Description: {experiment.description}")
        print(f"🎯 Expected: {experiment.expected_improvement}")
        print(f"{'='*80}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Train model
        start_time = time.time()
        try:
            model, final_metrics = train_moe_model(experiment.config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Collect results
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'expected_improvement': experiment.expected_improvement,
                'config': {
                    'd_model': experiment.config.d_model,
                    'n_layers': experiment.config.n_layers,
                    'n_heads': experiment.config.n_heads,
                    'd_ff': experiment.config.d_ff,
                    'batch_size': experiment.config.batch_size,
                    'max_steps': experiment.config.max_steps,
                    'gradient_accumulation_steps': experiment.config.gradient_accumulation_steps,
                    'muon_lr': experiment.config.muon_lr,
                    'dropout': experiment.config.dropout,
                    'weight_decay': experiment.config.weight_decay,
                    'grad_clip': experiment.config.grad_clip,
                    'num_experts': experiment.config.num_experts,
                    'expert_top_k': experiment.config.expert_top_k,
                    'load_balancing_weight': experiment.config.load_balancing_weight,
                },
                'results': final_metrics,
                'training_time_minutes': training_time / 60,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print(f"\n✅ Experiment {experiment.name} completed successfully!")
            print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Experiment {experiment.name} failed: {str(e)}")
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'expected_improvement': experiment.expected_improvement,
                'config': {},
                'results': {},
                'training_time_minutes': 0,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_all_experiments(self, experiments: List, train_loader: DataLoader, val_loader: DataLoader):
        """Run all ablation experiments"""
        print(f"\n🚀 Starting Ablation Study with {len(experiments)} experiments")
        print(f"📊 Results will be saved to: {self.results_dir}")
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n📈 Progress: {i}/{len(experiments)} experiments")
            result = self.run_experiment(experiment, train_loader, val_loader)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between experiments
            time.sleep(2)
        
        print(f"\n🎉 All experiments completed!")
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "ablation_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and visualize results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Filter successful experiments
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("❌ No successful experiments to analyze")
            return
        
        # Create results DataFrame
        df_data = []
        for result in successful_results:
            row = {
                'experiment': result['experiment_name'],
                'val_loss': result['results']['val_loss'],
                'val_accuracy': result['results']['val_accuracy'],
                'val_perplexity': result['results']['val_perplexity'],
                'training_time': result['training_time_minutes']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by validation loss (lower is better)
        df_sorted = df.sort_values('val_loss')
        
        print(f"\n📊 ABLATION STUDY RESULTS")
        print(f"{'='*80}")
        print(f"{'Rank':<4} {'Experiment':<20} {'Val Loss':<10} {'Val Acc':<10} {'Val PPL':<10} {'Time (min)':<10}")
        print(f"{'-'*80}")
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i:<4} {row['experiment']:<20} {row['val_loss']:<10.4f} {row['val_accuracy']:<10.4f} {row['val_perplexity']:<10.2f} {row['training_time']:<10.1f}")
        
        # Find best experiment
        best_experiment = df_sorted.iloc[0]
        print(f"\n🏆 BEST EXPERIMENT: {best_experiment['experiment']}")
        print(f"   Validation Loss: {best_experiment['val_loss']:.4f}")
        print(f"   Validation Accuracy: {best_experiment['val_accuracy']:.4f}")
        print(f"   Training Time: {best_experiment['training_time']:.1f} minutes")
        
        # Create visualizations
        self.create_visualizations(df_sorted)
        
        # Save analysis
        self.save_analysis(df_sorted)
    
    def create_visualizations(self, df_sorted):
        """Create visualization plots"""
        # Validation Loss Comparison
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Validation Loss
        plt.subplot(2, 2, 1)
        plt.bar(range(len(df_sorted)), df_sorted['val_loss'])
        plt.xlabel('Experiment Rank')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss by Experiment')
        plt.xticks(range(len(df_sorted)), df_sorted['experiment'], rotation=45, ha='right')
        
        # Plot 2: Validation Accuracy
        plt.subplot(2, 2, 2)
        plt.bar(range(len(df_sorted)), df_sorted['val_accuracy'])
        plt.xlabel('Experiment Rank')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy by Experiment')
        plt.xticks(range(len(df_sorted)), df_sorted['experiment'], rotation=45, ha='right')
        
        # Plot 3: Training Time vs Performance
        plt.subplot(2, 2, 3)
        plt.scatter(df_sorted['training_time'], df_sorted['val_loss'], alpha=0.7)
        plt.xlabel('Training Time (minutes)')
        plt.ylabel('Validation Loss')
        plt.title('Training Time vs Validation Loss')
        
        # Plot 4: Performance Summary
        plt.subplot(2, 2, 4)
        x = range(len(df_sorted))
        plt.plot(x, df_sorted['val_loss'], 'o-', label='Val Loss', linewidth=2)
        plt.plot(x, df_sorted['val_accuracy'], 's-', label='Val Accuracy', linewidth=2)
        plt.xlabel('Experiment Rank')
        plt.ylabel('Score')
        plt.title('Performance Summary')
        plt.legend()
        plt.xticks(x, df_sorted['experiment'], rotation=45, ha='right')
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "ablation_results_visualization.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Visualization saved to: {plot_file}")
    
    def save_analysis(self, df_sorted):
        """Save detailed analysis report"""
        report_file = os.path.join(self.results_dir, "ablation_analysis_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Ablation Study Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Successful experiments: {len([r for r in self.results if r['success']])}\n")
            f.write(f"Failed experiments: {len([r for r in self.results if not r['success']])}\n\n")
            
            f.write("## Top 5 Performing Experiments\n\n")
            f.write("| Rank | Experiment | Val Loss | Val Accuracy | Val Perplexity | Training Time |\n")
            f.write("|------|------------|----------|--------------|----------------|---------------|\n")
            
            for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
                f.write(f"| {i} | {row['experiment']} | {row['val_loss']:.4f} | {row['val_accuracy']:.4f} | {row['val_perplexity']:.2f} | {row['training_time']:.1f} min |\n")
            
            f.write("\n## Best Experiment Details\n\n")
            best_exp = df_sorted.iloc[0]
            f.write(f"**Experiment**: {best_exp['experiment']}\n")
            f.write(f"**Validation Loss**: {best_exp['val_loss']:.4f}\n")
            f.write(f"**Validation Accuracy**: {best_exp['val_accuracy']:.4f}\n")
            f.write(f"**Training Time**: {best_exp['training_time']:.1f} minutes\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the ablation study results:\n\n")
            f.write("1. **Best Configuration**: Use the top-performing experiment configuration\n")
            f.write("2. **Key Improvements**: Focus on the parameters that showed the most improvement\n")
            f.write("3. **Further Experiments**: Consider combining the best individual improvements\n\n")
            
            f.write("## Failed Experiments\n\n")
            failed_experiments = [r for r in self.results if not r['success']]
            if failed_experiments:
                for exp in failed_experiments:
                    f.write(f"- **{exp['experiment_name']}**: {exp.get('error', 'Unknown error')}\n")
            else:
                f.write("All experiments completed successfully!\n")
        
        print(f"📄 Analysis report saved to: {report_file}")


def main():
    """Main function to run ablation study"""
    print("🔬 Blueberry LLM Ablation Study")
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
    
    # Get experiments
    experiments = get_ablation_experiments(vocab_size)
    best_combinations = get_best_combination_experiments(vocab_size)
    all_experiments = experiments + best_combinations
    
    print(f"\n🧪 Total experiments to run: {len(all_experiments)}")
    
    # Run ablation study
    runner = AblationExperimentRunner()
    runner.run_all_experiments(all_experiments, train_loader, val_loader)


if __name__ == "__main__":
    main()
