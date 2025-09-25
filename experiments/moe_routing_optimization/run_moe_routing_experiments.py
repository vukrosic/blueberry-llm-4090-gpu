"""
MoE Routing Optimization Experiment Runner
Building on 7.5% cumulative validation loss improvement
Goal: Optimize expert routing strategies for additional performance gains
"""

import torch
import time
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.moe_routing_optimization.moe_routing_configs import (
    get_moe_routing_experiments, create_moe_config_from_experiment, 
    MoERoutingExperiment, analyze_moe_complexity, estimate_parameter_counts
)


class MoERoutingExperimentRunner:
    def __init__(self, results_dir: str = "experiments/moe_routing_optimization/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
    
    def create_custom_moe_model(self, experiment: MoERoutingExperiment, config: MoEModelConfig):
        """Create custom MoE model with experiment-specific routing parameters"""
        from models.moe_llm import MoEMinimalLLM
        from models.components import TopKRouter
        
        # Create model with experiment configuration
        model = MoEMinimalLLM(config)
        
        # Customize routing noise if needed
        if hasattr(experiment, 'routing_noise_std'):
            for block in model.transformer_blocks:
                if hasattr(block.feed_forward, 'router'):
                    block.feed_forward.router.noise_std = experiment.routing_noise_std
        
        return model
    
    def run_experiment(self, experiment: MoERoutingExperiment, config: MoEModelConfig, 
                      train_loader: DataLoader, val_loader: DataLoader):
        """Run a single MoE routing experiment"""
        print(f"\n{'='*90}")
        print(f"🧪 MoE Routing Experiment: {experiment.name}")
        print(f"📝 Description: {experiment.description}")
        print(f"⚙️ Experts: {experiment.num_experts}")
        print(f"⚙️ Top-K: {experiment.expert_top_k}")
        print(f"⚙️ Load Balance Weight: {experiment.load_balancing_weight}")
        print(f"⚙️ Routing Noise: {experiment.routing_noise_std}")
        print(f"🎯 Expected: {experiment.expected_improvement}")
        print(f"{'='*90}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Run training with experiment configuration
        start_time = time.time()
        try:
            model, final_metrics = train_moe_model(config, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Calculate parameter efficiency
            total_params = sum(p.numel() for p in model.parameters())
            active_params = sum(p.numel() for n, p in model.named_parameters() if 'expert' not in n)
            expert_params = total_params - active_params
            
            # Estimate active parameters during forward pass
            params_per_expert = expert_params // experiment.num_experts
            active_expert_params = params_per_expert * experiment.expert_top_k
            total_active_params = active_params + active_expert_params
            param_efficiency = total_active_params / total_params
            
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'moe_config': {
                    'num_experts': experiment.num_experts,
                    'expert_top_k': experiment.expert_top_k,
                    'load_balancing_weight': experiment.load_balancing_weight,
                    'routing_noise_std': experiment.routing_noise_std,
                },
                'results': final_metrics,
                'training_time_minutes': training_time / 60,
                'model_stats': {
                    'total_parameters': total_params,
                    'active_parameters': total_active_params,
                    'expert_parameters': expert_params,
                    'parameter_efficiency': param_efficiency,
                    'experts_per_token': experiment.expert_top_k,
                    'expert_utilization_ratio': experiment.expert_top_k / experiment.num_experts,
                },
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print(f"\n✅ Experiment {experiment.name} completed successfully!")
            print(f"   📉 Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   📈 Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   📊 Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
            print(f"   ⚙️ Parameter Efficiency: {param_efficiency:.1%}")
            print(f"   🏗️ Total Parameters: {total_params:,}")
            print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Experiment {experiment.name} failed: {str(e)}")
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'moe_config': {},
                'results': {},
                'training_time_minutes': 0,
                'model_stats': {},
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_all_experiments(self, experiments: List[MoERoutingExperiment], 
                          train_loader: DataLoader, val_loader: DataLoader, vocab_size: int):
        """Run all MoE routing experiments"""
        print(f"\n🔬 MoE Routing Optimization Study")
        print(f"📊 Running {len(experiments)} experiments")
        print(f"🎯 Goal: Improve upon 7.5% cumulative validation loss improvement")
        print(f"💾 Results will be saved to: {self.results_dir}")
        
        # Show complexity analysis
        print(f"\n📋 Experiment Overview:")
        analyze_moe_complexity(experiments[:8])  # Show first 8 for overview
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n📈 Progress: {i}/{len(experiments)} experiments")
            
            # Create configuration for this experiment
            config = create_moe_config_from_experiment(experiment, vocab_size)
            
            # Run experiment
            result = self.run_experiment(experiment, config, train_loader, val_loader)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between experiments
            time.sleep(2)
        
        print(f"\n🎉 All MoE routing experiments completed!")
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "moe_routing_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and visualize MoE routing results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            print("❌ No successful experiments to analyze")
            return
        
        # Sort by validation loss
        successful_results.sort(key=lambda x: x['results']['val_loss'])
        
        # Find baseline
        baseline_result = next((r for r in successful_results if 'baseline' in r['experiment_name']), None)
        baseline_loss = baseline_result['results']['val_loss'] if baseline_result else None
        
        print(f"\n📊 MOE ROUTING OPTIMIZATION RESULTS")
        print(f"{'='*120}")
        print(f"{'Rank':<4} {'Experiment':<25} {'Val Loss':<10} {'Val Acc':<10} {'Experts':<8} {'Top-K':<6} {'Param Eff':<10} {'Improvement':<12}")
        print(f"{'-'*120}")
        
        for i, result in enumerate(successful_results, 1):
            improvement = ""
            if baseline_loss and result['experiment_name'] != baseline_result['experiment_name']:
                improvement_pct = ((baseline_loss - result['results']['val_loss']) / baseline_loss * 100)
                improvement = f"{improvement_pct:+.1f}%"
            elif baseline_result and result['experiment_name'] == baseline_result['experiment_name']:
                improvement = "baseline"
            
            moe_config = result.get('moe_config', {})
            model_stats = result.get('model_stats', {})
            
            print(f"{i:<4} {result['experiment_name']:<25} {result['results']['val_loss']:<10.4f} "
                  f"{result['results']['val_accuracy']:<10.4f} {moe_config.get('num_experts', 0):<8} "
                  f"{moe_config.get('expert_top_k', 0):<6} {model_stats.get('parameter_efficiency', 0):<10.1%} {improvement:<12}")
        
        # Best experiment
        best_result = successful_results[0]
        print(f"\n🏆 BEST MOE ROUTING: {best_result['experiment_name']}")
        print(f"   📝 Description: {best_result['description']}")
        print(f"   📉 Validation Loss: {best_result['results']['val_loss']:.4f}")
        print(f"   📈 Validation Accuracy: {best_result['results']['val_accuracy']:.4f}")
        print(f"   ⚙️ Configuration: {best_result['moe_config']['num_experts']} experts, "
              f"top-{best_result['moe_config']['expert_top_k']}")
        if baseline_loss:
            improvement = ((baseline_loss - best_result['results']['val_loss']) / baseline_loss * 100)
            print(f"   🎯 Improvement over baseline: {improvement:.1f}%")
        
        # Create visualizations
        self.create_visualizations(successful_results)
        
        # Save analysis report
        self.save_analysis_report(successful_results)
    
    def create_visualizations(self, results):
        """Create comprehensive MoE routing visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Validation Loss by Experiment
        names = [r['experiment_name'] for r in results]
        val_losses = [r['results']['val_loss'] for r in results]
        colors = ['darkgreen' if i == 0 else 'lightcoral' for i in range(len(results))]
        
        bars = ax1.bar(range(len(names)), val_losses, color=colors, alpha=0.8)
        ax1.set_xlabel('Experiment Rank')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Validation Loss by MoE Routing Configuration')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Plot 2: Expert Count vs Performance
        expert_counts = [r.get('moe_config', {}).get('num_experts', 0) for r in results]
        ax2.scatter(expert_counts, val_losses, alpha=0.7, s=100)
        ax2.set_xlabel('Number of Experts')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Expert Count vs Validation Loss')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top-K vs Performance
        top_k_values = [r.get('moe_config', {}).get('expert_top_k', 0) for r in results]
        ax3.scatter(top_k_values, val_losses, alpha=0.7, s=100, c=expert_counts, cmap='viridis')
        ax3.set_xlabel('Top-K Routing')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Top-K Routing vs Validation Loss')
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Number of Experts')
        
        # Plot 4: Parameter Efficiency vs Performance
        param_effs = [r.get('model_stats', {}).get('parameter_efficiency', 0) for r in results]
        ax4.scatter(param_effs, val_losses, alpha=0.7, s=100)
        ax4.set_xlabel('Parameter Efficiency')
        ax4.set_ylabel('Validation Loss')
        ax4.set_title('Parameter Efficiency vs Validation Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "moe_routing_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Analysis visualization saved to: {plot_file}")
    
    def save_analysis_report(self, results):
        """Save detailed MoE routing analysis report"""
        report_file = os.path.join(self.results_dir, "moe_routing_analysis_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# MoE Routing Optimization Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Objective\n\n")
            f.write("Optimize MoE expert routing strategies to achieve additional validation loss improvements ")
            f.write("beyond the current 7.5% cumulative improvement.\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Successful experiments: {len(results)}\n\n")
            
            f.write("## Top 5 Performing MoE Configurations\n\n")
            f.write("| Rank | Experiment | Val Loss | Val Accuracy | Experts | Top-K | Load Balance | Improvement |\n")
            f.write("|------|------------|----------|--------------|---------|-------|--------------|-------------|\n")
            
            baseline_loss = None
            baseline_result = next((r for r in results if 'baseline' in r['experiment_name']), None)
            if baseline_result:
                baseline_loss = baseline_result['results']['val_loss']
            
            for i, result in enumerate(results[:5], 1):
                improvement = ""
                if baseline_loss and result['experiment_name'] != baseline_result['experiment_name']:
                    improvement_pct = ((baseline_loss - result['results']['val_loss']) / baseline_loss * 100)
                    improvement = f"{improvement_pct:+.1f}%"
                elif baseline_result and result['experiment_name'] == baseline_result['experiment_name']:
                    improvement = "baseline"
                
                moe_config = result.get('moe_config', {})
                f.write(f"| {i} | {result['experiment_name']} | {result['results']['val_loss']:.4f} | "
                       f"{result['results']['val_accuracy']:.4f} | {moe_config.get('num_experts', 0)} | "
                       f"{moe_config.get('expert_top_k', 0)} | {moe_config.get('load_balancing_weight', 0):.3f} | {improvement} |\n")
            
            f.write("\n## Best Configuration\n\n")
            best = results[0]
            f.write(f"**Experiment**: {best['experiment_name']}\n")
            f.write(f"**Description**: {best['description']}\n")
            f.write(f"**Configuration**:\n")
            f.write(f"- Number of Experts: {best['moe_config']['num_experts']}\n")
            f.write(f"- Top-K Routing: {best['moe_config']['expert_top_k']}\n")
            f.write(f"- Load Balancing Weight: {best['moe_config']['load_balancing_weight']}\n")
            f.write(f"- Routing Noise: {best['moe_config']['routing_noise_std']}\n")
            f.write(f"**Performance**:\n")
            f.write(f"- Validation Loss: {best['results']['val_loss']:.4f}\n")
            f.write(f"- Validation Accuracy: {best['results']['val_accuracy']:.4f}\n")
            f.write(f"- Parameter Efficiency: {best['model_stats']['parameter_efficiency']:.1%}\n")
            if baseline_loss:
                improvement = ((baseline_loss - best['results']['val_loss']) / baseline_loss * 100)
                f.write(f"- Improvement: {improvement:.1f}%\n")
            
            f.write("\n## Key Insights\n\n")
            f.write("1. **Expert Count Impact**: Analysis of how expert count affects performance\n")
            f.write("2. **Top-K Routing**: Optimal routing strategy for best performance\n")
            f.write("3. **Load Balancing**: Effect of load balancing weight on expert utilization\n")
            f.write("4. **Parameter Efficiency**: Trade-offs between model size and performance\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the MoE routing optimization:\n\n")
            f.write("1. **Optimal Configuration**: Use the top-performing MoE routing setup\n")
            f.write("2. **Production Implementation**: Apply the best configuration to main branch\n")
            f.write("3. **Further Research**: Explore dynamic routing and adaptive expert selection\n")
        
        print(f"📄 Analysis report saved to: {report_file}")


def main():
    """Main function to run MoE routing optimization study"""
    print("🔬 MoE Routing Optimization Study")
    print("="*60)
    
    # System info
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
    
    # Get experiments (run key subset for efficiency)
    all_experiments = get_moe_routing_experiments(vocab_size)
    key_experiments = all_experiments[:10]  # Run first 10 experiments
    
    print(f"\n🧪 Running {len(key_experiments)} key MoE routing experiments")
    
    # Run experiments
    runner = MoERoutingExperimentRunner()
    runner.run_all_experiments(key_experiments, train_loader, val_loader, vocab_size)


if __name__ == "__main__":
    main()
