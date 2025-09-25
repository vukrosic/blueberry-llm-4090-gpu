"""
Learning Rate Schedule Optimization Experiment Runner
Building on the 6.5% validation loss improvement from ablation study
Goal: Achieve additional validation loss improvements through LR schedule optimization
"""

import torch
import time
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model, setup_muon_optimizer
from models.moe_llm import MoEMinimalLLM
from training.evaluation import evaluate_model
from utils.helpers import set_seed
from experiments.lr_schedule_optimization.lr_schedule_configs import (
    get_lr_schedule_experiments, get_schedule_function, LRScheduleExperiment
)


class LRScheduleExperimentRunner:
    def __init__(self, results_dir: str = "experiments/lr_schedule_optimization/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
    
    def create_custom_trainer(self, config: MoEModelConfig, schedule_fn, train_loader: DataLoader, val_loader: DataLoader):
        """Create a custom trainer with specific LR schedule"""
        print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")
        
        # Initialize model
        set_seed(42)
        model = MoEMinimalLLM(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizers
        optimizers = setup_muon_optimizer(model, config)
        
        # Custom learning rate scheduler
        schedulers = []
        for optimizer in optimizers:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)
            schedulers.append(scheduler)
        
        # Run training loop (simplified version of train_moe_model)
        return self._custom_training_loop(model, optimizers, schedulers, config, train_loader, val_loader, device)
    
    def _custom_training_loop(self, model, optimizers, schedulers, config, train_loader, val_loader, device):
        """Custom training loop with LR schedule tracking"""
        from torch.amp import autocast, GradScaler
        import torch.nn.functional as F
        from tqdm import tqdm
        import math
        
        scaler = GradScaler() if config.use_amp else None
        model.train()
        step = 0
        pbar = tqdm(total=config.max_steps, desc="Training MoE")
        
        # Track LR and metrics
        lr_history = []
        eval_metrics = []
        
        while step < config.max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= config.max_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                if config.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        logits, aux_loss = model(x, return_aux_loss=True)
                        ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                        total_loss = ce_loss
                        if aux_loss is not None:
                            total_loss = total_loss + aux_loss
                        loss = total_loss / config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    loss = total_loss / config.gradient_accumulation_steps
                    loss.backward()
                
                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_amp:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                    
                    # Track learning rate
                    current_lr = schedulers[0].get_last_lr()[0]
                    lr_history.append((step, current_lr))
                
                # Evaluation
                if step % config.eval_every == 0 and step > 0:
                    eval_result = evaluate_model(model, val_loader, config)
                    eval_metrics.append((step, eval_result))
                    
                    pbar.set_postfix({
                        'loss': f'{ce_loss.item():.4f}',
                        'val_loss': f'{eval_result["val_loss"]:.4f}',
                        'lr': f'{current_lr:.6f}'
                    })
                
                step += 1
                if step % 20 == 0:
                    pbar.update(20)
        
        pbar.close()
        
        # Final evaluation
        final_eval = evaluate_model(model, val_loader, config)
        
        return model, final_eval, lr_history, eval_metrics
    
    def run_experiment(self, experiment: LRScheduleExperiment, config: MoEModelConfig, 
                      train_loader: DataLoader, val_loader: DataLoader):
        """Run a single LR schedule experiment"""
        print(f"\n{'='*80}")
        print(f"🧪 LR Schedule Experiment: {experiment.name}")
        print(f"📝 Description: {experiment.description}")
        print(f"⚙️ Warmup Ratio: {experiment.warmup_ratio:.1%}")
        print(f"⚙️ Min LR Ratio: {experiment.min_lr_ratio:.1%}")
        print(f"⚙️ Schedule Type: {experiment.schedule_type}")
        print(f"🎯 Expected: {experiment.expected_improvement}")
        print(f"{'='*80}")
        
        # Get schedule function
        schedule_fn = get_schedule_function(experiment, config.max_steps)
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Run training with custom schedule
        start_time = time.time()
        try:
            model, final_metrics, lr_history, eval_metrics = self.create_custom_trainer(
                config, schedule_fn, train_loader, val_loader
            )
            training_time = time.time() - start_time
            
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'schedule_config': {
                    'warmup_ratio': experiment.warmup_ratio,
                    'min_lr_ratio': experiment.min_lr_ratio,
                    'schedule_type': experiment.schedule_type,
                },
                'results': final_metrics,
                'training_time_minutes': training_time / 60,
                'lr_history': lr_history,
                'eval_metrics': eval_metrics,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print(f"\n✅ Experiment {experiment.name} completed successfully!")
            print(f"   📉 Validation Loss: {final_metrics['val_loss']:.4f}")
            print(f"   📈 Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
            print(f"   📊 Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
            print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ Experiment {experiment.name} failed: {str(e)}")
            result = {
                'experiment_name': experiment.name,
                'description': experiment.description,
                'schedule_config': {},
                'results': {},
                'training_time_minutes': 0,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_all_experiments(self, experiments, config: MoEModelConfig, 
                          train_loader: DataLoader, val_loader: DataLoader):
        """Run all LR schedule experiments"""
        print(f"\n🔬 Learning Rate Schedule Optimization Study")
        print(f"📊 Running {len(experiments)} experiments")
        print(f"🎯 Goal: Improve upon 6.5% validation loss gain from ablation study")
        print(f"💾 Results will be saved to: {self.results_dir}")
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n📈 Progress: {i}/{len(experiments)} experiments")
            result = self.run_experiment(experiment, config, train_loader, val_loader)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between experiments
            time.sleep(2)
        
        print(f"\n🎉 All LR schedule experiments completed!")
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "lr_schedule_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and visualize LR schedule results"""
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
        
        print(f"\n📊 LEARNING RATE SCHEDULE OPTIMIZATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Rank':<4} {'Experiment':<25} {'Val Loss':<10} {'Val Acc':<10} {'Improvement':<12} {'Schedule':<15}")
        print(f"{'-'*90}")
        
        for i, result in enumerate(successful_results, 1):
            improvement = ""
            if baseline_loss and result['experiment_name'] != baseline_result['experiment_name']:
                improvement_pct = ((baseline_loss - result['results']['val_loss']) / baseline_loss * 100)
                improvement = f"{improvement_pct:+.1f}%"
            elif baseline_result and result['experiment_name'] == baseline_result['experiment_name']:
                improvement = "baseline"
            
            schedule_type = result.get('schedule_config', {}).get('schedule_type', 'unknown')
            
            print(f"{i:<4} {result['experiment_name']:<25} {result['results']['val_loss']:<10.4f} "
                  f"{result['results']['val_accuracy']:<10.4f} {improvement:<12} {schedule_type:<15}")
        
        # Best experiment
        best_result = successful_results[0]
        print(f"\n🏆 BEST LR SCHEDULE: {best_result['experiment_name']}")
        print(f"   📝 Description: {best_result['description']}")
        print(f"   📉 Validation Loss: {best_result['results']['val_loss']:.4f}")
        print(f"   📈 Validation Accuracy: {best_result['results']['val_accuracy']:.4f}")
        if baseline_loss:
            improvement = ((baseline_loss - best_result['results']['val_loss']) / baseline_loss * 100)
            print(f"   🎯 Improvement over baseline: {improvement:.1f}%")
        
        # Create visualizations
        self.create_visualizations(successful_results)
        
        # Save analysis report
        self.save_analysis_report(successful_results)
    
    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        # Validation Loss Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Validation Loss Ranking
        names = [r['experiment_name'] for r in results]
        val_losses = [r['results']['val_loss'] for r in results]
        colors = ['darkgreen' if i == 0 else 'lightcoral' for i in range(len(results))]
        
        bars = ax1.bar(range(len(names)), val_losses, color=colors, alpha=0.8)
        ax1.set_xlabel('Experiment Rank')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Validation Loss by LR Schedule')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Plot 2: Validation Accuracy
        val_accuracies = [r['results']['val_accuracy'] for r in results]
        bars2 = ax2.bar(range(len(names)), val_accuracies, color=colors, alpha=0.8)
        ax2.set_xlabel('Experiment Rank')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy by LR Schedule')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        # Plot 3: Schedule Type Comparison
        schedule_types = {}
        for result in results:
            stype = result.get('schedule_config', {}).get('schedule_type', 'unknown')
            if stype not in schedule_types:
                schedule_types[stype] = []
            schedule_types[stype].append(result['results']['val_loss'])
        
        avg_losses = [sum(losses)/len(losses) for losses in schedule_types.values()]
        ax3.bar(schedule_types.keys(), avg_losses, color='lightblue', alpha=0.8)
        ax3.set_xlabel('Schedule Type')
        ax3.set_ylabel('Average Validation Loss')
        ax3.set_title('Performance by Schedule Type')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Learning Rate vs Performance
        if len(results) > 0 and 'lr_history' in results[0]:
            # Show LR curves for top 3 experiments
            for i, result in enumerate(results[:3]):
                lr_history = result.get('lr_history', [])
                if lr_history:
                    steps, lrs = zip(*lr_history)
                    ax4.plot(steps, lrs, label=f"{result['experiment_name']}", linewidth=2)
            
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedules (Top 3)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "lr_schedule_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Analysis visualization saved to: {plot_file}")
    
    def save_analysis_report(self, results):
        """Save detailed analysis report"""
        report_file = os.path.join(self.results_dir, "lr_schedule_analysis_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Learning Rate Schedule Optimization Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Objective\n\n")
            f.write("Optimize learning rate scheduling to achieve additional validation loss improvements ")
            f.write("beyond the 6.5% gain from the ablation study.\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Successful experiments: {len(results)}\n\n")
            
            f.write("## Top 5 Performing LR Schedules\n\n")
            f.write("| Rank | Experiment | Val Loss | Val Accuracy | Schedule Type | Improvement |\n")
            f.write("|------|------------|----------|--------------|---------------|-------------|\n")
            
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
                
                schedule_type = result.get('schedule_config', {}).get('schedule_type', 'unknown')
                f.write(f"| {i} | {result['experiment_name']} | {result['results']['val_loss']:.4f} | "
                       f"{result['results']['val_accuracy']:.4f} | {schedule_type} | {improvement} |\n")
            
            f.write("\n## Best Configuration\n\n")
            best = results[0]
            f.write(f"**Experiment**: {best['experiment_name']}\n")
            f.write(f"**Description**: {best['description']}\n")
            f.write(f"**Validation Loss**: {best['results']['val_loss']:.4f}\n")
            f.write(f"**Validation Accuracy**: {best['results']['val_accuracy']:.4f}\n")
            if baseline_loss:
                improvement = ((baseline_loss - best['results']['val_loss']) / baseline_loss * 100)
                f.write(f"**Improvement**: {improvement:.1f}%\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the learning rate schedule optimization:\n\n")
            f.write("1. **Optimal Schedule**: Use the top-performing schedule configuration\n")
            f.write("2. **Key Insights**: Analyze which schedule parameters contribute most to improvement\n")
            f.write("3. **Implementation**: Apply the best schedule to the main branch\n")
        
        print(f"📄 Analysis report saved to: {report_file}")


def main():
    """Main function to run LR schedule optimization study"""
    print("🔬 Learning Rate Schedule Optimization Study")
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
    
    # Use optimized config from ablation study as base
    config = MoEModelConfig(vocab_size=vocab_size)
    
    # Get experiments (run subset for efficiency)
    all_experiments = get_lr_schedule_experiments()
    key_experiments = all_experiments[:8]  # Run first 8 experiments
    
    print(f"\n🧪 Running {len(key_experiments)} key LR schedule experiments")
    
    # Run experiments
    runner = LRScheduleExperimentRunner()
    runner.run_all_experiments(key_experiments, config, train_loader, val_loader)


if __name__ == "__main__":
    main()
