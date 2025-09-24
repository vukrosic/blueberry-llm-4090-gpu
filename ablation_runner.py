import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from configs.ablation_config import AblationStudyConfig, create_default_ablation_config
from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed


class AblationExperimentRunner:
    """Runs comprehensive ablation study on learning rates"""
    
    def __init__(self, config: AblationStudyConfig):
        self.config = config
        self.results = []
        self.start_time = None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create experiment subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{config.experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"🧪 Ablation Study: {config.experiment_name}")
        print(f"📁 Output directory: {self.experiment_dir}")
        print(f"🔢 Total experiments: {len(config.get_experiment_combinations())}")
    
    def run_study(self) -> Dict[str, Any]:
        """Run the complete ablation study"""
        self.start_time = time.time()
        
        # Load data once for all experiments
        print("\n📊 Loading dataset...")
        texts, tokenizer, tokens = self._load_data()
        
        # Get all experiment combinations
        experiments = self.config.get_experiment_combinations()
        
        print(f"\n🚀 Starting ablation study with {len(experiments)} experiments...")
        
        for i, experiment in enumerate(experiments):
            print(f"\n{'='*60}")
            print(f"🧪 Experiment {i+1}/{len(experiments)}: {experiment['experiment_id']}")
            print(f"{'='*60}")
            
            try:
                result = self._run_single_experiment(experiment, tokens)
                self.results.append(result)
                
                # Save intermediate results
                self._save_intermediate_results()
                
                print(f"✅ Experiment {experiment['experiment_id']} completed successfully")
                
            except Exception as e:
                print(f"❌ Experiment {experiment['experiment_id']} failed: {str(e)}")
                error_result = {
                    'experiment_id': experiment['experiment_id'],
                    'error': str(e),
                    'status': 'failed'
                }
                self.results.append(error_result)
        
        # Generate final analysis
        analysis = self._generate_analysis()
        
        # Save final results
        self._save_final_results(analysis)
        
        print(f"\n🎉 Ablation study completed!")
        print(f"⏱️  Total time: {time.time() - self.start_time:.1f} seconds")
        print(f"📊 Results saved to: {self.experiment_dir}")
        
        return analysis
    
    def _load_data(self):
        """Load and prepare data for experiments"""
        temp_config = MoEModelConfig()
        texts, tokenizer, tokens = load_and_cache_data(temp_config)
        return texts, tokenizer, tokens
    
    def _run_single_experiment(self, experiment: Dict[str, Any], tokens: List[int]) -> Dict[str, Any]:
        """Run a single experiment with specific hyperparameters"""
        experiment_id = experiment['experiment_id']
        config = experiment['config']
        
        print(f"📋 Hyperparameters:")
        print(f"   Muon LR: {experiment['muon_lr']:.4f}")
        print(f"   AdamW LR: {experiment['adamw_lr']:.4f} (ratio: {experiment['adamw_lr']/experiment['muon_lr']:.2f})")
        print(f"   Momentum: {experiment['momentum']:.2f}")
        print(f"   Weight Decay: {experiment['weight_decay']:.2f}")
        
        # Set seed for reproducibility
        set_seed(self.config.random_seed)
        
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
        
        # Run training with custom optimizer setup
        start_time = time.time()
        training_results = self._train_with_custom_config(config, train_loader, val_loader)
        training_time = time.time() - start_time
        
        # Compile results
        result = {
            'experiment_id': experiment_id,
            'hyperparameters': {
                'muon_lr': experiment['muon_lr'],
                'adamw_lr': experiment['adamw_lr'],
                'adamw_lr_ratio': experiment['adamw_lr'] / experiment['muon_lr'],
                'momentum': experiment['momentum'],
                'weight_decay': experiment['weight_decay']
            },
            'training_time': training_time,
            'training_results': training_results,
            'status': 'completed'
        }
        
        return result
    
    def _train_with_custom_config(self, config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader):
        """Train model with custom hyperparameters"""
        from models.moe_llm import MoEMinimalLLM
        from optimizers.muon import Muon
        from training.evaluation import evaluate_model
        import math
        
        # Initialize model
        model = MoEMinimalLLM(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup custom optimizers
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if (param.ndim == 2 and 
                'token_embedding' not in name and 
                'norm' not in name and 
                param.requires_grad):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        # Create optimizers with custom hyperparameters
        muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=config.experiment_momentum)
        adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr * config.experiment_adamw_lr_ratio, 
                                          weight_decay=config.weight_decay)
        
        optimizers = [muon_optimizer, adamw_optimizer]
        
        # Learning rate schedule
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = config.max_steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)
        
        # Training loop (simplified version)
        model.train()
        step = 0
        losses = []
        accuracies = []
        
        while step < config.max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= config.max_steps:
                    break
                    
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits, aux_loss = model(x)
                ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
                
                # Backward pass
                total_loss.backward()
                
                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                
                # Logging
                if step % 10 == 0:
                    with torch.no_grad():
                        predictions = logits.argmax(dim=-1)
                        accuracy = (predictions == y).float().mean().item()
                        losses.append(ce_loss.item())
                        accuracies.append(accuracy)
                
                step += 1
        
        # Final evaluation
        model.eval()
        eval_results = evaluate_model(model, val_loader, device, config.eval_steps)
        
        return {
            'final_loss': losses[-1] if losses else float('inf'),
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'eval_loss': eval_results.get('loss', float('inf')),
            'eval_accuracy': eval_results.get('accuracy', 0.0),
            'eval_perplexity': eval_results.get('perplexity', float('inf')),
            'training_losses': losses,
            'training_accuracies': accuracies
        }
    
    def _save_intermediate_results(self):
        """Save intermediate results to file"""
        results_file = self.experiment_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _save_final_results(self, analysis: Dict[str, Any]):
        """Save final results and analysis"""
        # Save raw results
        results_file = self.experiment_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save analysis
        analysis_file = self.experiment_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save configuration
        config_file = self.experiment_dir / "config.json"
        config_dict = {
            'experiment_name': self.config.experiment_name,
            'base_config': self.config.base_config.__dict__,
            'lr_config': self.config.lr_config.__dict__,
            'total_experiments': len(self.results)
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of results"""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Filter successful experiments
        successful_results = [r for r in self.results if r.get('status') == 'completed']
        
        if not successful_results:
            return {'error': 'No successful experiments'}
        
        # Extract metrics
        metrics = {
            'final_loss': [r['training_results']['final_loss'] for r in successful_results],
            'final_accuracy': [r['training_results']['final_accuracy'] for r in successful_results],
            'eval_loss': [r['training_results']['eval_loss'] for r in successful_results],
            'eval_accuracy': [r['training_results']['eval_accuracy'] for r in successful_results],
            'eval_perplexity': [r['training_results']['eval_perplexity'] for r in successful_results],
            'training_time': [r['training_time'] for r in successful_results]
        }
        
        # Find best performing experiments
        best_accuracy_idx = np.argmax(metrics['eval_accuracy'])
        best_loss_idx = np.argmin(metrics['eval_loss'])
        
        analysis = {
            'summary': {
                'total_experiments': len(self.results),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(self.results) - len(successful_results),
                'total_time': time.time() - self.start_time
            },
            'best_performers': {
                'best_accuracy': {
                    'experiment_id': successful_results[best_accuracy_idx]['experiment_id'],
                    'hyperparameters': successful_results[best_accuracy_idx]['hyperparameters'],
                    'eval_accuracy': metrics['eval_accuracy'][best_accuracy_idx],
                    'eval_loss': metrics['eval_loss'][best_accuracy_idx]
                },
                'best_loss': {
                    'experiment_id': successful_results[best_loss_idx]['experiment_id'],
                    'hyperparameters': successful_results[best_loss_idx]['hyperparameters'],
                    'eval_accuracy': metrics['eval_accuracy'][best_loss_idx],
                    'eval_loss': metrics['eval_loss'][best_loss_idx]
                }
            },
            'statistics': {
                'eval_accuracy': {
                    'mean': np.mean(metrics['eval_accuracy']),
                    'std': np.std(metrics['eval_accuracy']),
                    'min': np.min(metrics['eval_accuracy']),
                    'max': np.max(metrics['eval_accuracy'])
                },
                'eval_loss': {
                    'mean': np.mean(metrics['eval_loss']),
                    'std': np.std(metrics['eval_loss']),
                    'min': np.min(metrics['eval_loss']),
                    'max': np.max(metrics['eval_loss'])
                }
            },
            'all_results': successful_results
        }
        
        return analysis


def main():
    """Main function to run the ablation study"""
    config = create_default_ablation_config()
    runner = AblationExperimentRunner(config)
    analysis = runner.run_study()
    
    print("\n📊 Analysis Summary:")
    print(f"   Best Accuracy: {analysis['best_performers']['best_accuracy']['eval_accuracy']:.4f}")
    print(f"   Best Loss: {analysis['best_performers']['best_loss']['eval_loss']:.4f}")
    print(f"   Mean Accuracy: {analysis['statistics']['eval_accuracy']['mean']:.4f} ± {analysis['statistics']['eval_accuracy']['std']:.4f}")


if __name__ == "__main__":
    main()
