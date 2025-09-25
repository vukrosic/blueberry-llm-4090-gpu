"""
Learning Rate Schedule Optimization Experiment
Building on the 6.5% validation loss improvement from ablation study
Focus: Further optimize learning rate scheduling for additional gains
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import math
import torch


@dataclass
class LRScheduleExperiment:
    name: str
    description: str
    warmup_ratio: float
    min_lr_ratio: float
    schedule_type: str
    expected_improvement: str


def create_cosine_schedule(warmup_ratio: float, min_lr_ratio: float, max_steps: int) -> Callable:
    """Create cosine annealing schedule with configurable warmup and minimum LR"""
    def lr_lambda(step):
        warmup_steps = int(max_steps * warmup_ratio)
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def create_linear_schedule(warmup_ratio: float, min_lr_ratio: float, max_steps: int) -> Callable:
    """Create linear decay schedule with configurable warmup"""
    def lr_lambda(step):
        warmup_steps = int(max_steps * warmup_ratio)
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 1.0 - progress * (1 - min_lr_ratio)
    return lr_lambda


def create_exponential_schedule(warmup_ratio: float, min_lr_ratio: float, max_steps: int) -> Callable:
    """Create exponential decay schedule"""
    def lr_lambda(step):
        warmup_steps = int(max_steps * warmup_ratio)
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * math.exp(-3 * progress)
    return lr_lambda


def create_polynomial_schedule(warmup_ratio: float, min_lr_ratio: float, max_steps: int, power: float = 2.0) -> Callable:
    """Create polynomial decay schedule"""
    def lr_lambda(step):
        warmup_steps = int(max_steps * warmup_ratio)
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress) ** power
    return lr_lambda


def get_lr_schedule_experiments() -> List[LRScheduleExperiment]:
    """Generate learning rate schedule experiments"""
    experiments = []
    
    # Current baseline (from previous optimization)
    experiments.append(LRScheduleExperiment(
        name="baseline_optimized",
        description="Current optimized schedule (10% warmup, 5% min LR, cosine)",
        warmup_ratio=0.1,
        min_lr_ratio=0.05,
        schedule_type="cosine",
        expected_improvement="Baseline from ablation study"
    ))
    
    # Experiment 1: Different warmup ratios with cosine
    experiments.append(LRScheduleExperiment(
        name="cosine_warmup_05",
        description="Shorter warmup (5%) with cosine annealing",
        warmup_ratio=0.05,
        min_lr_ratio=0.05,
        schedule_type="cosine",
        expected_improvement="Faster initial learning, better convergence"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="cosine_warmup_15",
        description="Longer warmup (15%) with cosine annealing",
        warmup_ratio=0.15,
        min_lr_ratio=0.05,
        schedule_type="cosine",
        expected_improvement="More stable initial training"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="cosine_warmup_20",
        description="Extended warmup (20%) with cosine annealing",
        warmup_ratio=0.20,
        min_lr_ratio=0.05,
        schedule_type="cosine",
        expected_improvement="Very stable warmup, better for complex models"
    ))
    
    # Experiment 2: Different minimum LR ratios
    experiments.append(LRScheduleExperiment(
        name="cosine_minlr_01",
        description="Very low minimum LR (1%) with cosine",
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        schedule_type="cosine",
        expected_improvement="More aggressive decay, better final performance"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="cosine_minlr_02",
        description="Low minimum LR (2%) with cosine",
        warmup_ratio=0.1,
        min_lr_ratio=0.02,
        schedule_type="cosine",
        expected_improvement="Aggressive decay while maintaining stability"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="cosine_minlr_10",
        description="Higher minimum LR (10%) with cosine",
        warmup_ratio=0.1,
        min_lr_ratio=0.10,
        schedule_type="cosine",
        expected_improvement="Less aggressive decay, stable learning"
    ))
    
    # Experiment 3: Different schedule types
    experiments.append(LRScheduleExperiment(
        name="linear_decay",
        description="Linear decay schedule (10% warmup, 5% min LR)",
        warmup_ratio=0.1,
        min_lr_ratio=0.05,
        schedule_type="linear",
        expected_improvement="Simpler schedule, potentially more stable"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="exponential_decay",
        description="Exponential decay schedule (10% warmup, 5% min LR)",
        warmup_ratio=0.1,
        min_lr_ratio=0.05,
        schedule_type="exponential",
        expected_improvement="Fast initial decay, stable later training"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="polynomial_decay",
        description="Polynomial decay schedule (10% warmup, 5% min LR)",
        warmup_ratio=0.1,
        min_lr_ratio=0.05,
        schedule_type="polynomial",
        expected_improvement="Smooth decay, balanced learning"
    ))
    
    # Experiment 4: Best combinations
    experiments.append(LRScheduleExperiment(
        name="aggressive_cosine",
        description="Aggressive cosine (5% warmup, 1% min LR)",
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
        schedule_type="cosine",
        expected_improvement="Fast learning with aggressive decay"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="conservative_cosine",
        description="Conservative cosine (15% warmup, 2% min LR)",
        warmup_ratio=0.15,
        min_lr_ratio=0.02,
        schedule_type="cosine",
        expected_improvement="Stable warmup with good final performance"
    ))
    
    experiments.append(LRScheduleExperiment(
        name="optimal_polynomial",
        description="Optimized polynomial (12% warmup, 3% min LR, power=1.5)",
        warmup_ratio=0.12,
        min_lr_ratio=0.03,
        schedule_type="polynomial_15",
        expected_improvement="Balanced approach with smooth transitions"
    ))
    
    return experiments


def get_schedule_function(experiment: LRScheduleExperiment, max_steps: int) -> Callable:
    """Get the appropriate schedule function for an experiment"""
    if experiment.schedule_type == "cosine":
        return create_cosine_schedule(experiment.warmup_ratio, experiment.min_lr_ratio, max_steps)
    elif experiment.schedule_type == "linear":
        return create_linear_schedule(experiment.warmup_ratio, experiment.min_lr_ratio, max_steps)
    elif experiment.schedule_type == "exponential":
        return create_exponential_schedule(experiment.warmup_ratio, experiment.min_lr_ratio, max_steps)
    elif experiment.schedule_type == "polynomial":
        return create_polynomial_schedule(experiment.warmup_ratio, experiment.min_lr_ratio, max_steps, power=2.0)
    elif experiment.schedule_type == "polynomial_15":
        return create_polynomial_schedule(experiment.warmup_ratio, experiment.min_lr_ratio, max_steps, power=1.5)
    else:
        raise ValueError(f"Unknown schedule type: {experiment.schedule_type}")


def visualize_schedules(experiments: List[LRScheduleExperiment], max_steps: int = 20, base_lr: float = 0.065):
    """Visualize different learning rate schedules"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    steps = np.arange(max_steps)
    plt.figure(figsize=(15, 10))
    
    for i, experiment in enumerate(experiments[:6]):  # Show first 6 for clarity
        schedule_fn = get_schedule_function(experiment, max_steps)
        lrs = [schedule_fn(step) * base_lr for step in steps]
        
        plt.plot(steps, lrs, marker='o', label=f"{experiment.name}", linewidth=2, markersize=4)
    
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt


if __name__ == "__main__":
    experiments = get_lr_schedule_experiments()
    print(f"🔬 Learning Rate Schedule Experiments: {len(experiments)} configurations")
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:2}. {exp.name:<20} | {exp.description}")
    
    # Visualize schedules
    plt = visualize_schedules(experiments)
    plt.savefig("experiments/lr_schedule_optimization/lr_schedules_preview.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Schedule visualization saved to lr_schedules_preview.png")
