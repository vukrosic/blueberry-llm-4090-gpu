"""
MoE Routing Optimization Experiment
Building on 7.5% cumulative validation loss improvement
Focus: Optimize expert routing strategies for additional performance gains
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig


@dataclass
class MoERoutingExperiment:
    name: str
    description: str
    num_experts: int
    expert_top_k: int
    load_balancing_weight: float
    routing_noise_std: float
    expected_improvement: str


def get_baseline_moe_config(vocab_size: int) -> MoEModelConfig:
    """Get current optimized baseline configuration"""
    return MoEModelConfig(
        # Model architecture (optimized)
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=20,
        
        # Training parameters (optimized from previous studies)
        gradient_accumulation_steps=2,  # From ablation study
        muon_lr=0.065,
        
        # Data parameters
        max_seq_len=512,
        num_documents=2000,
        max_tokens=500000,
        
        # Evaluation
        eval_every=10,
        eval_steps=100,
        
        # Regularization (optimized from ablation study)
        weight_decay=0.1,
        dropout=0.05,  # From ablation study
        grad_clip=1.0,
        
        # Technical
        use_amp=True,
        vocab_size=vocab_size,
        log_milestones=(2000, 5000, 10000),
        
        # Current MoE parameters (baseline to optimize)
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,
    )


def get_moe_routing_experiments(vocab_size: int) -> List[MoERoutingExperiment]:
    """Generate comprehensive MoE routing optimization experiments"""
    experiments = []
    
    # Current baseline
    experiments.append(MoERoutingExperiment(
        name="baseline_optimized",
        description="Current optimized configuration (8 experts, top-2, 0.01 load balance)",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Baseline from previous optimizations"
    ))
    
    # Experiment 1: Different Top-K Routing Strategies
    experiments.append(MoERoutingExperiment(
        name="top1_routing",
        description="Top-1 expert routing for maximum specialization",
        num_experts=8,
        expert_top_k=1,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Better expert specialization, cleaner routing decisions"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="top3_routing",
        description="Top-3 expert routing for better coverage",
        num_experts=8,
        expert_top_k=3,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Better expert utilization, improved robustness"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="top4_routing",
        description="Top-4 expert routing for maximum coverage",
        num_experts=8,
        expert_top_k=4,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Maximum expert utilization, ensemble benefits"
    ))
    
    # Experiment 2: Different Expert Counts
    experiments.append(MoERoutingExperiment(
        name="4_experts_top2",
        description="Fewer experts (4) with top-2 routing",
        num_experts=4,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Reduced complexity, potentially better training"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="12_experts_top2",
        description="More experts (12) with top-2 routing",
        num_experts=12,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="More model capacity, better specialization"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="16_experts_top2",
        description="Many experts (16) with top-2 routing",
        num_experts=16,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.1,
        expected_improvement="Maximum capacity, finest specialization"
    ))
    
    # Experiment 3: Load Balancing Weight Optimization
    experiments.append(MoERoutingExperiment(
        name="low_load_balance",
        description="Lower load balancing weight (0.005) for less constraint",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.005,
        routing_noise_std=0.1,
        expected_improvement="More natural expert selection, better specialization"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="high_load_balance",
        description="Higher load balancing weight (0.02) for better utilization",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.02,
        routing_noise_std=0.1,
        expected_improvement="Better expert utilization, more stable training"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="very_high_load_balance",
        description="Very high load balancing weight (0.05) for forced utilization",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.05,
        routing_noise_std=0.1,
        expected_improvement="Forced expert utilization, uniform load distribution"
    ))
    
    # Experiment 4: Routing Noise Variations
    experiments.append(MoERoutingExperiment(
        name="no_routing_noise",
        description="No routing noise for deterministic expert selection",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.0,
        expected_improvement="More stable routing, deterministic expert selection"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="high_routing_noise",
        description="Higher routing noise (0.2) for more exploration",
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,
        routing_noise_std=0.2,
        expected_improvement="Better expert exploration, more diverse routing"
    ))
    
    # Experiment 5: Optimal Combinations
    experiments.append(MoERoutingExperiment(
        name="optimal_12_top1",
        description="12 experts with top-1 routing and optimized load balance",
        num_experts=12,
        expert_top_k=1,
        load_balancing_weight=0.015,
        routing_noise_std=0.1,
        expected_improvement="Maximum specialization with increased capacity"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="optimal_16_top3",
        description="16 experts with top-3 routing for maximum performance",
        num_experts=16,
        expert_top_k=3,
        load_balancing_weight=0.02,
        routing_noise_std=0.1,
        expected_improvement="Maximum capacity with robust routing"
    ))
    
    experiments.append(MoERoutingExperiment(
        name="efficient_6_top2",
        description="Efficient 6 experts with optimized load balance",
        num_experts=6,
        expert_top_k=2,
        load_balancing_weight=0.005,
        routing_noise_std=0.05,
        expected_improvement="Balanced efficiency and performance"
    ))
    
    return experiments


def create_moe_config_from_experiment(experiment: MoERoutingExperiment, vocab_size: int) -> MoEModelConfig:
    """Create MoEModelConfig from experiment specification"""
    config = get_baseline_moe_config(vocab_size)
    
    # Apply experiment-specific MoE parameters
    config.num_experts = experiment.num_experts
    config.expert_top_k = experiment.expert_top_k
    config.load_balancing_weight = experiment.load_balancing_weight
    
    return config


def analyze_moe_complexity(experiments: List[MoERoutingExperiment]) -> None:
    """Analyze computational complexity of different MoE configurations"""
    print("🔍 MoE Routing Complexity Analysis")
    print("="*70)
    print(f"{'Experiment':<25} {'Experts':<8} {'Top-K':<6} {'Active %':<10} {'Complexity':<12}")
    print("-"*70)
    
    for exp in experiments:
        active_ratio = exp.expert_top_k / exp.num_experts
        
        # Estimate relative complexity (higher is more complex)
        complexity_score = exp.num_experts * exp.expert_top_k * exp.load_balancing_weight * 100
        complexity_level = "Low" if complexity_score < 20 else "Medium" if complexity_score < 40 else "High"
        
        print(f"{exp.name:<25} {exp.num_experts:<8} {exp.expert_top_k:<6} {active_ratio:<10.1%} {complexity_level:<12}")


def estimate_parameter_counts(experiments: List[MoERoutingExperiment], d_model: int = 384, d_ff: int = 1536) -> None:
    """Estimate parameter counts for different MoE configurations"""
    print(f"\n📊 Parameter Count Analysis (d_model={d_model}, d_ff={d_ff})")
    print("="*80)
    print(f"{'Experiment':<25} {'Experts':<8} {'Expert Params':<15} {'Total Params':<15} {'Active %':<10}")
    print("-"*80)
    
    # Base model parameters (non-expert)
    base_params = 22_436_736  # From current model
    
    for exp in experiments:
        # Each expert has: d_model -> d_ff -> d_model (2 linear layers)
        params_per_expert = d_model * d_ff + d_ff * d_model  # Input and output projections
        total_expert_params = params_per_expert * exp.num_experts
        total_params = base_params + total_expert_params
        
        # Active parameters during forward pass
        active_expert_params = params_per_expert * exp.expert_top_k
        active_params = base_params + active_expert_params
        active_ratio = active_params / total_params
        
        print(f"{exp.name:<25} {exp.num_experts:<8} {total_expert_params:<15,} {total_params:<15,} {active_ratio:<10.1%}")


if __name__ == "__main__":
    # Example usage
    experiments = get_moe_routing_experiments(vocab_size=50000)
    
    print(f"🧪 MoE Routing Optimization Experiments: {len(experiments)} configurations")
    print("="*80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:2}. {exp.name:<25} | {exp.description}")
    
    print("\n")
    analyze_moe_complexity(experiments)
    estimate_parameter_counts(experiments)
    
    print(f"\n🎯 Goal: Improve upon current 7.5% cumulative validation loss improvement")
    print(f"📈 Focus: Expert routing efficiency and utilization optimization")
