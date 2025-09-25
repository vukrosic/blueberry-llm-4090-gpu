"""
Ablation Study Configurations for Blueberry LLM
Focus: Improving validation loss through systematic experiments
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from configs.moe_config import MoEModelConfig


@dataclass
class AblationExperiment:
    name: str
    description: str
    config: MoEModelConfig
    expected_improvement: str


def get_baseline_config(vocab_size: int) -> MoEModelConfig:
    """Baseline configuration (current optimized settings)"""
    return MoEModelConfig(
        # Model architecture
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=20,
        
        # Training parameters
        gradient_accumulation_steps=4,
        muon_lr=0.065,
        
        # Data parameters
        max_seq_len=512,
        num_documents=2000,
        max_tokens=500000,
        
        # Evaluation
        eval_every=10,
        eval_steps=100,
        
        # Regularization
        weight_decay=0.1,
        dropout=0.1,
        grad_clip=1.0,
        
        # Technical
        use_amp=True,
        vocab_size=vocab_size,
        log_milestones=(2000, 5000, 10000),
        
        # MoE specific parameters
        num_experts=8,
        expert_top_k=2,
        load_balancing_weight=0.01,
    )


def get_ablation_experiments(vocab_size: int) -> List[AblationExperiment]:
    """Generate all ablation experiments"""
    experiments = []
    
    # Experiment 1: Gradient Accumulation Optimization
    config1 = get_baseline_config(vocab_size)
    config1.gradient_accumulation_steps = 2
    experiments.append(AblationExperiment(
        name="grad_accum_2",
        description="Test gradient accumulation steps = 2 for faster convergence",
        config=config1,
        expected_improvement="Better gradient updates, faster convergence"
    ))
    
    config1b = get_baseline_config(vocab_size)
    config1b.gradient_accumulation_steps = 8
    experiments.append(AblationExperiment(
        name="grad_accum_8",
        description="Test gradient accumulation steps = 8 for more stable training",
        config=config1b,
        expected_improvement="More stable gradients, better generalization"
    ))
    
    # Experiment 2: MoE Routing Strategies
    config2a = get_baseline_config(vocab_size)
    config2a.expert_top_k = 1
    experiments.append(AblationExperiment(
        name="moe_top1",
        description="Test top-1 expert routing for specialization",
        config=config2a,
        expected_improvement="Better expert specialization, lower loss"
    ))
    
    config2b = get_baseline_config(vocab_size)
    config2b.expert_top_k = 3
    experiments.append(AblationExperiment(
        name="moe_top3",
        description="Test top-3 expert routing for better coverage",
        config=config2b,
        expected_improvement="Better expert utilization, improved performance"
    ))
    
    config2c = get_baseline_config(vocab_size)
    config2c.num_experts = 12
    config2c.expert_top_k = 2
    experiments.append(AblationExperiment(
        name="moe_12_experts",
        description="Test 12 experts instead of 8 for more capacity",
        config=config2c,
        expected_improvement="More model capacity, better task specialization"
    ))
    
    # Experiment 3: Learning Rate Schedule Variations
    config3a = get_baseline_config(vocab_size)
    config3a.muon_lr = 0.08  # Higher LR
    experiments.append(AblationExperiment(
        name="lr_008",
        description="Test higher learning rate (0.08) for faster learning",
        config=config3a,
        expected_improvement="Faster initial learning, better convergence"
    ))
    
    config3b = get_baseline_config(vocab_size)
    config3b.muon_lr = 0.05  # Lower LR
    experiments.append(AblationExperiment(
        name="lr_005",
        description="Test lower learning rate (0.05) for stability",
        config=config3b,
        expected_improvement="More stable training, better generalization"
    ))
    
    # Experiment 4: Regularization Techniques
    config4a = get_baseline_config(vocab_size)
    config4a.dropout = 0.05  # Lower dropout
    experiments.append(AblationExperiment(
        name="dropout_005",
        description="Test lower dropout (0.05) for less regularization",
        config=config4a,
        expected_improvement="Less overfitting, better training loss"
    ))
    
    config4b = get_baseline_config(vocab_size)
    config4b.dropout = 0.15  # Higher dropout
    experiments.append(AblationExperiment(
        name="dropout_015",
        description="Test higher dropout (0.15) for better generalization",
        config=config4b,
        expected_improvement="Better generalization, lower validation loss"
    ))
    
    config4c = get_baseline_config(vocab_size)
    config4c.weight_decay = 0.05  # Lower weight decay
    experiments.append(AblationExperiment(
        name="weight_decay_005",
        description="Test lower weight decay (0.05) for less regularization",
        config=config4c,
        expected_improvement="Less regularization, better training performance"
    ))
    
    config4d = get_baseline_config(vocab_size)
    config4d.weight_decay = 0.2  # Higher weight decay
    experiments.append(AblationExperiment(
        name="weight_decay_02",
        description="Test higher weight decay (0.2) for better generalization",
        config=config4d,
        expected_improvement="Better generalization, lower validation loss"
    ))
    
    # Experiment 5: Model Architecture Variations
    config5a = get_baseline_config(vocab_size)
    config5a.n_layers = 8  # More layers
    experiments.append(AblationExperiment(
        name="layers_8",
        description="Test 8 layers instead of 6 for more depth",
        config=config5a,
        expected_improvement="More model capacity, better representation learning"
    ))
    
    config5b = get_baseline_config(vocab_size)
    config5b.d_model = 512  # Larger model
    config5b.n_heads = 8  # Keep same head count
    config5b.d_ff = 2048  # Scale FF dimension
    experiments.append(AblationExperiment(
        name="d_model_512",
        description="Test larger model dimension (512) for more capacity",
        config=config5b,
        expected_improvement="More model capacity, better performance"
    ))
    
    config5c = get_baseline_config(vocab_size)
    config5c.n_heads = 12  # More attention heads
    config5c.d_model = 384  # Keep same model dim
    experiments.append(AblationExperiment(
        name="heads_12",
        description="Test 12 attention heads instead of 8",
        config=config5c,
        expected_improvement="Better attention patterns, improved performance"
    ))
    
    # Experiment 6: Advanced Optimizations
    config6a = get_baseline_config(vocab_size)
    config6a.grad_clip = 0.5  # Lower gradient clipping
    experiments.append(AblationExperiment(
        name="grad_clip_05",
        description="Test lower gradient clipping (0.5) for gentler updates",
        config=config6a,
        expected_improvement="More stable training, better convergence"
    ))
    
    config6b = get_baseline_config(vocab_size)
    config6b.load_balancing_weight = 0.02  # Higher load balancing
    experiments.append(AblationExperiment(
        name="load_balancing_002",
        description="Test higher load balancing weight (0.02) for better expert utilization",
        config=config6b,
        expected_improvement="Better expert utilization, improved performance"
    ))
    
    return experiments


def get_best_combination_experiments(vocab_size: int) -> List[AblationExperiment]:
    """Generate experiments combining the best individual improvements"""
    experiments = []
    
    # Best combination 1: Lower dropout + higher learning rate + top-1 routing
    config_best1 = get_baseline_config(vocab_size)
    config_best1.dropout = 0.05
    config_best1.muon_lr = 0.08
    config_best1.expert_top_k = 1
    experiments.append(AblationExperiment(
        name="best_combo_1",
        description="Best combo: dropout=0.05, lr=0.08, top-1 routing",
        config=config_best1,
        expected_improvement="Combined benefits of individual improvements"
    ))
    
    # Best combination 2: More experts + lower gradient accumulation + higher weight decay
    config_best2 = get_baseline_config(vocab_size)
    config_best2.num_experts = 12
    config_best2.gradient_accumulation_steps = 2
    config_best2.weight_decay = 0.2
    experiments.append(AblationExperiment(
        name="best_combo_2",
        description="Best combo: 12 experts, grad_accum=2, weight_decay=0.2",
        config=config_best2,
        expected_improvement="Combined benefits of capacity and regularization"
    ))
    
    return experiments
