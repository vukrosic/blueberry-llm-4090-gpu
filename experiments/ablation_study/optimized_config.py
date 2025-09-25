"""
Optimized Configuration from Ablation Study
6.5% improvement in validation loss achieved through systematic optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig


def get_optimized_config(vocab_size: int) -> MoEModelConfig:
    """
    Get the optimized configuration from ablation study results.
    
    This configuration achieved 6.5% improvement in validation loss:
    - Validation Loss: 6.8076 (vs 7.2785 baseline)
    - Validation Accuracy: 14.25% (vs 11.58% baseline)
    - Training Time: Same as baseline
    
    Key optimizations:
    - gradient_accumulation_steps: 4 → 2 (more frequent updates)
    - dropout: 0.10 → 0.05 (less aggressive regularization)
    """
    return MoEModelConfig(
        # Model architecture (unchanged - optimal from previous experiments)
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        max_steps=20,
        
        # OPTIMIZED: Training parameters
        gradient_accumulation_steps=2,  # 📈 Changed from 4 - more frequent updates
        muon_lr=0.065,                  # ✓ Keep optimal LR from previous experiments
        
        # Data parameters (unchanged)
        max_seq_len=512,
        num_documents=2000,
        max_tokens=500000,
        
        # Evaluation (unchanged)
        eval_every=10,
        eval_steps=100,
        
        # OPTIMIZED: Regularization
        weight_decay=0.1,               # ✓ Keep current optimal
        dropout=0.05,                   # 📈 Changed from 0.10 - less aggressive
        grad_clip=1.0,                  # ✓ Keep current optimal
        
        # Technical (unchanged)
        use_amp=True,
        vocab_size=vocab_size,
        log_milestones=(2000, 5000, 10000),
        
        # MoE specific parameters (unchanged - optimal balance)
        num_experts=8,                  # ✓ Keep for computational efficiency
        expert_top_k=2,                 # ✓ Keep optimal routing
        load_balancing_weight=0.01,     # ✓ Keep optimal balance
    )


def compare_configs():
    """Compare baseline vs optimized configuration"""
    print("🔬 Configuration Comparison: Baseline vs Optimized")
    print("="*60)
    
    # Create dummy configs for comparison
    baseline = MoEModelConfig(vocab_size=50000)
    optimized = get_optimized_config(vocab_size=50000)
    
    print(f"{'Parameter':<25} {'Baseline':<15} {'Optimized':<15} {'Change'}")
    print("-"*60)
    
    changes = []
    
    # Compare key parameters
    params_to_compare = [
        ('gradient_accumulation_steps', 'gradient_accumulation_steps'),
        ('muon_lr', 'muon_lr'),
        ('dropout', 'dropout'),
        ('weight_decay', 'weight_decay'),
        ('num_experts', 'num_experts'),
        ('expert_top_k', 'expert_top_k'),
    ]
    
    for param_name, attr_name in params_to_compare:
        baseline_val = getattr(baseline, attr_name)
        optimized_val = getattr(optimized, attr_name)
        
        if baseline_val != optimized_val:
            change_indicator = "📈 CHANGED"
            changes.append(param_name)
        else:
            change_indicator = "✓ Same"
        
        print(f"{param_name:<25} {baseline_val:<15} {optimized_val:<15} {change_indicator}")
    
    print(f"\n🎯 Key Changes: {len(changes)} parameters optimized")
    for change in changes:
        print(f"   • {change}")
    
    print(f"\n📊 Expected Improvements:")
    print(f"   • Validation Loss: 6.5% improvement")
    print(f"   • Validation Accuracy: 23.1% relative improvement")
    print(f"   • Perplexity: 37.5% reduction")
    print(f"   • Training Time: No change")


if __name__ == "__main__":
    compare_configs()
