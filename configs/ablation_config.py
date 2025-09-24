from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import itertools
from configs.moe_config import MoEModelConfig


@dataclass
class LearningRateConfig:
    """Configuration for learning rate ablation study"""
    muon_lr_values: List[float] = None
    adamw_lr_ratio_values: List[float] = None  # Ratio relative to muon_lr
    momentum_values: List[float] = None
    weight_decay_values: List[float] = None
    
    def __post_init__(self):
        if self.muon_lr_values is None:
            # Comprehensive range covering orders of magnitude
            self.muon_lr_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        
        if self.adamw_lr_ratio_values is None:
            # AdamW typically uses 10x smaller LR than Muon
            self.adamw_lr_ratio_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        if self.momentum_values is None:
            # Test different momentum values for Muon
            self.momentum_values = [0.9, 0.95, 0.99]
        
        if self.weight_decay_values is None:
            # Test different weight decay values
            self.weight_decay_values = [0.01, 0.1, 0.5]


@dataclass
class AblationStudyConfig:
    """Configuration for the complete ablation study"""
    base_config: MoEModelConfig
    lr_config: LearningRateConfig
    experiment_name: str = "learning_rate_ablation"
    output_dir: str = "ablation_results"
    save_models: bool = False
    early_stopping_patience: int = 5
    min_improvement: float = 0.001
    
    # Experiment control
    max_experiments: Optional[int] = None  # None for all combinations
    random_seed: int = 42
    
    def get_experiment_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters to test"""
        combinations = list(itertools.product(
            self.lr_config.muon_lr_values,
            self.lr_config.adamw_lr_ratio_values,
            self.lr_config.momentum_values,
            self.lr_config.weight_decay_values
        ))
        
        experiments = []
        for i, (muon_lr, adamw_ratio, momentum, weight_decay) in enumerate(combinations):
            if self.max_experiments and i >= self.max_experiments:
                break
                
            experiment_config = {
                'experiment_id': f"exp_{i:03d}",
                'muon_lr': muon_lr,
                'adamw_lr': muon_lr * adamw_ratio,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'config': self._create_experiment_config(muon_lr, adamw_ratio, momentum, weight_decay)
            }
            experiments.append(experiment_config)
        
        return experiments
    
    def _create_experiment_config(self, muon_lr: float, adamw_ratio: float, 
                                 momentum: float, weight_decay: float) -> MoEModelConfig:
        """Create a MoEModelConfig with specific hyperparameters"""
        config_dict = self.base_config.__dict__.copy()
        config_dict.update({
            'muon_lr': muon_lr,
            'weight_decay': weight_decay
        })
        
        # Create new config instance
        config = MoEModelConfig(**config_dict)
        
        # Store additional experiment-specific parameters
        config.experiment_momentum = momentum
        config.experiment_adamw_lr_ratio = adamw_ratio
        
        return config


def create_default_ablation_config() -> AblationStudyConfig:
    """Create a default ablation study configuration"""
    base_config = MoEModelConfig(
        max_steps=50,  # Reduced for faster ablation study
        eval_every=5,
        batch_size=16  # Smaller batch for more experiments
    )
    
    lr_config = LearningRateConfig()
    
    return AblationStudyConfig(
        base_config=base_config,
        lr_config=lr_config,
        experiment_name="comprehensive_lr_ablation",
        output_dir="ablation_results",
        max_experiments=36  # Reasonable number for comprehensive study
    )
