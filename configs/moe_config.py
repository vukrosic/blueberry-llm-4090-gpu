from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 20

    # Training parameters - OPTIMIZED from ablation study (6.5% validation loss improvement)
    gradient_accumulation_steps: int = 2  # Changed from 4 - more frequent updates
    muon_lr: float = 0.065

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 10
    eval_steps: int = 100

    # Regularization - OPTIMIZED from ablation study
    weight_decay: float = 0.1
    dropout: float = 0.05  # Changed from 0.1 - less aggressive regularization
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters - OPTIMIZED from MoE routing study (+1.8% improvement)
    num_experts: int = 16  # Changed from 8 - more model capacity and specialization
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
