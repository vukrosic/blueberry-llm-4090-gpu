# Blueberry LLM 🫐 - T4 Optimized

A Tesla T4 GPU-optimized Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible on T4 GPUs** - optimized specifically for Tesla T4 GPU performance with automatic configuration and hardware optimization to create state of the art LLM on single T4 GPU.

## Quick Start

## Contributing

We welcome contributions! Fork the repo, experiment with different architectures, and submit PRs with your findings.

## Vision

Any company or person (even with no technical experience) should be able to download this repository and run it on their Tesla T4 GPU setup. The system will automatically detect your T4 GPU configuration, tune hyperparameters for optimal T4 performance, and run the best possible training with or without manual configuration from your side.

## 📅 TIMELINE

*Community experiments and findings will be documented here*

### Sep 25 2025
- **🚀 Learning Rate Optimization Breakthrough**: Discovered optimal learning rate (0.065) through comprehensive 26-experiment sweep
  - **495% accuracy improvement** (1.6% → 9.3%)
  - **22.5% loss reduction** (9.567 → 7.414)
  - **88% perplexity reduction** (14,286 → 1,659)
  - Applied to main branch for immediate production benefits
  - ![Learning Rate Comparison](experiments/learning_rate_optimization/learning_rate_comparison_extended.png)
  - Full analysis: [Learning Rate Optimization Results](experiments/learning_rate_optimization/LEARNING_RATE_OPTIMIZATION_RESULTS.md)

### Sep 22 2025
- **Repository Launch**: Initial T4-optimized MoE implementation
- *Your experiment results will appear here when you submit them*

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@software{blueberry_llm_t4,
  title={Blueberry LLM: Pretrain LLM On A Single T4 GPU,
  author={Vuk Rosić},
  year={2025},
  url={https://github.com/Open-Superintelligence-Lab/blueberry-llm-t4-gpu},
  note={Tesla T4 GPU-optimized LLM for accessible LLM training}
}
```
