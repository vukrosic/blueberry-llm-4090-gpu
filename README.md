# Blueberry LLM 🫐 - T4 Optimized

A Tesla T4 GPU-optimized Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible on T4 GPUs** - optimized specifically for Tesla T4 GPU performance with automatic configuration and hardware optimization to create state of the art LLM on single T4 GPU.

## 📖 **Complete Learning Rate Tutorial**

**[Learning Rate Mastery Tutorial](LEARNING_RATE_MASTERY_TUTORIAL.md)** - A comprehensive guide covering our complete learning rate optimization journey, from systematic experimentation to production deployment. Learn how we achieved **497% accuracy improvement** through rigorous hyperparameter optimization.

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
  - Comprehensive tutorial: [Learning Rate Tutorial Guide](https://github.com/Open-Superintelligence-Lab/blueberry-llm-t4-gpu/blob/ablation-study-learning-rates/LEARNING_RATE_TUTORIAL_GUIDE.md) (26-experiment methodology)

- **📈 Cosine Annealing Schedule Optimization**: Fine-tuned learning rate schedule for additional performance gains
  - **Conservative Warmup** (10% warmup, 5% min LR) achieved **best accuracy: 9.4%**
  - **2.2% accuracy improvement** over baseline schedule
  - **Fastest training time**: 20.5s
  - Tested 5 different cosine annealing configurations
  - ![Cosine Annealing Results](experiments/cosine_annealing/cosine_annealing_results_comparison.png)
  - Analysis: [Cosine Annealing Results](experiments/cosine_annealing/RESULTS_ANALYSIS.md)

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
