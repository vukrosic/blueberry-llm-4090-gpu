# Learning Rate Ablation Study

This directory contains a comprehensive ablation study framework for analyzing learning rates in the MoE (Mixture of Experts) model. The study systematically tests different combinations of hyperparameters to find optimal learning rate configurations.

## 🎯 Study Overview

The ablation study examines the following hyperparameters:

- **Muon Learning Rate**: Primary learning rate for the Muon optimizer (0.001 to 0.3)
- **AdamW Learning Rate Ratio**: Ratio of AdamW LR relative to Muon LR (0.01 to 0.5)
- **Momentum**: Momentum parameter for Muon optimizer (0.9 to 0.99)
- **Weight Decay**: Regularization strength (0.01 to 0.5)

## 📁 Files Structure

```
├── configs/
│   └── ablation_config.py          # Configuration classes for ablation study
├── ablation_runner.py             # Main experiment runner
├── ablation_analyzer.py           # Results analysis and visualization
├── run_ablation_study.py          # Main script to run the study
└── ablation_results/              # Output directory (created automatically)
```

## 🚀 Quick Start

### Run a Quick Study (4 experiments)
```bash
python run_ablation_study.py --quick
```

### Run Full Study (90 experiments)
```bash
python run_ablation_study.py
```

### Run Limited Study
```bash
python run_ablation_study.py --max-experiments 20
```

### Analyze Existing Results
```bash
python run_ablation_study.py --analyze-only --results-dir ablation_results/experiment_name_timestamp
```

## 📊 Study Configuration

### Default Hyperparameter Ranges

- **Muon LR**: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
- **AdamW LR Ratio**: [0.01, 0.05, 0.1, 0.2, 0.5]
- **Momentum**: [0.9, 0.95, 0.99]
- **Weight Decay**: [0.01, 0.1, 0.5]

**Total Combinations**: 6 × 5 × 3 × 3 = 270 experiments (can be limited)

### Model Configuration

- **Architecture**: 384d, 6L, 8H, 1536ff
- **MoE**: 8 experts, top-2 routing
- **Training Steps**: 50 (reduced for ablation study)
- **Batch Size**: 16
- **Evaluation**: Every 5 steps

## 📈 Output and Analysis

### Generated Files

Each experiment run creates:

```
ablation_results/
└── learning_rate_ablation_YYYYMMDD_HHMMSS/
    ├── final_results.json           # Raw experiment results
    ├── analysis.json                 # Statistical analysis
    ├── config.json                   # Study configuration
    ├── comprehensive_report.json     # Detailed analysis
    ├── report.md                     # Human-readable report
    ├── performance_distributions.png # Performance histograms
    ├── hyperparameter_analysis.png   # Box plots by hyperparameter
    ├── correlation_heatmap.png       # Correlation matrix
    └── top_performers_analysis.png   # Scatter plots highlighting best
```

### Key Metrics Tracked

- **Final Training Loss**: Cross-entropy loss at end of training
- **Final Training Accuracy**: Token prediction accuracy
- **Evaluation Loss**: Loss on validation set
- **Evaluation Accuracy**: Accuracy on validation set
- **Evaluation Perplexity**: Model perplexity
- **Training Time**: Total time per experiment

## 🔍 Analysis Features

### Statistical Analysis
- Summary statistics (mean, std, min, max)
- Top/bottom performer identification
- Hyperparameter effect analysis
- Correlation analysis

### Visualizations
- Performance distribution histograms
- Hyperparameter vs performance box plots
- Correlation heatmaps
- Top performer scatter plots

### Recommendations
- Optimal hyperparameter identification
- Performance insights
- Hyperparameter sensitivity analysis

## 🛠️ Customization

### Modify Hyperparameter Ranges

Edit `configs/ablation_config.py`:

```python
lr_config = LearningRateConfig(
    muon_lr_values=[0.005, 0.01, 0.02],  # Custom Muon LR range
    adamw_lr_ratio_values=[0.05, 0.1],   # Custom AdamW ratio range
    momentum_values=[0.95],              # Single momentum value
    weight_decay_values=[0.1, 0.2]       # Custom weight decay range
)
```

### Adjust Model Configuration

Modify the base config in `create_default_ablation_config()`:

```python
base_config = MoEModelConfig(
    max_steps=100,        # More training steps
    eval_every=10,        # Less frequent evaluation
    batch_size=32         # Larger batch size
)
```

## 📋 Example Usage

### 1. Quick Test Run
```bash
# Run 4 experiments quickly
python run_ablation_study.py --quick
```

### 2. Comprehensive Study
```bash
# Run all 270 experiments
python run_ablation_study.py
```

### 3. Limited Study
```bash
# Run only 20 experiments
python run_ablation_study.py --max-experiments 20
```

### 4. Analyze Previous Results
```bash
# Analyze results from a previous run
python run_ablation_study.py --analyze-only --results-dir ablation_results/learning_rate_ablation_20241201_143022
```

## 🎯 Expected Results

The study will help identify:

1. **Optimal Learning Rate Ranges**: Which Muon and AdamW learning rates work best
2. **Hyperparameter Interactions**: How different parameters interact
3. **Performance Sensitivity**: Which parameters have the most impact
4. **Training Efficiency**: Balance between performance and training time

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config
2. **Long Training Time**: Use `--quick` flag or reduce `max_steps`
3. **Missing Dependencies**: Install required packages:
   ```bash
   pip install matplotlib seaborn plotly pandas
   ```

### Performance Tips

- Use `--quick` for initial testing
- Limit experiments with `--max-experiments` for faster results
- Monitor GPU memory usage during experiments

## 📊 Interpreting Results

### Key Metrics to Focus On

1. **Evaluation Accuracy**: Primary performance metric
2. **Evaluation Loss**: Secondary performance metric
3. **Training Time**: Efficiency consideration
4. **Hyperparameter Correlations**: Understanding parameter effects

### Best Practices

1. Start with quick study to identify promising ranges
2. Run full study on promising hyperparameter ranges
3. Focus on top 10% performers for detailed analysis
4. Consider training time vs performance trade-offs

## 🤝 Contributing

To extend the ablation study:

1. Add new hyperparameters to `LearningRateConfig`
2. Modify experiment combinations in `get_experiment_combinations()`
3. Update analysis in `AblationAnalyzer`
4. Add new visualizations as needed

## 📝 Notes

- Each experiment uses the same random seed for reproducibility
- Results are automatically saved and can be analyzed later
- The study uses a reduced model configuration for faster execution
- All experiments use the same dataset split for fair comparison
