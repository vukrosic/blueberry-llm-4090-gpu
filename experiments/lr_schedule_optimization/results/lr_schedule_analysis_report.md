# Learning Rate Schedule Optimization Report

Generated on: 2025-09-25 08:42:45

## Objective

Optimize learning rate scheduling to achieve additional validation loss improvements beyond the 6.5% gain from the ablation study.

## Summary

Total experiments: 8
Successful experiments: 8

## Top 5 Performing LR Schedules

| Rank | Experiment | Val Loss | Val Accuracy | Schedule Type | Improvement |
|------|------------|----------|--------------|---------------|-------------|
| 1 | cosine_warmup_05 | 6.7518 | 0.1512 | cosine | +1.6% |
| 2 | cosine_minlr_01 | 6.8310 | 0.1426 | cosine | +0.5% |
| 3 | cosine_minlr_02 | 6.8368 | 0.1409 | cosine | +0.4% |
| 4 | linear_decay | 6.8370 | 0.1452 | linear | +0.4% |
| 5 | cosine_minlr_10 | 6.8526 | 0.1332 | cosine | +0.1% |

## Best Configuration

**Experiment**: cosine_warmup_05
**Description**: Shorter warmup (5%) with cosine annealing
**Validation Loss**: 6.7518
**Validation Accuracy**: 0.1512
**Improvement**: 1.6%

## Recommendations

Based on the learning rate schedule optimization:

1. **Optimal Schedule**: Use the top-performing schedule configuration
2. **Key Insights**: Analyze which schedule parameters contribute most to improvement
3. **Implementation**: Apply the best schedule to the main branch
