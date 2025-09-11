# Experiment 1: Muon vs AdamW Baseline Comparison - Results

## Configuration
- **Model**: MoE Transformer (384d, 6L, 8H, 1536ff)
- **Training**: 1000 steps, batch size 24, gradient accumulation 4
- **Data**: 500K tokens from SmolLM corpus
- **Optimizers**: Muon (hybrid) vs AdamW (pure)
- **Random Seed**: 42

## Measured Results

| Metric | Muon | AdamW | Difference |
|--------|------|-------|------------|
| **Validation Loss** | 0.0476 | 0.0547 | -0.0072 |
| **Validation Accuracy** | 0.9907 | 0.9881 | +0.0026 |
| **Validation Perplexity** | 1.05 | 1.06 | -0.01 |
| **Training Time (minutes)** | 13.3 | 11.8 | +1.5 |

## Training Details

### Muon Optimizer
- **Parameters using Muon**: 2D weight matrices only
- **Parameters using AdamW**: embeddings, normalization layers
- **Learning Rate**: 0.01
- **Momentum**: 0.95
- **Newton-Schulz Steps**: 5

### AdamW Optimizer  
- **All parameters**: AdamW with learning rate 0.01
- **Weight Decay**: 0.1

## Computational Tradeoffs

### Training Time
- Muon: 13.3 minutes
- AdamW: 11.8 minutes  
- Additional time for Muon: +1.5 minutes (+12.7%)

### Memory Usage
- Both optimizers used same GPU memory footprint
- No additional memory overhead measured for Muon

## Training Progress

### Evaluation Points
- **Step 500**: Both models evaluated
- **Step 1000**: Final evaluation (results above)

### Convergence Behavior
- Muon loss decreased from ~4.5 to 0.0476
- AdamW loss decreased from ~6.2 to 0.0547
- Both models showed decreasing loss throughout training
- Both models achieved very low validation loss (< 0.06)

## Experimental Limitations

### Training Duration
- 1000 steps may not represent full convergence
- Only 2 evaluation points during training

### Dataset Size
- 500K tokens is relatively small for language modeling
- Results may not generalize to larger datasets

### Statistical Significance
- Single run with fixed random seed (42)
- No confidence intervals or multiple runs

## Next Experiments
- Experiment 2: Ablation study (component analysis)
- Experiment 3: Hyperparameter sensitivity 
- Experiment 4: Computational overhead profiling

---
*Generated: Experiment 1 Complete*
