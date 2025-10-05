# Analysis of Muon Optimizer in LLMs

This repository contains a comprehensive analysis of the Muon optimizer for training Large Language Models (LLMs) with a Mixture-of-Experts (MoE) architecture. The Muon optimizer combines momentum with Newton-Schulz orthogonalization to improve training efficiency and convergence, with potential for enhanced perplexity (PPL) through extended training, particularly for Multi-Head Latent Attention (MHLA).

## Overview

The Muon optimizer is a novel optimization algorithm that applies orthogonalization to the momentum buffer using Newton-Schulz iteration, reducing redundancy in weight matrices and improving convergence in transformer-based LLMs. This repository includes experiments comparing Muon to AdamW, analyzing its components, evaluating hyperparameter sensitivity, and testing activation functions and attention mechanisms.

## Repository Structure

- `llm.py`: Original implementation of the MoE LLM with Muon optimizer.
- `RESEARCH_PLAN.md`: Detailed research plan and methodology.
- `paper_activationFN_experiments.tex`: LaTeX source for the paper analyzing Muon with activation functions and attention mechanisms.
- `paper_activationFN_experiments.pdf`: Compiled PDF of the paper.
- `experiment_*.py`: Individual experiment implementations (1–6).
- `EXPERIMENT_5_RESULTS.md`: Results and analysis for attention mechanism experiments.
- `requirements.txt`: Python dependencies.

## Experiments

This repository contains four comprehensive experiments analyzing the Muon optimizer:

### Experiment 1: Baseline Comparison (`experiment_1_baseline_comparison.py`)

**Objective**: Compare hybrid Muon optimizer (Muon for 2D layers, AdamW for embeddings/norms) vs. pure AdamW.

**What it does**:
- Trains identical MoE models with both optimizers.
- Measures validation loss, accuracy, perplexity, and training time.
- Generates comparative analysis and saves results.

**Key Metrics**:
- Validation Loss: Muon (0.0476) vs. AdamW (0.0547, -13.2% better).
- Validation Accuracy: +0.26% for Muon.
- Perplexity: Muon (1.05) vs. AdamW (1.06, -0.94%).
- Training Time: Muon (13.3 min) vs. AdamW (11.8 min, +12.7%).

### Experiment 2: Ablation Study (`experiment_2_ablation_muon.py`)

**Objective**: Isolate the impact of Muon’s momentum and Newton-Schulz (NS) components.

**What it does**:
- Tests four variants: Full Muon (momentum + NS), momentum only, NS only, basic SGD-like.
- Quantifies individual component contributions and synergy effects.

**Key Insights**:
- Full Muon achieves lowest loss (2.5347), with momentum contributing most (114.4% loss increase without it).
- Synergy between momentum and NS critical for performance.

### Experiment 3: Hyperparameter Sensitivity (`experiment_3_hyperparameter_sensitivity.py`)

**Objective**: Evaluate Muon’s robustness to hyperparameter changes.

**What it does**:
- Sweeps learning rates ([0.001, 0.005, 0.01, 0.02, 0.05]), momentum ([0.9, 0.95, 0.99]), and NS steps ([3, 5, 7]).
- Identifies optimal hyperparameters and analyzes sensitivity.

**Key Findings**:
- Optimal: learning rate (0.05, 0.3277 loss), momentum (0.95), NS steps (7).
- High sensitivity to learning rate (18.6x loss improvement), moderate to momentum, weak to NS steps.

### Experiment 4: Activation Functions and Attention Mechanisms (`experiment_6_activate_fn.py`)

**Objective**: Evaluate Muon across activation functions (SiLU, GELU, ReLU, Tanh) and attention mechanisms (MHSA, MHLA).

**What it does**:
- Trains MoE models over 500 steps, batch size 16, on a 500,000-token SmolLM subset.
- Compares MHSA and MHLA with four activations, measuring loss, accuracy, and PPL.
- Analyzes MHLA’s scalability for long-term training.

**Key Metrics**:
- MHSA-ReLU: Best performance (loss: 4.6883, PPL: 108.67, accuracy: 0.2560).
- MHLA-ReLU: Close second (loss: 4.7078, PPL: 110.80).
- MHLA-Tanh: Estimated loss (4.85), PPL (130) due to incomplete evaluation.
- **Note**: MHLA’s linear complexity (\(O(nk)\)) suggests better PPL with longer training (e.g., 2000+ steps, projecting 5–10% loss reduction).

## Usage

### Running Individual Experiments

```bash
# Experiment 1: Baseline Comparison
python experiment_1_baseline_comparison.py

# Experiment 2: Ablation Study
python experiment_2_ablation_muon.py

# Experiment 3: Hyperparameter Sensitivity
python experiment_3_hyperparameter_sensitivity.py

# Experiment 4: Activation and Attention
python experiment_6_activate_fn.py --config configs/exp4.yaml

# Experiment 5: Attention Head Type
python experiment_5_attention_head_type.py
```

### Running Original Implementation
```python llm.py```

### Requirements
Install dependencies:
```pip install -r requirements.txt```

### Expected Results
Each experiment generates:

- Console output with detailed metrics.
- JSON results files (experiment_*_results.json).
- Comparative analysis tables in paper_activationFN_experiments.pdf.

Key Insight: MHLA’s linear complexity and latent maturation suggest improved PPL with longer training (e.g., 2000 steps, targeting ~100 PPL for MHLA-ReLU).
Key Research Questions Addressed

- Performance Comparison: How does Muon compare to AdamW? (Muon outperforms by 13.2% in loss.)
- Component Analysis: What’s the contribution of momentum vs. NS? (Momentum drives 114.4% loss reduction.)
- Hyperparameter Robustness: How sensitive is Muon to parameters? (High sensitivity to learning rate.)
- Activation and Attention Impact: How do activation functions and attention types affect performance? (MHSA-ReLU optimal, MHLA promising for longer runs.)

### Model Architecture

- Architecture: Transformer with MoE feed-forward layers.
- Model Size: 384d, 6 layers, 8 heads, 1536 FF dimension.
- MoE: 8 experts, top-2 routing.
- MHLA: 64 latent tokens.
- Dataset: SmolLM corpus (500,000 tokens).
- Training: 1000 steps (Experiments 1–3), 500 steps (Experiment 4), cosine learning rate schedule.

Contributing
This is a research repository. For questions or contributions, join our Research Discord: .
