# Analysis of Muon Optimizer in LLMs

This repository contains a comprehensive analysis of the Muon optimizer for training Large Language Models (LLMs) with Mixture-of-Experts (MoE) architecture. The Muon optimizer combines momentum with Newton-Schulz orthogonalization to improve training efficiency and convergence.

## Overview

The Muon optimizer is a novel optimization algorithm that applies orthogonalization to the momentum buffer using the Newton-Schulz iteration. This approach aims to reduce redundancy in weight matrices and improve convergence speed in transformer-based language models.

## Repository Structure

- `llm.py` - Original implementation of the MoE LLM with Muon optimizer
- `RESEARCH_PLAN.md` - Detailed research plan and methodology
- `experiment_*.py` - Individual experiment implementations
- `requirements.txt` - Python dependencies

## Experiments

This repository contains four comprehensive experiments analyzing the Muon optimizer:

### Experiment 1: Baseline Comparison (`experiment_1_baseline_comparison.py`)

**Objective:** Compare Muon (hybrid) optimizer vs pure AdamW optimizer

**What it does:**
- Trains identical MoE models with both optimizers
- Uses hybrid approach: Muon for 2D weight matrices, AdamW for embeddings/norms
- Measures validation loss, accuracy, perplexity, and training time
- Generates comparative analysis and saves results

**Key Metrics:**
- Validation Loss, Accuracy, Perplexity
- Training time comparison
- Performance difference analysis

### Experiment 2: Ablation Study (`experiment_2_ablation_muon.py`)

**Objective:** Isolate the impact of Muon's components through ablation

**What it does:**
- Tests 4 variants of Muon optimizer:
  - Full Muon (momentum + Newton-Schulz)
  - Muon w/o Newton-Schulz (momentum only)
  - Muon w/o momentum (Newton-Schulz only)
  - Muon w/o both (basic SGD-like)
- Quantifies individual component contributions
- Analyzes synergy effects between components

**Key Insights:**
- Component contribution analysis
- Synergy effect measurement
- Performance impact of each component

### Experiment 3: Hyperparameter Sensitivity (`experiment_3_hyperparameter_sensitivity.py`)

**Objective:** Evaluate Muon's robustness to hyperparameter changes

**What it does:**
- Sweeps learning rates: [0.001, 0.005, 0.01, 0.02, 0.05]
- Sweeps momentum values: [0.9, 0.95, 0.99]
- Sweeps Newton-Schulz steps: [3, 5, 7]
- Identifies optimal hyperparameter combinations
- Analyzes sensitivity patterns

**Key Findings:**
- Optimal hyperparameter identification
- Sensitivity analysis for each parameter
- Robustness assessment

### Experiment 4: Computational Overhead (`experiment_4_profiling_overhead.py`)

**Objective:** Quantify the computational cost of Newton-Schulz orthogonalization

**What it does:**
- Uses PyTorch profiler to measure execution time
- Profiles Newton-Schulz function calls specifically
- Analyzes memory usage and performance impact
- Provides cost-benefit analysis

**Key Metrics:**
- Newton-Schulz overhead as % of optimizer time
- Memory usage analysis
- Performance impact assessment
- Cost-benefit recommendations

## Usage

### Running Individual Experiments

```bash
# Experiment 1: Baseline Comparison
python experiment_1_baseline_comparison.py

# Experiment 2: Ablation Study  
python experiment_2_ablation_muon.py

# Experiment 3: Hyperparameter Sensitivity
python experiment_3_hyperparameter_sensitivity.py

# Experiment 4: Computational Overhead
python experiment_4_profiling_overhead.py
```

### Running Original Implementation

```bash
python llm.py
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Results

Each experiment generates:
- Detailed console output with metrics
- JSON results files (`experiment_*_results.json`)
- Profiling traces (Experiment 4)
- Comparative analysis tables

## Key Research Questions Addressed

1. **Performance Comparison:** How does Muon compare to standard AdamW?
2. **Component Analysis:** What's the individual contribution of momentum vs Newton-Schulz?
3. **Hyperparameter Robustness:** How sensitive is Muon to its key parameters?
4. **Computational Cost:** What's the overhead of the orthogonalization step?

## Research Plan

See `RESEARCH_PLAN.md` for the complete research methodology, timeline, and expected outcomes.

## Model Architecture

- **Architecture:** Transformer with MoE feed-forward layers
- **Model Size:** 384d, 6 layers, 8 heads, 1536 FF dimension
- **MoE:** 8 experts, top-2 routing
- **Dataset:** SmolLM corpus (500K tokens)
- **Training:** 1000 steps with cosine learning rate schedule

## Contributing

This is a research repository. For questions or contributions, please refer to the research plan and experiment documentation. 