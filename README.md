
# 🧠 Analysis of Muon Optimizer in LLMs

![Status](https://img.shields.io/badge/Status-Active-success?logo=github&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Experiments](https://img.shields.io/badge/Experiments-6-orange)
![Last Updated](https://img.shields.io/badge/Updated-Oct%202025-lightgrey)

---

## 📘 Overview

This repository contains a comprehensive analysis of the **Muon optimizer** for training **Large Language Models (LLMs)** with a **Mixture-of-Experts (MoE)** architecture.

The **Muon optimizer** combines **momentum** with **Newton–Schulz orthogonalization** to improve training efficiency and convergence, with potential for enhanced **perplexity (PPL)** through extended training—particularly for **Multi-Head Latent Attention (MHLA)**.

---

## 🧩 How Muon Works

The **Muon optimizer** applies **orthogonalization** to the momentum buffer using **Newton–Schulz iteration**, reducing redundancy in weight matrices and improving convergence in transformer-based LLMs.

This repository includes:

- Comparisons between **Muon** and **AdamW**  
- **Ablation studies** of momentum and NS components  
- **Hyperparameter sensitivity** analysis  
- Evaluation across **activation functions** and **attention mechanisms**

---

## 📂 Repository Structure

```

llm.py                        # Original MoE LLM implementation with Muon optimizer
paper_activationFN_experiments.tex   # LaTeX source for the paper
paper_activationFN_experiments.pdf   # Compiled paper (analysis of activations & attention)
experiment_*.py               # Individual experiment implementations (1–6)
EXPERIMENT_5_RESULTS.md       # Attention mechanism experiment results
requirements.txt              # Dependencies

````

---

## 🧪 Experiments

### 🔹 **Experiment 1: Baseline Comparison**  
**File:** `experiment_1_baseline_comparison.py`  
**Objective:** Compare hybrid Muon (for 2D layers) vs. AdamW (for embeddings/norms)

#### What it Does
- Trains identical MoE models using both optimizers  
- Measures **validation loss**, **accuracy**, **perplexity**, and **training time**  
- Saves comparative analysis results  

#### Key Metrics
| Metric | Muon | AdamW | Δ Improvement |
|--------|------|-------|----------------|
| Validation Loss | **0.0476** | 0.0547 | **-13.2%** |
| Accuracy | **+0.26%** | — | — |
| Perplexity | **1.05** | 1.06 | -0.94% |
| Training Time | 13.3 min | 11.8 min | +12.7% |

---

### 🔹 **Experiment 2: Ablation Study**  
**File:** `experiment_2_ablation_muon.py`  
**Objective:** Isolate Muon’s **momentum** and **Newton–Schulz (NS)** components.

#### What it Does
- Tests four variants:
  1. Full Muon (momentum + NS)  
  2. Momentum only  
  3. NS only  
  4. SGD-like baseline  

#### Key Insights
- Full Muon achieves **lowest loss (2.5347)**  
- Removing momentum causes **114.4% loss increase**  
- Synergy between momentum & NS is **critical** for performance

---

### 🔹 **Experiment 3: Hyperparameter Sensitivity**  
**File:** `experiment_3_hyperparameter_sensitivity.py`  
**Objective:** Evaluate Muon’s robustness to parameter changes.

#### Parameters Tested
- Learning rates: `[0.001, 0.005, 0.01, 0.02, 0.05]`  
- Momentum: `[0.9, 0.95, 0.99]`  
- NS steps: `[3, 5, 7]`

#### Key Findings
| Hyperparameter | Optimal | Notes |
|----------------|----------|--------|
| Learning Rate | **0.05 (loss: 0.3277)** | High sensitivity (18.6× improvement) |
| Momentum | **0.95** | Moderate effect |
| NS Steps | **7** | Weak sensitivity |

---

### 🔹 **Experiment 4: Activation Functions & Attention Mechanisms**  
**File:** `experiment_6_activate_fn.py`  
**Objective:** Compare **SiLU**, **GELU**, **ReLU**, **Tanh** activations and **MHSA/MHLA** attention.

#### What it Does
- Trains MoE models for **500 steps**, batch size 16, on **SmolLM** (500k tokens)
- Measures **loss**, **accuracy**, **PPL**
- Analyzes **MHLA scalability**

#### Key Metrics
| Configuration | Loss | PPL | Accuracy |
|----------------|------|------|-----------|
| **MHSA-ReLU** | **4.6883** | **108.67** | **0.2560** |
| MHLA-ReLU | 4.7078 | 110.80 | — |
| MHLA-Tanh | ~4.85 | ~130 | (incomplete) |

> 💡 *MHLA’s linear complexity (O(nk)) suggests improved PPL with longer training (2000+ steps, projecting ~5–10% loss reduction).*

---

## ⚙️ Usage

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
````

### Running Original Implementation

```bash
python llm.py
```

### Compiling the Paper

```bash
pdflatex paper_activationFN_experiments.tex
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## 📈 Expected Results

Each experiment generates:

* Detailed **console metrics**
* **JSON** results (`experiment_*_results.json`)
* **Comparative tables** in the paper PDF (`paper_activationFN_experiments.pdf`)

> **Key Insight:**
> MHLA’s linear complexity and latent maturation suggest improved PPL with longer training (e.g. 2000 steps → targeting ~100 PPL for MHLA-ReLU).

---

## 🔍 Key Research Questions

| Research Question                 | Finding                                                   |
| --------------------------------- | --------------------------------------------------------- |
| **Performance Comparison**        | Muon outperforms AdamW by **13.2%** in loss               |
| **Component Analysis**            | Momentum drives **114.4%** loss reduction                 |
| **Hyperparameter Robustness**     | High sensitivity to learning rate                         |
| **Activation & Attention Impact** | **MHSA-ReLU** optimal; **MHLA** promising for longer runs |

---

## 🧠 Model Architecture

* **Architecture:** Transformer + MoE feed-forward layers
* **Model Size:** 384d, 6 layers, 8 heads, 1536 FF dimension
* **MoE:** 8 experts, top-2 routing
* **MHLA:** 64 latent tokens
* **Dataset:** SmolLM corpus (500,000 tokens)
* **Training:** 1000 steps (Exp 1–3), 500 steps (Exp 4), cosine LR schedule

---

## 🤝 Contributing

This is a **research repository**.
For questions or collaborations, join our **Research Discord**:
👉 [https://discord.gg/6AbXGpKTwN](https://discord.gg/6AbXGpKTwN)




