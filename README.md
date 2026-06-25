# 🏎️ F1 Pit Stop Strategy Predictor (Kaggle Playground S6E5)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

An advanced, production-grade deep learning solution optimized for tabular regression, developed specifically for the Kaggle Playground Series (Season 6, Episode 5) competition to predict Formula 1 pit stop behaviors (`PitNextLap`). 

This repository implements a highly sophisticated **Deep Tabular Residual Network (TabResNet)** incorporating modern architectural enhancements (Gated Linear Units, Squeeze-and-Excitation attention, and Stochastic Depth), paired with an automated robust feature engineering pipeline and a Non-Negative Ridge blending ensemble layer.

---

## 🌟 Key Features

*   **Advanced Tabular Architecture**: Custom PyTorch neural network combining Pre-Activation Residual Blocks (`PreActResBlock`), Gated Linear Units (`GLU`), Squeeze-and-Excitation (`SE`) 1D attention, and DropPath (Stochastic Depth).
*   **Robust Feature Engineering Pipeline**: Fully automated `FeatureProcessor` featuring quantile-based outlier clipping, `RobustScaler` normalization, and Non-linear `PowerTransformer` (Yeo-Johnson) mappings to maximize deep learning stability on tabular inputs.
*   **Competition-Grade Training Workflow**: Stratified $K$-Fold cross-validation framework engineered with `CosineAnnealingWarmRestarts` learning rate scheduling, gradient norm clipping, and rigid out-of-fold (`OOF`) validation tracking.
*   **Non-Negative Ridge Blender**: A custom specialized meta-ensembler constraint-optimized to blend Out-Of-Fold predictions from the neural network with external gradient-boosted trees (e.g., LightGBM, XGBoost, CatBoost) without zero-weight convergence issues.
*   **Adaptive Environment Awareness**: Auto-detects local execution directory maps versus Kaggle container paths for zero-config notebook/script execution.

---

## 🏗️ Model Architecture

Unlike naive multi-layer perceptrons (MLPs), this repository uses an optimized topology specifically tailored to continuous and encoded discrete tabular structures:

```
[Input Features] ──> [BatchNorm1d] ──> [Linear + SiLU] ──> [Dropout (0.075)]
                                 │
                                 ▼
                     [Gated Linear Unit (GLU)]
                                 │
                                 ▼
               [PreActResBlock × 4 w/ DropPath (0.00 -> 0.05)]
               ├── BatchNorm1d -> SiLU -> Linear -> Dropout
               ├── BatchNorm1d -> SiLU -> Linear
               └── Squeeze-and-Excitation (1D Channel Attention)
                                 │
                                 ▼
[Regression Head] ──> [BatchNorm1d -> Linear -> SiLU -> Dropout -> Linear(1)]
```

### Mathematical Modules Implemented
1. **Gated Linear Unit (GLU)**: Controls information flow using a sigmoid gating mechanism:
   $$	ext{GLU}(x) = (xW_1) \otimes \sigma(xW_2)$$
2. **Squeeze-and-Excitation (SE)**: Adapts 1D feature channel excitation weights dynamically by scaling global features:
   $$	ext{SE}(x) = x \cdot \sigma(W_2 \cdot 	ext{SiLU}(W_1 \cdot x))$$
3. **DropPath (Stochastic Depth)**: Linearly scales block-skipping probabilities across deep layers to regularize residual feature propagation.

---

## 📊 Feature Preprocessing Engine

Tabular deep learning models are highly sensitive to unnormalized distributions and extreme outliers. The automated `FeatureProcessor` addresses this via a multi-stage `scikit-learn` pipeline:

| Target Block | Operational Layer | Functional Objective |
| :--- | :--- | :--- |
| **Outlier Handling** | Quantile Clipper (`0.995`) | Caps extreme outliers at the 0.5th and 99.5th percentiles to avoid gradient explosions. |
| **Numerical Pathway** | `SimpleImputer(median)` | Imputes missing telemetry metrics robustly against heavily skewed distributions. |
| | `RobustScaler` | Centers and scales features using interquartile range (IQR) to mitigate residual outlier skew. |
| | `PowerTransformer` | Applies the *Yeo-Johnson* transformation to minimize heteroscedasticity and force normal feature shapes. |
| **Categorical Pathway**| `SimpleImputer(most_frequent)`| Replaces rare or missing categorical markers safely. |
| | `OneHotEncoder` | Converts multi-class strings to discrete sparse markers, filtering rare categories (`min_frequency=10`). |

---

## ⚙️ Hyperparameter Configuration

Configurations are completely decoupled using clean Python `@dataclass` objects for modular hyperparameter tuning:

```python
# Model Hidden Topologies
ModelConfig(
    hidden_dim=160,
    num_blocks=4,
    dropout=0.15,
    drop_path_rate=0.05,
    use_se=True,
    se_reduction=8,
    use_glu=True,
    glu_dim=64
)

# Training Optimization Loop
TrainConfig(
    epochs=30,
    batch_size=2048,
    learning_rate=2e-3,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    scheduler_t0=10,
    scheduler_t_mult=2,
    early_stop_patience=8,
    n_folds=5,
    random_seed=3407
)
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with PyTorch (CUDA supported recommended) and required analytics stacks.
```bash
pip install torch numpy pandas scikit-learn
```

### 2. Dataset Setup
Place your competition data files (`train.csv`, `test.csv`) into either your local root path, an `input/` folder, or a `data/` directory. The pipeline will automatically map and execute data loading paths.

```
.
├── main.py
├── README.md
└── data/
    ├── train.csv
    └── test.csv
```

### 3. Execution
Run the end-to-end processing, $K$-fold training cross-validation, and submission blending execution loop:
```bash
python main.py
```

---

## 📈 Ensemble & Ridge Blending Workflow

To extract top-tier leaderboard performance, the script provides structural scaffolding to load external predictions (e.g., high-performing GBDTs models saved inside `/blend/submission.csv`) and maps them using a custom **Non-Negative Ridge Blender**.

This module fits an $L_2$ regularized meta-learner over your multi-model predictions under the constraint that all model weights remain non-negative ($\mathbf{w} \geq 0$). It prevents destructive interference during testing and outputs a balanced, highly generalized `submission_optimized.csv` ready for Kaggle submission portals.

---

## 📄 License
This project is open-source software licensed under the [MIT License](LICENSE).
