# Contributing Guidelines

This document outlines the conventions for branching, commit messages, and code style to ensure the project remains clean, consistent, and easy to navigate.

## Branching Strategy

All work should be done in a dedicated branch, not directly on `main`. Branch names should follow the `type/short-description` format.

### Branch Types

*   **`feature/`**: For adding new features or functionality.
*   **`fix/`**: For fixing a bug.
*   **`refactor/`**: For improving code structure without changing functionality.
*   **`docs/`**: For adding or updating documentation.
*   **`test/`**: For adding or improving tests.
*   **`chore/`**: For routine maintenance, like updating dependencies.

**Example:** `feature/market-regime-integration`

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. The format is:

**`type(scope): short description`**

### Commit Types

The `type` should be one of the following: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

### Commit Scopes

The `scope` provides high-level context. Please choose the most relevant scope from the list below.

---

### 1. Statistical & Econometric Analysis
| Area | Scope |
|---|---|
| Descriptive Time Series Analysis | `stat-descriptive` |
| Stationarity Testing | `stat-stationarity` |
| Autocorrelation Analysis | `stat-autocorr` |
| ARIMA/SARIMA Models | `stat-arima` |
| GARCH Models | `stat-garch` |
| Cointegration Analysis | `stat-coint` |
| Vector Autoregression (VAR) | `stat-var` |

### 2. Quantitative Finance & Factor Investing
| Area | Scope |
|---|---|
| Factor Model Analysis (Fama-French) | `quant-factor-model`|
| Portfolio Optimization (Markowitz) | `quant-portfolio-opt`|
| Risk Modeling (VaR, Expected Shortfall)| `quant-risk-model` |
| Performance Attribution | `quant-performance` |

### 3. Technical Analysis
| Area | Scope |
|---|---|
| Trend Indicators (Moving Averages) | `ta-trend` |
| Momentum Indicators (RSI) | `ta-momentum` |
| Volatility Indicators (Bollinger Bands) | `ta-volatility` |
| Volume Indicators (On-Balance Volume) | `ta-volume` |
| Chart Pattern Recognition | `ta-pattern` |

### 4. Machine Learning & Data Mining
| Area | Scope |
|---|---|
| Regression | `ml-regression` |
| Classification | `ml-classification` |
| Clustering (K-Means) | `ml-clustering` |
| Dimensionality Reduction (PCA) | `ml-dim-reduction` |
| Reinforcement Learning (RL) | `ml-rl` |
| Deep Learning (RNNs, LSTMs) | `ml-dl-core` |

### 5. Alternative Data Analysis
| Area | Scope |
|---|---|
| Sentiment Analysis (NLP) | `alt-nlp` |
| Network Analysis | `alt-network` |
| High-Frequency Data Analysis | `alt-hft` |
| Satellite & Geospatial Data | `alt-geo` |
| Advanced Text Analysis | `alt-text-advanced` |
| Biometric & Behavioral Data | `alt-biometric` |

### 6. Causal & Event-Driven Analysis
| Area | Scope |
|---|---|
| Event Studies | `causal-event-study`|
| Causal Inference Methods | `causal-inference` |
| Causal Machine Learning | `causal-ml` |

### 7. Cutting-Edge Techniques
| Area | Scope |
|---|---|
| Time Series Transformers | `dl-transformers` |
| GANs / Diffusion Models | `dl-generative` |
| Graph Neural Networks (GNNs) | `dl-gnn` |
| Advanced Reinforcement Learning | `rl-advanced` |
| Model Interpretability (XAI, SHAP) | `xai` |
| Market Microstructure / DeFi | `micro-analysis` |
| RegTech / ESG Analytics | `regtech` |
| Privacy (Federated Learning, DP) | `privacy` |
| Quantum Finance | `quantum` |
| Multi-Modal Data Fusion | `multimodal` |
| Meta-Learning & Adaptation | `meta-learning` |

### 8. Operational
| Area | Scope |
|---|---|
| General Operations (Makefile, CI/CD) | `ops` |
| Data Pipeline & ETL | `data` |

---
**Example Commit:** `feat(ml-clustering): Integrate market regime detection`

## Git Tag Convention

To preserve distinct, recallable workflows, we use Git tags. The convention is:

**`workflow/{domain}-{technique}/v{version}`**

*   **`workflow/`**: A namespace for all workflow tags.
*   **`{domain}-{technique}`**: Describes the workflow, using the scopes defined above (e.g., `ta-classic`, `ml-regime-clustering`).
*   **`/v{version}`**: A version number for the workflow (e.g., `v1.0`).

**Example Tags:**
*   `workflow/ta-classic/v1.0`
*   `workflow/ml-regime-clustering/v1.0`
