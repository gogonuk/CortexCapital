## Project Journey Log

### 2025-09-27 - Integration of Market Regime Detection & Git Conventions

A significant step was taken today to enhance the project's predictive capabilities and establish robust development practices.

**Key Advancements:**

*   **Market Regime Detection:** Implemented a new data mining step (`gogo_test/mining.py`) using K-Means clustering to identify distinct market regimes. This feature is now integrated into the data pipeline, and the `train.py` and `predict.py` scripts have been updated to utilize these regime features (one-hot encoded) for improved model performance.
*   **Enhanced Pipeline:** The `Makefile` has been updated to include a `mining` step, ensuring the market regime features are generated automatically as part of the `make train` command.
*   **Git Conventions Established:** A comprehensive `CONTRIBUTING.md` file was created to document:
    *   A clear branching strategy (`type/short-description`).
    *   A Conventional Commits message format (`type(scope): short description`).
    *   A detailed list of commit scopes for various data analysis domains.
    *   A Git Tag convention (`workflow/{domain}-{technique}/v{version}`) for preserving distinct workflow snapshots.
*   **Workflow Snapshots:** The project's history now includes two key workflow tags:
    *   `workflow/ta-classic/v1.0`: Represents the original, classic technical analysis pipeline.
    *   `workflow/ml-regime-clustering/v1.0`: Represents the new, enhanced machine learning pipeline with market regime detection.

This work significantly advances the project's analytical depth and ensures a structured, auditable development process.


This document tracks the experiments and data mining studies performed on this project.

---

### 2025-09-27: Experiment 01 - Market Regime Clustering

**Notebook:** `notebooks/01_market_regime_clustering.ipynb`

**Objective:** To determine if the market exhibits distinct regimes that can be identified through unsupervised clustering. The hypothesis is that the model's performance can be improved by tailoring strategies to different market conditions.

**Methodology:**

1.  Selected key features for clustering: `RSI`, `MACD`, `BBM`, `daily_return`, and `volatility`.
2.  Used the Elbow Method to find the optimal number of clusters, which suggested k=3 or k=4.
3.  Applied K-Means clustering to group the data into distinct market regimes.
4.  Visualized the regimes on a price chart to analyze their characteristics.

**Outcome & Next Steps:**

The analysis successfully identified distinct market regimes (e.g., high-volatility, trending, etc.). The next step is to incorporate the identified `regime` as a new feature into the machine learning model in `gogo_test/modeling/train.py` to see if it improves predictive performance.
