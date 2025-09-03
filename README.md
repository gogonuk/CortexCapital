# CortexCapital
*Evolving Financial Intelligence. Quantitative Analysis for Modern Investing.*

[![CCDS - Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

This repository provides a robust and reproducible framework for financial time series analysis, built upon the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template. It demonstrates an end-to-end pipeline for predicting stock prices, incorporating sophisticated feature engineering, walk-forward validation for realistic model evaluation, and calculation of key financial metrics.

## Key Features

*   **Automated Data Acquisition:** Easily download historical stock data (e.g., AAPL) using `yfinance`.
*   **Sophisticated Feature Engineering:** Generate a rich set of features including Simple Moving Averages (SMA), Relative Strength Index (RSI), MACD, Bollinger Bands, and lagged price/volume data.
*   **Walk-Forward Validation:** A robust methodology for evaluating time series models, simulating real-world trading scenarios.
*   **Financial Metrics Calculation:** Assess strategy performance using Total Return, Sharpe Ratio, and Maximum Drawdown, accounting for prediction thresholds and transaction costs.
*   **Modular and Reproducible:** A clear project structure and `Makefile` commands ensure reproducibility and ease of collaboration.
*   **Model Agnostic:** Easily swap out predictive models (currently uses Random Forest Regressor).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands for the entire pipeline
├── README.md          <- The top-level README for developers and collaborators.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling (e.g., with features).
│   └── raw            <- The original, immutable data dump (e.g., downloaded stock data).
│
├── docs               <- Project documentation (using mkdocs).
│
├── models             <- Trained and serialized models (e.g., random_forest_model.pkl).
│
├── notebooks          <- Jupyter notebooks for exploration and analysis.
│
├── pyproject.toml     <- Project configuration for `gogo_test` package and tools like `ruff`.
│
├── references         <- Data dictionaries, manuals, and other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures (e.g., price vs. SMA plots).
│
├── requirements.txt   <- Python dependencies for reproducing the analysis environment.
│
└── gogo_test          <- Source code for the core financial analysis modules.
    │
    ├── __init__.py             <- Makes gogo_test a Python module
    ├── config.py               <- Stores useful variables and configuration (e.g., paths).
    ├── dataset.py              <- Scripts to download and prepare raw data.
    ├── features.py             <- Code to create sophisticated features for modeling.
    ├── plots.py                <- Code to create visualizations of data and features.
    └── modeling                
        ├── __init__.py 
        ├── predict.py          <- Code to run model inference with trained models.          
        └── train.py            <- Code to train models and perform walk-forward validation.
```

## Getting Started

To set up the environment and run the full pipeline:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    make requirements
    ```
4.  **Run the data pipeline:**
    ```bash
    make data
    make features
    make plot
    ```
5.  **Train the model with walk-forward validation and evaluate financial metrics:**
    ```bash
    # This command will take a significant amount of time due to hyperparameter tuning.
    # You can adjust prediction_threshold and transaction_cost as needed.
    .venv/bin/python gogo_test/modeling/train.py --prediction-threshold 0.005 --transaction-cost 0.0005
    ```
6.  **Make a prediction:**
    ```bash
    make predict
    ```

## How to Contribute

We welcome contributions to this project! If you have ideas for new features, models, or improvements to the existing pipeline, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure tests pass.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

[Specify your license here, e.g., MIT License]

---