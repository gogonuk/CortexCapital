# Data Mining Guide

This document provides a step-by-step guide to performing data mining tasks with this project.

## 1. Data Acquisition

The first step is to acquire the data. This can be done by running the `get_sol_data.py` script.

```bash
python gogo_test/dataset/get_sol_data.py
```

This will download the data and store it in the `data/raw` directory.

## 2. Data Preparation

Once the data is acquired, it needs to be prepared for training. The `dataset.py` script handles the data preparation.

## 3. Feature Engineering

The next step is to create features for the model. This is done by the `features.py` script, which will generate new features from the raw data.

## 4. Market Regime Generation

After feature engineering, market regimes are identified using clustering. The `mining.py` script performs this step.

```bash
python gogo_test/mining.py
```

This will add a 'regime' column to the processed data.

## 5. Model Training

With the data prepared, features created, and regimes identified, we can now train the model. The `train.py` script is used for this purpose.

```bash
python gogo_test/modeling/train.py
```

This will train the model and save it in the `models` directory.

## 6. Prediction

After the model is trained, we can use it to make predictions on new data. The `predict.py` script is used for this.

```bash
python gogo_test/modeling/predict.py
```

## 7. Evaluation

Finally, we can evaluate the performance of the model. The `profit_curve.py` script can be used to generate a profit curve plot.

```bash
python gogo_test/plots/profit_curve.py
```

This will generate a plot and save it in the `reports/figures` directory.