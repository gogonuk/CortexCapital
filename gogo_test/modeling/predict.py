"""
This script uses a trained model to make predictions on new data.

It loads the latest processed data and the trained model, then uses the model
to predict the next day's closing price. This script is intended to be used
for inference after a model has been trained and saved.
"""
from pathlib import Path

import pandas as pd
from loguru import logger
import joblib
import typer

from gogo_test.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_regimes.csv",
    model_path: Path = MODELS_DIR / "random_forest_model.pkl",
):
    """Makes a prediction using the trained model.

    This function loads the latest data, prepares the features, and then uses
    the trained model to predict the next day's closing price.

    Args:
        input_path (Path, optional): The path to the processed data file.
            Defaults to PROCESSED_DATA_DIR / "aapl_with_features.csv".
        model_path (Path, optional): The path to the trained model.
            Defaults to MODELS_DIR / "random_forest_model.pkl".
    """
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Drop rows with NaN values (due to feature calculation)
    data.dropna(inplace=True)

    # One-hot encode the 'regime' feature
    data = pd.get_dummies(data, columns=['regime'], prefix='regime')

    # Get the last row for prediction
    latest_data = data.iloc[[-1]]

    # Prepare features (X) for prediction
    features = [
        'Close', 'SMA', 'RSI', 'MACD', 'MACD_Signal',
        'BBL', 'BBM', 'BBH', 'Close_Lag1', 'Volume_Lag1', 'Volume_SMA'
    ]
    # Add the one-hot encoded regime columns to the feature list
    regime_cols = [col for col in data.columns if 'regime_' in col]
    features.extend(regime_cols)

    # Ensure all expected feature columns exist, fill missing with 0
    # This handles cases where the latest data point doesn't represent all possible regimes
    model_features = joblib.load(model_path).feature_names_in_
    for col in model_features:
        if col not in latest_data.columns:
            latest_data[col] = 0
    X_predict = latest_data[model_features] # Use model's feature order

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logger.info("Making prediction...")
    prediction = model.predict(X_predict)[0]

    logger.success(f"Predicted next day's closing price: {prediction:.2f}")


if __name__ == "__main__":
    app()