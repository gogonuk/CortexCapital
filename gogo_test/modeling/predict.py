from pathlib import Path

import pandas as pd
from loguru import logger
import joblib
import typer

from gogo_test.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_features.csv",
    model_path: Path = MODELS_DIR / "random_forest_model.pkl",
):
    """Uses the trained model to predict the next day's closing price."""
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Drop rows with NaN values (due to feature calculation)
    data.dropna(inplace=True)

    # Get the last row for prediction
    latest_data = data.iloc[[-1]]

    # Prepare features (X) for prediction
    features = [
        'Close', 'SMA', 'RSI', 'MACD', 'MACD_Signal', 
        'BBL', 'BBM', 'BBH', 'Close_Lag1', 'Volume_Lag1', 'Volume_SMA'
    ]
    X_predict = latest_data[features]

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logger.info("Making prediction...")
    prediction = model.predict(X_predict)[0]

    logger.success(f"Predicted next day's closing price: {prediction:.2f}")


if __name__ == "__main__":
    app()