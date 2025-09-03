from pathlib import Path

import pandas as pd
from loguru import logger
import typer
import ta

from gogo_test.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "aapl.csv",
    output_path: Path = PROCESSED_DATA_DIR / "aapl_with_features.csv",
    window_size: int = 20,
):
    """Calculates a simple moving average (SMA) of the closing price and other features."""
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Ensure numeric types
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Simple Moving Average (SMA)
    logger.info(f"Calculating {window_size}-day SMA...")
    data["SMA"] = data["Close"].rolling(window=window_size).mean()

    # RSI (Relative Strength Index)
    logger.info("Calculating RSI...")
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # MACD (Moving Average Convergence Divergence)
    logger.info("Calculating MACD...")
    data['MACD'] = ta.trend.macd(data['Close'])
    data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])

    # Bollinger Bands
    logger.info("Calculating Bollinger Bands...")
    data['BBL'] = ta.volatility.bollinger_lband(data['Close'])
    data['BBM'] = ta.volatility.bollinger_mavg(data['Close'])
    data['BBH'] = ta.volatility.bollinger_hband(data['Close'])

    # Lagged Features
    logger.info("Calculating lagged features...")
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Volume_Lag1'] = data['Volume'].shift(1)

    # Volume-based Features
    logger.info(f"Calculating {window_size}-day SMA of Volume...")
    data['Volume_SMA'] = data['Volume'].rolling(window=window_size).mean()

    # Drop rows with NaN values that result from feature calculation
    data.dropna(inplace=True)

    data.to_csv(output_path)
    logger.success(f"Data with features saved to {output_path}")


if __name__ == "__main__":
    app()