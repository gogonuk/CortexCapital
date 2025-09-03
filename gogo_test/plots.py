from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import typer

from gogo_test.config import PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_features.csv",
    output_path: Path = FIGURES_DIR / "aapl_close_vs_sma.png",
):
    """Creates a plot of the closing price and the SMA."""
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    logger.info("Creating plot...")
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close Price")
    plt.plot(data["SMA"], label="20-Day SMA")
    plt.title("AAPL Close Price vs. 20-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    logger.success(f"Plot saved to {output_path}")


if __name__ == "__main__":
    app()