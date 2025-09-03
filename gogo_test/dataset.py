from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
import typer

from gogo_test.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    ticker: str = "AAPL",
    output_file: Path = RAW_DATA_DIR / "aapl.csv",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
):
    """Downloads historical stock data from Yahoo Finance and saves it to a CSV file."""
    logger.info(f"Downloading {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.to_csv(output_file, index=False)
    logger.success(f"Data saved to {output_file}")


if __name__ == "__main__":
    app()