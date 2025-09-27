"""
This script generates market regime features using clustering.

It loads the processed data, applies K-Means clustering to identify different
market regimes (e.g., high volatility, trending), and saves the data with an
additional 'regime' column.
"""
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import typer

from gogo_test.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_features.csv",
    output_path: Path = PROCESSED_DATA_DIR / "aapl_with_regimes.csv",
    n_clusters: int = 3,
):
    """Generates market regime features using K-Means clustering.

    Args:
        input_path (Path): Path to the data with features.
        output_path (Path): Path to save the data with regime features.
        n_clusters (int): The number of market regimes to identify.
    """
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Select features for clustering
    features_for_clustering = ['RSI', 'MACD', 'BBM']
    X = data[features_for_clustering].copy()

    # Add other relevant features like volatility
    X['daily_return'] = data['Close'].pct_change()
    X['volatility'] = X['daily_return'].rolling(window=20).std()
    X.dropna(inplace=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-Means clustering
    logger.info(f"Identifying {n_clusters} market regimes...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add regime labels to the original data
    data_with_regimes = data.loc[X.index].copy()
    data_with_regimes["regime"] = clusters

    data_with_regimes.to_csv(output_path)
    logger.success(f"Data with market regimes saved to {output_path}")


if __name__ == "__main__":
    app()
