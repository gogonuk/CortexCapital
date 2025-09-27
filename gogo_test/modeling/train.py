"""
This script trains a Random Forest Regressor model on the processed data using walk-forward validation.

Walk-forward validation is a more realistic method for backtesting time series models.
Instead of a simple train-test split, the model is trained on a sliding window of data
and tested on the next data point. This process is repeated, and the model is retrained
at each step. This simulates how a model would be used in a real-world trading scenario.

The script also calculates a variety of financial metrics to evaluate the performance
of the trading strategy, including:
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Win Rate
- Average Win/Loss
- Number of Trades

The final trained model is saved to the models directory.
"""
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib
import typer
import numpy as np

from gogo_test.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


def calculate_financial_metrics(actual_prices, predicted_prices, prediction_threshold=0.001, transaction_cost=0.0005, take_profit_percentage=0.02, stop_loss_percentage=0.01):
    """Calculates financial metrics for a trading strategy based on model predictions.

    This function simulates a trading strategy where trades are executed based on the
    predicted price change. It incorporates transaction costs, take-profit, and stop-loss
    levels to provide a more realistic backtest.

    Args:
        actual_prices (pd.Series): The actual closing prices of the asset.
        predicted_prices (pd.Series): The prices predicted by the model.
        prediction_threshold (float, optional): The minimum predicted price change required
            to trigger a trade. Defaults to 0.001.
        transaction_cost (float, optional): The cost of each transaction as a percentage
            of the trade value. Defaults to 0.0005.
        take_profit_percentage (float, optional): The percentage gain at which to close
            a position and take profit. Defaults to 0.02.
        stop_loss_percentage (float, optional): The percentage loss at which to close
            a position and stop losses. Defaults to 0.01.

    Returns:
        dict: A dictionary containing the calculated financial metrics.
    """
    signals = pd.Series(0, index=actual_prices.index)
    buy_condition = predicted_prices > actual_prices * (1 + prediction_threshold)
    sell_condition = predicted_prices < actual_prices * (1 - prediction_threshold)
    signals[buy_condition] = 1
    signals[sell_condition] = -1

    portfolio_value = pd.Series(1.0, index=actual_prices.index)
    current_position = 0
    entry_price = 0
    trades = []

    for i in range(len(actual_prices) - 1):
        current_price = actual_prices.iloc[i]
        next_price = actual_prices.iloc[i+1]
        daily_return = (next_price - current_price) / current_price

        if current_position == 0:
            if signals.iloc[i] == 1:
                current_position = 1
                entry_price = current_price
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - transaction_cost)
                trades.append({'entry_price': entry_price, 'exit_price': None, 'type': 'long', 'status': 'open', 'entry_date': actual_prices.index[i]})
            elif signals.iloc[i] == -1:
                current_position = -1
                entry_price = current_price
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - transaction_cost)
                trades.append({'entry_price': entry_price, 'exit_price': None, 'type': 'short', 'status': 'open', 'entry_date': actual_prices.index[i]})
            else:
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i]
        else:
            exit_condition = False
            if current_position == 1:
                if next_price >= entry_price * (1 + take_profit_percentage) or next_price <= entry_price * (1 - stop_loss_percentage):
                    exit_condition = True
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + daily_return)
            elif current_position == -1:
                if next_price <= entry_price * (1 - take_profit_percentage) or next_price >= entry_price * (1 + stop_loss_percentage):
                    exit_condition = True
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - daily_return)

            if exit_condition:
                portfolio_value.iloc[i+1] *= (1 - transaction_cost)
                if trades:
                    trades[-1].update({'exit_price': next_price, 'status': 'closed'})
                current_position = 0

    total_return = (portfolio_value.iloc[-1] - 1) * 100
    strategy_returns = portfolio_value.pct_change().dropna()
    annualized_return = strategy_returns.mean() * 252
    annualized_std = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_std if annualized_std != 0 else 0

    downside_returns = strategy_returns[strategy_returns < 0]
    sortino_ratio = annualized_return / (downside_returns.std() * np.sqrt(252)) if downside_returns.std() != 0 else 0
    
    peak = portfolio_value.expanding(min_periods=1).max()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    calmar_ratio = annualized_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0

    closed_trades = [t for t in trades if t['status'] == 'closed']
    num_trades = len(closed_trades)

    if num_trades > 0:
        winning_trades = sum(1 for t in closed_trades if (t['type'] == 'long' and t['exit_price'] > t['entry_price']) or (t['type'] == 'short' and t['exit_price'] < t['entry_price']))
        win_rate = (winning_trades / num_trades) * 100
        
        total_profit = sum(t['exit_price'] - t['entry_price'] if t['type'] == 'long' else t['entry_price'] - t['exit_price'] for t in closed_trades)
        average_win = total_profit / winning_trades if winning_trades > 0 else 0
        
        losing_trades = num_trades - winning_trades
        average_loss = (total_profit - (winning_trades * average_win)) / losing_trades if losing_trades > 0 else 0
    else:
        win_rate = 0
        average_win = 0
        average_loss = 0

    return {
        "total_return": total_return, "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio, "calmar_ratio": calmar_ratio, "num_trades": num_trades,
        "win_rate": win_rate, "average_win": average_win, "average_loss": average_loss,
        "portfolio_value": portfolio_value, "trades": trades
    }


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_regimes.csv",
    model_path: Path = MODELS_DIR / "random_forest_model.pkl",
    initial_train_size: int = 252, # Approximately 1 year of trading days
    forecast_horizon: int = 1, # Predict 1 day ahead
    prediction_threshold: float = 0.001, # Only trade if predicted change > 0.1%
    transaction_cost: float = 0.0005, # 0.05% transaction cost per trade
    take_profit_percentage: float = 0.02, # Take profit at 2% gain
    stop_loss_percentage: float = 0.01, # Stop loss at 1% loss
):
    """Trains a Random Forest Regressor model using walk-forward validation.

    This function orchestrates the entire model training and evaluation process.
    It loads the data, performs walk-forward validation, calculates financial metrics,
    and saves the final trained model.

    Args:
        input_path (Path, optional): The path to the processed data file.
            Defaults to PROCESSED_DATA_DIR / "aapl_with_features.csv".
        model_path (Path, optional): The path to save the trained model.
            Defaults to MODELS_DIR / "random_forest_model.pkl".
        initial_train_size (int, optional): The initial size of the training set
            for walk-forward validation. Defaults to 252.
        forecast_horizon (int, optional): The number of days to forecast ahead.
            Defaults to 1.
        prediction_threshold (float, optional): The minimum predicted price change
            required to trigger a trade. Defaults to 0.001.
        transaction_cost (float, optional): The cost of each transaction as a
            percentage of the trade value. Defaults to 0.0005.
        take_profit_percentage (float, optional): The percentage gain at which to
            close a position and take profit. Defaults to 0.02.
        stop_loss_percentage (float, optional): The percentage loss at which to
            close a position and stop losses. Defaults to 0.01.
    """
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Drop rows with NaN values (due to feature calculation)
    data.dropna(inplace=True)

    # One-hot encode the 'regime' feature
    data = pd.get_dummies(data, columns=['regime'], prefix='regime')

    # Prepare features (X) and target (y)
    # Predict the next day's closing price
    data['target'] = data['Close'].shift(-forecast_horizon)
    data.dropna(inplace=True) # Drop rows with NaN for target

    features = [
        'Close', 'SMA', 'RSI', 'MACD', 'MACD_Signal',
        'BBL', 'BBM', 'BBH', 'Close_Lag1', 'Volume_Lag1', 'Volume_SMA'
    ]
    # Add the one-hot encoded regime columns to the feature list
    regime_cols = [col for col in data.columns if 'regime_' in col]
    features.extend(regime_cols)

    X = data[features]
    y = data['target']

    all_predictions = []
    all_actuals = []
    all_actual_close_prices = [] # Store actual close prices for financial metrics
    best_params = {}

    logger.info("Starting walk-forward validation with hyperparameter tuning...")
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    for i in range(initial_train_size, len(data) - forecast_horizon + 1):
        train_X = X.iloc[:i]
        train_y = y.iloc[:i]
        test_X = X.iloc[i:i + forecast_horizon]
        test_y = y.iloc[i:i + forecast_horizon]
        
        actual_close_for_prediction_day = data['Close'].iloc[i:i + forecast_horizon]

        if len(test_X) == 0:
            break

        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Using TimeSeriesSplit for cross-validation in RandomizedSearchCV
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=10,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(train_X, train_y)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        prediction = best_model.predict(test_X)

        all_predictions.extend(prediction)
        all_actuals.extend(test_y)
        all_actual_close_prices.extend(actual_close_for_prediction_day)

    logger.info("Evaluating overall walk-forward performance...")
    mae = mean_absolute_error(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)

    logger.info(f"Overall Mean Absolute Error (Walk-Forward): {mae:.2f}")
    logger.info(f"Overall R-squared (Walk-Forward): {r2:.2f}")
    logger.info(f"Best hyperparameters found: {best_params}")

    # Calculate financial metrics
    logger.info("Calculating financial metrics...")
    actual_prices_series = pd.Series(all_actual_close_prices, index=data.index[initial_train_size:len(data) - forecast_horizon + 1])
    predicted_prices_series = pd.Series(all_predictions, index=data.index[initial_train_size:len(data) - forecast_horizon + 1])

    financial_metrics = calculate_financial_metrics(actual_prices_series, predicted_prices_series, prediction_threshold, transaction_cost, take_profit_percentage, stop_loss_percentage)
    logger.info(f"Total Return: {financial_metrics['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {financial_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {financial_metrics['sortino_ratio']:.2f}")
    logger.info(f"Calmar Ratio: {financial_metrics['calmar_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {financial_metrics['max_drawdown']:.2f}%")
    logger.info(f"Number of Trades: {financial_metrics['num_trades']}")
    logger.info(f"Win Rate: {financial_metrics['win_rate']:.2f}%")
    logger.info(f"Average Win: {financial_metrics['average_win']:.4f}")
    logger.info(f"Average Loss: {financial_metrics['average_loss']:.4f}")

    # For prediction, we'll still save the model trained on the full dataset
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    final_model.fit(X, y)
    logger.info(f"Saving final model (trained on full data with best params) to {model_path}...")
    joblib.dump(final_model, model_path)
    logger.success("Model training and walk-forward validation complete.")


if __name__ == "__main__":
    app()
