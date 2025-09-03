from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import typer
import numpy as np

from gogo_test.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


def calculate_financial_metrics(actual_prices, predicted_prices, prediction_threshold=0.001, transaction_cost=0.0005, take_profit_percentage=0.02, stop_loss_percentage=0.01):
    """Calculates financial metrics for a simple trading strategy with a threshold, transaction costs, take-profit, and stop-loss."""
    signals = pd.Series(0, index=actual_prices.index) # 0 for hold
    
    # Generate signals based on threshold
    buy_condition = predicted_prices > actual_prices * (1 + prediction_threshold)
    sell_condition = predicted_prices < actual_prices * (1 - prediction_threshold)
    
    signals[buy_condition] = 1 # Buy signal
    signals[sell_condition] = -1 # Sell signal

    # Initialize portfolio value and returns
    portfolio_value = pd.Series(1.0, index=actual_prices.index)
    current_position = 0 # 0: no position, 1: long, -1: short
    entry_price = 0
    num_trades = 0

    for i in range(len(actual_prices) - 1):
        current_price = actual_prices.iloc[i]
        next_price = actual_prices.iloc[i+1]
        daily_return = (next_price - current_price) / current_price

        if current_position == 0: # No open position
            if signals.iloc[i] == 1: # Buy signal
                current_position = 1
                entry_price = current_price
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - transaction_cost)
                num_trades += 1
            elif signals.iloc[i] == -1: # Sell signal
                current_position = -1
                entry_price = current_price
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - transaction_cost)
                num_trades += 1
            else:
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i]
        else: # Position is open
            if current_position == 1: # Long position
                # Check for Take Profit
                if next_price >= entry_price * (1 + take_profit_percentage): 
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + (next_price - current_price) / current_price) * (1 - transaction_cost) 
                    current_position = 0
                # Check for Stop Loss
                elif next_price <= entry_price * (1 - stop_loss_percentage): 
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + (next_price - current_price) / current_price) * (1 - transaction_cost) 
                    current_position = 0
                else:
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + daily_return)
            elif current_position == -1: # Short position
                # Check for Take Profit
                if next_price <= entry_price * (1 - take_profit_percentage): 
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + (current_price - next_price) / current_price) * (1 - transaction_cost) 
                    current_position = 0
                # Check for Stop Loss
                elif next_price >= entry_price * (1 + stop_loss_percentage): 
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + (current_price - next_price) / current_price) * (1 - transaction_cost) 
                    current_position = 0
                else:
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - daily_return) # Short position return is inverse
            
            # If a new signal is generated while in a position, close current and open new
            if signals.iloc[i] != 0 and signals.iloc[i] != current_position and current_position != 0:
                # Close current position (apply exit cost)
                if current_position == 1: # Long
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 + daily_return) * (1 - transaction_cost)
                elif current_position == -1: # Short
                    portfolio_value.iloc[i+1] = portfolio_value.iloc[i] * (1 - daily_return) * (1 - transaction_cost)
                
                # Open new position (apply entry cost)
                current_position = signals.iloc[i]
                entry_price = current_price
                portfolio_value.iloc[i+1] = portfolio_value.iloc[i+1] * (1 - transaction_cost)
                num_trades += 1 # Increment trade count for new position

    # Calculate total return
    total_return = (portfolio_value.iloc[-1] - 1) * 100

    # Calculate Sharpe Ratio
    strategy_returns = portfolio_value.pct_change().dropna()
    annualized_return = strategy_returns.mean() * 252
    annualized_std = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_std if annualized_std != 0 else 0

    # Calculate Maximum Drawdown
    peak = portfolio_value.expanding(min_periods=1).max()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min() * 100

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
    }


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "aapl_with_features.csv",
    model_path: Path = MODELS_DIR / "random_forest_model.pkl",
    initial_train_size: int = 252, # Approximately 1 year of trading days
    forecast_horizon: int = 1, # Predict 1 day ahead
    prediction_threshold: float = 0.001, # Only trade if predicted change > 0.1%
    transaction_cost: float = 0.0005, # 0.05% transaction cost per trade
    take_profit_percentage: float = 0.02, # Take profit at 2% gain
    stop_loss_percentage: float = 0.01, # Stop loss at 1% loss
):
    """Trains a Random Forest Regressor model using walk-forward validation."""
    logger.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path, index_col="Date", parse_dates=True)

    # Drop rows with NaN values (due to feature calculation)
    data.dropna(inplace=True)

    # Prepare features (X) and target (y)
    # Predict the next day's closing price
    data['target'] = data['Close'].shift(-forecast_horizon)
    data.dropna(inplace=True) # Drop rows with NaN for target

    features = [
        'Close', 'SMA', 'RSI', 'MACD', 'MACD_Signal', 
        'BBL', 'BBM', 'BBH', 'Close_Lag1', 'Volume_Lag1', 'Volume_SMA'
    ]
    X = data[features]
    y = data['target']

    all_predictions = []
    all_actuals = []
    all_actual_close_prices = [] # Store actual close prices for financial metrics

    logger.info("Starting walk-forward validation...")
    for i in range(initial_train_size, len(data) - forecast_horizon + 1):
        train_X = X.iloc[:i]
        train_y = y.iloc[:i]
        test_X = X.iloc[i:i + forecast_horizon]
        test_y = y.iloc[i:i + forecast_horizon]
        
        # Get the actual close price for the day the prediction is made for
        actual_close_for_prediction_day = data['Close'].iloc[i:i + forecast_horizon]

        if len(test_X) == 0: # Handle cases where forecast_horizon goes beyond data
            break

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_X, train_y)

        prediction = model.predict(test_X)

        all_predictions.extend(prediction)
        all_actuals.extend(test_y)
        all_actual_close_prices.extend(actual_close_for_prediction_day)

    logger.info("Evaluating overall walk-forward performance...")
    mae = mean_absolute_error(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)

    logger.info(f"Overall Mean Absolute Error (Walk-Forward): {mae:.2f}")
    logger.info(f"Overall R-squared (Walk-Forward): {r2:.2f}")

    # Calculate financial metrics
    logger.info("Calculating financial metrics...")
    # Convert lists to pandas Series for easier calculation
    actual_prices_series = pd.Series(all_actual_close_prices, index=data.index[initial_train_size:len(data) - forecast_horizon + 1])
    predicted_prices_series = pd.Series(all_predictions, index=data.index[initial_train_size:len(data) - forecast_horizon + 1])

    financial_metrics = calculate_financial_metrics(actual_prices_series, predicted_prices_series, prediction_threshold, transaction_cost, take_profit_percentage, stop_loss_percentage)
    logger.info(f"Total Return: {financial_metrics['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {financial_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {financial_metrics['max_drawdown']:.2f}%")
    logger.info(f"Number of Trades: {financial_metrics['num_trades']}")

    # For prediction, we'll still save the model trained on the full dataset
    # This is a common practice for deployment, but evaluation is done via walk-forward
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    logger.info(f"Saving final model (trained on full data) to {model_path}...")
    joblib.dump(final_model, model_path)
    logger.success("Model training and walk-forward validation complete.")


if __name__ == "__main__":
    app()
