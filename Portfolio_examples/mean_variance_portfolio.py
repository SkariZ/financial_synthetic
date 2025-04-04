"""
Mean-Variance Portfolio Optimization Example:

Similar to the Sharpe Ratio optimization, this example demonstrates how to optimize a portfolio using the Mean-Variance approach. 
But instead of maximizing the Sharpe Ratio, we minimize the negative of the expected return minus a risk aversion term multiplied by the variance. 
This approach is more traditional in finance and allows for a more direct control over risk preferences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function (Mean-Variance)
def mean_variance_objective(weights, mu, cov_matrix, risk_aversion):
    expected_return = np.dot(weights, mu)
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    # Negative because we minimize
    return - (expected_return - risk_aversion * variance)

# Define the constraint (weights sum to 1)
def constraint(weights):
    return np.sum(weights) - 1

def main():
    # Load the simulated stock data
    stock_data = pd.read_csv('../Simulated_data/simulated_stock_data.csv')  # Adjust the path if needed

    # Parameters
    fit_time = 500                          # Fit the model on the first 500 days
    n_stocks = stock_data.shape[1] - 1      # Number of stocks
    n_obs = stock_data.shape[0] - fit_time  # Forecast period
    initial_capital = 10000                 # Starting capital
    dt = 1 / 252                            # Daily steps (for scaling if needed)
    risk_aversion = 1e2                     # Adjust this parameter to control risk preference

    # Extract time and stock prices
    time_data = stock_data.iloc[:, 0]
    stock_data_fit = stock_data.iloc[:fit_time, 1:]
    stock_data_forecast = stock_data.iloc[fit_time:, 1:]

    # Log returns
    stocks_returns_fit = np.log(stock_data_fit / stock_data_fit.shift(1)).dropna()
    stocks_returns_forecast = np.log(stock_data_forecast / stock_data_forecast.shift(1)).dropna()

    # Covariance matrix and expected returns (mean returns over the fit period)
    cov_matrix = stocks_returns_fit.cov() * 1e3  # Optional scaling for stability
    mu = stocks_returns_fit.mean().values * 1e3  # Scaled mean return per stock

    # Initial guess (equal weights) and bounds
    initial_weights = np.ones(n_stocks) / n_stocks
    bounds = [(0, 1) for _ in range(n_stocks)]
    constraints = {'type': 'eq', 'fun': constraint}

    # Run the optimization (Mean-Variance)
    result = minimize(mean_variance_objective, initial_weights, args=(mu, cov_matrix, risk_aversion),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    mv_weights = result.x
    #print('Optimized Portfolio Weights:', mv_weights)

    # Initial investment per stock based on optimized weights
    initial_investment_per_stock = initial_capital * mv_weights

    # Track portfolio value over time (Buy and Hold)
    portfolio_values = np.zeros(n_obs)
    purchase_price = stock_data_forecast.iloc[0, :].values
    for t in range(n_obs):
        current_prices = stock_data_forecast.iloc[t, :].values
        portfolio_value = np.sum(current_prices * initial_investment_per_stock / purchase_price)
        portfolio_values[t] = portfolio_value

    # Performance Metrics
    final_portfolio_value = portfolio_values[-1]
    total_return_perc = 100 * (final_portfolio_value - initial_capital) / initial_capital
    max_drawdown_perc = 100 * np.min(portfolio_values - np.maximum.accumulate(portfolio_values)) / initial_capital
    annualized_return_perc = 100 * (((final_portfolio_value / initial_capital) ** (252 / n_obs)) - 1)
    daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    sharp_ratio = np.mean(daily_returns) / np.std(daily_returns)

    print("\nFinal Portfolio Value:", final_portfolio_value)
    print('Total Return:', total_return_perc, '%')
    print('Maximum Drawdown:', max_drawdown_perc, '%')
    print('Annualized Return:', annualized_return_perc, '%')
    print('Sharpe Ratio:', sharp_ratio)

    # Create a dictionary to store performance metrics
    performance_metrics = {
        'Final Portfolio Value': final_portfolio_value,
        'Total Return (%)': total_return_perc,
        'Maximum Drawdown (%)': max_drawdown_perc,
        'Annualized Return (%)': annualized_return_perc,
        'Sharpe Ratio': sharp_ratio,
        'mv_weights': mv_weights.tolist()
    }

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Mean-Variance Optimized Portfolio")
    plt.title("Portfolio Value Over Time (Mean-Variance Optimization)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

    return performance_metrics, portfolio_values

if __name__ == "__main__":
    main()
