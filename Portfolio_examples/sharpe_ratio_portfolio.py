"""
Sharpe Ratio Portfolio Optimization Example:

We will create a portfolio that maximizes the Sharpe Ratio using the historical stock data. 
The Sharpe Ratio is a measure of risk-adjusted return, and maximizing it allows us to find the optimal weights for our portfolio. 
Its formula is: Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility. In this example, we will assume a risk-free rate of 0.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the negative Sharpe Ratio objective
def negative_sharpe(weights, mu, cov_matrix):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Add a small epsilon to avoid division by zero
    return - (port_return / (port_vol + 1e-8))

# Define the constraint (weights sum to 1)
def constraint(weights):
    return np.sum(weights) - 1

def main():
    stock_data = pd.read_csv('../Simulated_data/simulated_stock_data.csv')  # Adjust the path if needed

    # Parameters
    fit_time = 500
    n_stocks = stock_data.shape[1] - 1
    n_obs = stock_data.shape[0] - fit_time
    initial_capital = 10000

    # Extract data
    stock_data_fit = stock_data.iloc[:fit_time, 1:]
    stock_data_forecast = stock_data.iloc[fit_time:, 1:]

    # Calculate returns
    stocks_returns_fit = np.log(stock_data_fit / stock_data_fit.shift(1)).dropna()
    stocks_returns_forecast = np.log(stock_data_forecast / stock_data_forecast.shift(1)).dropna()

    # Covariance matrix and expected returns (annualized for scale)
    cov_matrix = stocks_returns_fit.cov() * 252
    mu = stocks_returns_fit.mean().values * 252  # Annualized expected returns

    # Optimization setup
    initial_weights = np.ones(n_stocks) / n_stocks
    bounds = [(0, 1) for _ in range(n_stocks)]
    constraints = {'type': 'eq', 'fun': constraint}

    # Run the optimization (Sharpe maximization)
    result = minimize(negative_sharpe, initial_weights, args=(mu, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = result.x

    # Portfolio simulation (buy and hold)
    initial_investment_per_stock = initial_capital * opt_weights
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
        'mv_weights': opt_weights.tolist()
    }

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Max Sharpe Portfolio")
    plt.title("Portfolio Value Over Time (Max Sharpe Ratio)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

    return performance_metrics, portfolio_values

if __name__ == "__main__":
    main()
    
