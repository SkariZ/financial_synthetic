
"""
Minimum Variance Portfolio:

The minimum variance portfolio is a portfolio of assets that has the lowest possible risk (variance) for a given level of expected return.
This strategy is based on the idea that by diversifying across multiple assets, we can reduce the overall risk of the portfolio.

In this example, we will use the covariance matrix of stock returns to find the optimal weights for the minimum variance portfolio.
We will assume that the investor has a fixed amount of capital to invest and will not rebalance the portfolio over time.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function (portfolio variance)
def portfolio_variance(weights, corr_matrix):
    return np.dot(weights.T, np.dot(corr_matrix, weights))

# Define the constraint (weights sum to 1)
def constraint(weights):
    return np.sum(weights) - 1

def main():
    # Load the simulated stock data
    stock_data = pd.read_csv('../Simulated_data/simulated_stock_data.csv')  # Load stock data into a DataFrame

    # Parameters
    fit_time = 500                          # Fit the model on the first 1000 days
    n_stocks = stock_data.shape[1] - 1      # Number of stocks is the number of columns
    n_obs = stock_data.shape[0] - fit_time  # Number of observations (days)
    initial_capital = 10000                 # 10,000 units of capital
    dt = 1 / 252                            # daily steps assuming 252 trading days per year

    # The first column as time data
    time_data = stock_data.iloc[:, 0]

    # The remaining columns as stock prices  
    stock_data_fit = stock_data.iloc[:fit_time, 1:]
    stock_data_forecast = stock_data.iloc[fit_time:, 1:]

    # Log stock returns
    stocks_returns_fit = np.log(stock_data_fit / stock_data_fit.shift(1)).dropna()
    stocks_returns_forecast = np.log(stock_data_forecast / stock_data_forecast.shift(1)).dropna()

    # Now use stock_returns for your covariance matrix, scale it by 1e3 for better optimization...
    cov_matrix = stocks_returns_fit.cov()*1e3

    # Calculate the number of stocks
    n_stocks = len(stocks_returns_fit.columns)

    # Set up the initial guess for the weights (equally distributed)
    initial_weights = np.ones(n_stocks) / n_stocks

    # Set up bounds for the weights (each weight should be between 0 and 1)
    bounds = [(0, 1) for _ in range(n_stocks)]

    # Define constraints (the sum of weights should be 1)
    constraints = ({'type': 'eq', 'fun': constraint})

    # Use Scipy's minimize function to minimize the portfolio variance
    result = minimize(portfolio_variance, initial_weights, args=(cov_matrix,), method='SLSQP',
                    bounds=bounds, constraints=constraints)

    # Extract the optimized weights (Minimum Variance Portfolio weights)
    mv_weights = result.x

    # Initial investment in each stock
    initial_investment_per_stock = initial_capital * mv_weights

    # Initialize portfolio value array
    portfolio_values = np.zeros(n_obs)

    # Calculate portfolio value over time (assuming buy and hold - no rebalancing)
    purchase_price = stock_data_forecast.iloc[0, :].values  # Price at the time of purchase
    for t in range(n_obs):
        current_prices = stock_data_forecast.iloc[t, :].values
        portfolio_value = np.sum(current_prices * initial_investment_per_stock / purchase_price)
        portfolio_values[t] = portfolio_value

    # Print the final portfolio value
    print("Final Portfolio Value:", portfolio_values[-1])
    print('Percentage Return:', 100*(portfolio_values[-1] - initial_capital) / initial_capital, '%')
    print('Maximum Drawdown:', 100* np.min(portfolio_values - np.maximum.accumulate(portfolio_values)) / initial_capital, '%')
    print('Yearly Percentage Return:', 100*((portfolio_values[-1] / initial_capital) ** (252 / len(portfolio_values)) - 1), '%')
    daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    print('Sharpe Ratio:', np.mean(daily_returns) / np.std(daily_returns))

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Minimum Variance Portfolio")
    plt.title("Portfolio Value Over Time (Minimum Variance Portfolio)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    
