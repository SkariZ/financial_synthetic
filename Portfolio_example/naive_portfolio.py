"""
Naive Portfolio Example:

The naive portfolio is a simple investment strategy where an investor allocates equal weights to all available stocks in the portfolio.
This approach does not take into account the risk or return characteristics of the individual stocks, making it a straightforward but potentially suboptimal strategy.
We calculate the portfolio value over time assuming a buy and hold strategy (no rebalancing).
This portfolio and can be used as a benchmark for comparison with other portfolio strategies.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
     # Load the simulated stock data
    stock_data = pd.read_csv('../Simulated_data/simulated_stock_data.csv')  # Load stock data into a DataFrame

    # Parameters
    fit_time = 500                              # Fit the model on the first 1000 days (for comparison with minimum variance portfolio)
    n_stocks = stock_data.shape[1]-1            # Number of stocks is the number of columns
    n_obs = stock_data.shape[0] - fit_time      # Number of observations (days)
    initial_capital = 10000                     # 10,000 units of capital
    dt = 1 / 252                                # daily steps assuming 252 trading days per year

    # The first column as time data
    time_data = stock_data.iloc[:, 0]

    # The remaining columns as stock prices  
    stock_data_fit = stock_data.iloc[:fit_time, 1:]
    stock_data_forecast = stock_data.iloc[fit_time:, 1:]
    
    # Calculate equal weights for naive portfolio
    weights = np.ones(n_stocks) / n_stocks  # Equal-weighted portfolio
    initial_investment_per_stock = initial_capital * weights  # Initial investment in each stock

    # Initialize portfolio value array
    portfolio_values = np.zeros(n_obs)

    # Calculate portfolio value over time (assuming buy and hold - no rebalancing)
    purchase_price = stock_data_forecast.iloc[0, :].values  # Price at the time of purchase
    for t in range(n_obs):
        current_prices = stock_data_forecast.iloc[t, :].values
        portfolio_value = np.sum(current_prices * initial_investment_per_stock / purchase_price)
        portfolio_values[t] = portfolio_value

    print('Final Portfolio Value:', portfolio_values[-1])
    print('Percentage Return:', 100*(portfolio_values[-1] - initial_capital) / initial_capital, '%')
    print('Maximum Drawdown:', 100* np.min(portfolio_values - np.maximum.accumulate(portfolio_values)) / initial_capital, '%')
    print('Yearly Percentage Return:', 100*((portfolio_values[-1] / initial_capital) ** (252 / n_obs) - 1), '%')
    daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    print('Sharpe Ratio:', np.mean(daily_returns) / np.std(daily_returns))

    # Plot the portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Naive Portfolio')
    plt.title('Naive Portfolio Value Over Time (10,000 Initial Investment)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
