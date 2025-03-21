import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


if __name__ == "__main__":
    # Load the simulated stock data
    stock_data = pd.read_csv('../simulated_stock_data.csv')  # Load stock data into a DataFrame

     # The first column as time data
    time_data = stock_data.iloc[:, 0]

    # The remaining columns as stock prices  
    stock_data = stock_data.iloc[:, 1:]

    # Log stock returns
    stocks_returns = np.log(stock_data / stock_data.shift(1)).dropna()

    # Covariance matrix of log returns
    cov_matrix = stocks_returns.cov() * 252  # Annualized covariance matrix

    # Mean returns
    mu = stocks_returns.mean()*252  # Annualized returns

    n_stocks = len(mu)

    # Constraint: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n_stocks))  # Long-only, no shorting

    # Generate a range of target returns
    target_returns = np.linspace(mu.min(), mu.max(), 50)

    # Store results
    frontier_volatility = []
    frontier_return = []

    for target in target_returns:
        # Add target return constraint
        constraints_with_target = (
            constraints,
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target}
        )
        
        # Minimize portfolio variance for each target return
        result = minimize(
            lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
            x0=np.ones(n_stocks) / n_stocks,
            bounds=bounds,
            constraints=constraints_with_target,
            method='SLSQP'
        )
        
        if result.success:
            frontier_volatility.append(np.sqrt(result.fun))  # std dev = sqrt(variance)
            frontier_return.append(target)
        else:
            print("Optimization failed for target:", target)

    # Plot Efficient Frontier
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_volatility, frontier_return, 'b--', label='Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Risk)')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
