"""
# Simulate multiple time series using Geometric Brownian Motion (GBM) with different starting prices
They are generated with the same drift (mu) and volatility (sigma) parameters. But are independent of each other.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(123)

# Parameters for Geometric Brownian Motion (with flexible seasonality and different mu values)
n_series = 20  # Number of time series
n_days = 1200  # Number of trading days (this can be any number of days)

time = np.linspace(0, 1, n_days)  # Time from 0 to 1 (normalized for given days)

# Randomly generate different starting prices for each series between 50 and 200
starting_prices = np.random.uniform(50, 200, n_series)

mu = 0.5  # Expected return (drift) for all series
sigma = 0.2  # Volatility for all series

# Directory to save individual time series CSV files
output_dir = 'Simulated_data/Synthetic/same_drift/'
os.makedirs(output_dir, exist_ok=True)

# Simulate the time series data
simulated_data = {}
for i in range(1, n_series + 1):
    # Generate a random starting price for each time series
    start_price = starting_prices[i - 1]
    
    # Simulate returns using GBM with seasonal component
    dt = 1 / n_days  # Time step (daily)
    random_walk = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    
    # Cumulative sum of returns to get the price series
    log_returns = np.cumsum(random_walk)  # Cumulative sum returns
    price_series = start_price * np.exp(log_returns)  # GBM formula

    # Create a DataFrame for the current time series
    df = pd.DataFrame({
        'Day': np.arange(1, n_days + 1),
        'Price': price_series
    })
    
    # Save the individual time series to a CSV file
    output_file = os.path.join(output_dir, f'simulated_series_{i}.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Saved Series {i} to {output_file}")

# Plot the simulated time series in one figure
plt.figure(figsize=(12, 5))
for i in range(1, n_series + 1):
    df = pd.read_csv(os.path.join(output_dir, f'simulated_series_{i}.csv'))
    # Plot each series
    plt.plot(df['Day'], df['Price'], label=f'Series {i}')
plt.title('GBMs with same drift and volatility')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='best', ncol=4, fontsize='small')
plt.savefig('Imgs/simulated_series_plot.png', dpi=300, bbox_inches='tight')
plt.show()