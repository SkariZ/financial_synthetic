"""
Simulate time series data with Geometric Brownian Motion (GBM) and flexible seasonality.
This script generates multiple time series with different starting prices, expected returns (mu), and volatilities (sigma). The time series are seasonal and independent of each other.
The seasonality is modeled as a sine wave with a specified amplitude and frequency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(123456)

# Parameters for Geometric Brownian Motion (with flexible seasonality and different mu values)
n_series = 20  # Number of time series
n_days = 1200  # Number of trading days (this can be any number of days)
seasonal_amplitude = 0.0035 # Seasonality amplitude (how much seasonality affects the returns)
seasonal_frequency = 4  # Number of seasonal peaks (e.g., 4 for quarterly-like seasonality)
time = np.linspace(0, 1, n_days)  # Time from 0 to 1 (normalized for given days)

# Randomly generate different starting prices for each series between 50 and 200
starting_prices = np.random.uniform(50, 200, n_series)

# Randomly generate different mu values (expected returns) for each series
mu_values = np.random.uniform(-0.5, 0.5, n_series)

# Randomly generate different sigma values (volatility) for each series
sigma_values = np.random.uniform(0.25, 0.5, n_series)

# Directory to save individual time series CSV files
output_dir = 'Simulated_data/Synthetic/same_seasonality/'
os.makedirs(output_dir, exist_ok=True)

# Simulate the time series data
simulated_data = {}
for i in range(1, n_series + 1):
    # Generate a random starting price for each time series
    start_price = starting_prices[i - 1]
    
    # Use a different mu and sigma for each series
    mu = mu_values[i - 1]
    sigma = sigma_values[i - 1]
    
    # Generate seasonality as a sine wave with flexible periodicity
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * seasonal_frequency * time)
    
    # Simulate returns using GBM with seasonal component
    dt = 1 / n_days  # Time step (daily)
    random_walk = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    
    # Add the seasonality component to the returns
    seasonal_returns = random_walk + seasonality
    
    # Cumulative sum of returns to get the price series
    log_returns = np.cumsum(seasonal_returns)  # Cumulative sum of seasonal returns
    price_series = start_price * np.exp(log_returns)  # GBM formula with seasonality

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
plt.title('GBMs with same seasonality and different mu/sigma')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='best', ncol=4, fontsize='small')
plt.savefig('Imgs/simulated_series_seasonality_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Simulated time series data with flexible seasonality saved to {output_file}")