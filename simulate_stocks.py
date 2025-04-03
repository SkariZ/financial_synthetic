import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(133455)
#np.random.seed(13379)

def simulate_ornstein_uhlenbeck(mu, theta, sigma, dt, n_steps, X0):
    """
    Simulate the Ornstein-Uhlenbeck process using the Euler-Maruyama method.
    """

    # Initialize the array to store the process values
    X = np.zeros(n_steps)
    X[0] = X0

    # Simulate the Ornstein-Uhlenbeck process
    for t in range(1, n_steps):
        Z = np.random.normal(0, 1)  # Random normal noise
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * np.sqrt(dt) * Z

    return X

if __name__ == "__main__":

    ##### Simulate Geometric Brownian Motion (GBM) for stock prices #####
    # Parameters
    n_stocks = 120
    n_obs = 2500
    dt = 1/252  # daily steps assuming 252 trading days/year
    time_index = np.arange(n_obs)

    # Initialize the stock prices
    initial_prices = np.random.uniform(50, 150, n_stocks)
    prices = np.zeros((n_obs, n_stocks))
    prices[0, :] = initial_prices

    groups = 6
    group_sizes = [40, 20, 10, 30, 5, 15]  # Number of stocks per group

    # Randomly assign μ (drift) and σ (volatility) per group
    mus_g = np.random.uniform(0.025, 0.125, 4)  # Drift positive
    mus_g = np.concatenate([mus_g, np.random.uniform(-0.1125, -0.025, 2)])  # Drift negative

    sigmas_g = np.random.uniform(0.05, 0.3, groups)  # Volatility

    # Assign μ and σ to each stock based on the group
    mus = []
    sigmas = []
    for i in range(groups):
        mus = np.concatenate([mus, np.repeat(mus_g[i], group_sizes[i])])
        sigmas = np.concatenate([sigmas, np.repeat(sigmas_g[i], group_sizes[i])])
    
    for i in range(n_stocks - sum(group_sizes)-2):
        mus = np.concatenate([mus, np.random.uniform(-0.1, 0.1, 1)])
        sigmas = np.concatenate([sigmas, np.random.uniform(0.05, 0.3, 1)])
     
    # Decide which stocks get seasonality
    seasonal_stocks_mu = np.random.choice([0, 1], size=n_stocks, p=[0.6, 0.4])  # 40% of stocks seasonal
    seasonal_amplitudes = np.random.uniform(0.1, 0.225, n_stocks) * seasonal_stocks_mu
    seasonal_periods = np.random.randint(100, 400, n_stocks) * seasonal_stocks_mu

    # Seasonality for volatility (sigma)
    seasonal_stocks_vol = np.random.choice([0, 1], size=n_stocks, p=[0.6, 0.4])  # 40% of stocks have seasonal
    seasonal_sigma_amplitudes = np.random.uniform(0.05, 0.275, n_stocks) * seasonal_stocks_vol  # Smaller amplitudes
    seasonal_sigma_periods = np.random.randint(50, 150, n_stocks) * seasonal_stocks_vol

    # Jump Parameters
    jump_prob = 0.005  # .05% chance for a jump each day
    jump_mean = 0  # Mean of jump size (no drift in jump)
    jump_std = 0.025  # Standard deviation of jump size

    # Global Seasonality Terms
    global_seasonality_amplitude = 0.45  # Amplitude of the global seasonality
    global_seasonality_period = 756  # Length of the global seasonality (e.g., 252 days = 1 trading year)

    # Global seasonality impacts 70% of the stocks
    impacted_stocks_global = np.random.choice([1, 0], size=n_stocks, p=[0.70, 0.30]) 
    #impacted_stocks_global[-2:] = 0  # Last two stocks are not impacted by global seasonality

    # Stocks not impacted by global seasonality. Take a small subset of stocks to be impacted inversely
    impacted_stocks_global_inv = np.where(impacted_stocks_global == 0)[0]
    impacted_stocks_global_inv = np.random.choice(impacted_stocks_global_inv, size=20, replace=False)

    # Build a stock-to-group map (for the first stocks that follow groups)
    stock_to_group = []
    for group_idx, size in enumerate(group_sizes):
        stock_to_group.extend([group_idx] * size)

    # Assign -1 to the remaining stocks that are not part of any group
    remaining = n_stocks - len(stock_to_group)
    stock_to_group.extend([-1] * remaining)

    # Group-level correlation strength , 0.02 to 0.07
    group_correlation_strength = np.random.uniform(0.02, 0.07, groups)             

    # Generate GBM paths
    for t in range(1, n_obs):
        #rand = np.random.normal(0, 1, n_stocks)
        jump_rand = np.random.rand(n_stocks) < jump_prob

        group_rand = np.random.normal(0, 1, groups)       # Group shocks
        idio_rand = np.random.normal(0, 1, n_stocks)      # Idiosyncratic shocks
        sigma_small = 1e-4 * np.random.normal(0, 1, n_stocks)  # Small noise indepencence for volatility

        # Loop over each stock
        for i in range(n_stocks):
            group_idx = stock_to_group[i]
            if group_idx != -1:  # If the stock is part of a group
                correlated_noise = group_correlation_strength[group_idx] * group_rand[group_idx]
            else:
                correlated_noise = 0  # No group-level correlation for "non-group" stocks
            total_rand = correlated_noise + idio_rand[i]  # Total noise for the stock

            # Calculate the volatility with seasonality
            seasonal_component_vol = 0
            if seasonal_stocks_vol[i] == 1:
                seasonal_component_vol = seasonal_sigma_amplitudes[i] * np.cos(
                    2 * np.pi * t / seasonal_sigma_periods[i]
                )
            sigma_t = sigmas[i] + sigma_small[i] + seasonal_component_vol  # Base volatility + seasonal component

            # Calculate the drift with seasonality
            seasonal_component_mu = 0  # Default if no seasonality
            if seasonal_stocks_mu[i] == 1:
                seasonal_component_mu = seasonal_amplitudes[i] * np.sin(
                    2 * np.pi * t / seasonal_periods[i]
                )
            mu_t = mus[i] + seasonal_component_mu

            # Add the global seasonality term to both drift and volatility
            if impacted_stocks_global[i] == 1:
                global_seasonality_factor = global_seasonality_amplitude * np.sin(2 * np.pi * t / global_seasonality_period)
                mu_t += global_seasonality_factor  # Adding to the drift (global trend)
                sigma_t += global_seasonality_factor * 0.3  # Adding a smaller effect to the volatility (global volatility)

            if i in impacted_stocks_global_inv:
                global_seasonality_factor = -global_seasonality_amplitude * np.sin(2 * np.pi * t / global_seasonality_period)
                mu_t += global_seasonality_factor  # Adding to the drift (global trend)
                sigma_t += global_seasonality_factor * 0.3  # Adding a smaller effect to the volatility (global volatility)

            # Calculate the drift and volatility with seasonality
            price_change = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * total_rand
            
            # Check if a jump occurs and apply it
            if jump_rand[i]:  # If the stock experiences a jump
                jump = np.random.normal(jump_mean, jump_std)  # Randomly sample jump size from normal distribution
                price_change += jump  # Add the jump to the daily price change
        
            # Update the price for the current stock
            prices[t, i] = prices[t-1, i] * np.exp(price_change)  # Update stock price based on the return

    # Metadata for the stocks
    stock_metadata = pd.DataFrame({
        'Stock': [f'Stock_{i+1}' for i in range(n_stocks)],
        'Group': stock_to_group,
        'Drift': mus,
        'Volatility': sigmas,
        'Seasonal_Mu': seasonal_stocks_mu,
        'Seasonal_Vol': seasonal_stocks_vol,
        'Global_Impact': impacted_stocks_global
    })

    ##### Simulate the Ornstein-Uhlenbeck process for a subset of stocks/commodities #####
    n_stocks_ornstein = 10
    mu = np.random.uniform(75, 125, n_stocks_ornstein)          # Long-term mean
    theta = np.random.uniform(0.2, 0.5, n_stocks_ornstein)      # Mean reversion rate
    sigma = np.random.uniform(3, 6, n_stocks_ornstein)          # Volatility (standard deviation)
    dt = 1/252                                                  # Time step size (e.g., 1 day, can be adjusted)
    n_steps = n_obs                                             # Number of time steps (e.g., number of days)
    X0 = np.random.uniform(75, 125, n_stocks_ornstein)          # Initial value (start price)

    # Simulate the Ornstein-Uhlenbeck process
    ou_stocks = np.zeros((n_steps, n_stocks_ornstein))
    for i in range(n_stocks_ornstein):
        X = simulate_ornstein_uhlenbeck(mu[i], theta[i], sigma[i], dt, n_steps, X0[i])
        ou_stocks[:, i] = X

    #Append the OU stocks to the simulated stock prices
    prices = np.concatenate([prices, ou_stocks], axis=1)
    
    # Metadata for the OU stocks
    ou_metadata = pd.DataFrame({
        'Stock': [f'Stock_{i+1}' for i in range(n_stocks_ornstein)],
        'Long_Term_Mean': mu,
        'Mean_Reversion': theta,
        'Volatility': sigma
    })

    ###### Simulate the Heston Model for a subset of stocks/commodities ######
    n_stocks_heston = 20
    S0 = np.random.uniform(75, 125, n_stocks_heston) # Initial price
    mu = 0.07  # Drift
    kappa = 1.0  # Mean-reversion speed for volatility
    theta = 0.05  # Long-term volatility variance
    sigma = 0.2  # Volatility of volatility
    v0 = np.random.uniform(0.04, 0.8, n_stocks_heston)  # Initial volatility
    T = 1  # Time horizon (1 year)
    n_steps = n_obs  # Daily steps
    n_simulations = n_stocks_heston  # Simulations

    # Time discretization
    dt = T / n_steps
    prices_h = np.zeros((n_simulations, n_steps))
    volatilities = np.zeros((n_simulations, n_steps))

    # Initial conditions
    prices_h[:, 0] = S0
    volatilities[:, 0] = v0

    # Simulate asset price and volatility using the Heston model
    for i in range(n_simulations):
        for t in range(1, n_steps):
            # Stochastic volatility model (Ornstein-Uhlenbeck for volatility)
            dW_v = np.random.normal(0, np.sqrt(dt))  # Volatility noise
            v_t = volatilities[i, t-1]
            v_t_new = v_t + kappa * (theta - v_t) * dt + sigma * np.sqrt(v_t) * dW_v
            volatilities[i, t] = max(v_t_new, 0)  # Ensure non-negative volatility
            
            # Asset price dynamics (GBM with stochastic volatility)
            dW_s = np.random.normal(0, np.sqrt(dt))  # Asset price noise
            price_change = (mu - 0.5 * volatilities[i, t] ** 2) * dt + np.sqrt(volatilities[i, t]) * dW_s
            prices_h[i, t] = prices_h[i, t-1] * np.exp(price_change)

    #Append the Heston stocks to the simulated stock prices
    prices = np.concatenate([prices, prices_h.T], axis=1)

    # Metadata for the Heston stocks
    heston_metadata = pd.DataFrame({
        'Stock': [f'Stock_{i+1}' for i in range(n_stocks_heston)],
        'Initial_Price': S0,
        'Initial_Volatility': v0,
        'Drift': mu,
        'Mean_Reversion': kappa,
        'Long_Term_Volatility': theta,
        'Volatility_Volatility': sigma
    })

    ##### Simulate 30 (3*10) stocks that have the same daily return (almost) as GBM , stronger correlation than the GBM's above#####
    n_stocks_gbm = 10
    times = 3
    idiosyncratic_strength = 0.4
    for _ in range(times):
        mu_gbm = np.random.uniform(-0.05, 0.15, n_stocks_gbm)   # Drift per stock
        sigma_gbm = np.random.uniform(0.15, 0.3, n_stocks_gbm) # Volatility per stock
        prices_gbm = np.zeros((n_obs, n_stocks_gbm))
        prices_gbm[0, :] = np.random.uniform(50, 150, n_stocks_gbm)

        seasonal_amp_gbm = np.random.uniform(0.075, 0.2, n_stocks_gbm)
        seasonal_p_gbm = np.random.randint(100, 400, n_stocks_gbm)

        for t in range(1, n_obs):
            common_rand = np.random.normal(0, 1)
            idio_rand = np.random.normal(0, 1, n_stocks_gbm)
            for i in range(n_stocks_gbm):
                seasonal_component = seasonal_amp_gbm[i] * np.sin(2 * np.pi * t / seasonal_p_gbm[i])
                mu_gbm_tmp = mu_gbm[i] + seasonal_component
                price_change = (
                    (mu_gbm_tmp - 0.5 * sigma_gbm[i] ** 2) * dt
                    + sigma_gbm[i] * np.sqrt(dt) * (common_rand + idiosyncratic_strength * idio_rand[i])
                )
                prices_gbm[t, i] = prices_gbm[t - 1, i] * np.exp(price_change)

        #Append the GBM stocks to the simulated stock prices
        prices = np.concatenate([prices, prices_gbm], axis=1)

    # Metadata for the GBM stocks
    gbm_metadata = pd.DataFrame({
        'Stock': [f'Stock_{i+1}' for i in range(n_stocks_gbm)],
        'Drift': mu_gbm,
        'Volatility': sigma_gbm,
        'Seasonal_Amplitude': seasonal_amp_gbm,
        'Seasonal_Period': seasonal_p_gbm
    })

    # Randomize the order of the stocks
    stock_order = np.random.permutation(len(prices[0]))
    prices = prices[:, stock_order]

    # Convert to DataFrame. Time index is the date range from 2017-01-01, has no real meaning
    dates = pd.date_range(start="2017-01-01", periods=n_obs, freq='B')  # Business days
    price_df = pd.DataFrame(prices, index=dates, columns=[f"Stock_{i+1}" for i in range(len(prices[0]))])

    # Show 10 time-series in 10 plots
    for i in range(0, len(prices[0]), 10):
        price_df.iloc[:, i:i+10].plot(figsize=(12, 6), title="Sample Simulated Stock Prices")
        price_df.mean(axis=1).plot(color='black', label='Mean')
        plt.show()
    
    #Find the correlation matrix
    corr_matrix = price_df.corr().round(4)
    plt.figure(figsize=(5, 5))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Matrix (random order)')
    plt.savefig("Imgs/simulated_stock_correlation_matrix.png", dpi=300, bbox_inches='tight')

    # Save to CSV for student use
    price_df.index.name = 'Date'
    price_df.to_csv("Simulated_data/simulated_stock_data.csv")

    # Plot all stock prices and the mean
    price_df.plot(figsize=(12, 5), title="Simulated Stock Prices", legend=False)
    price_df.mean(axis=1).plot(color='black', label='Mean of all stocks', linewidth=3)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    #plt.ylim([0, 1000])
    plt.savefig("Imgs/simulated_stock_prices.png", dpi=300, bbox_inches='tight')
    plt.show()

    #Count how many stocks have a positive return from the beginning to the end
    positive_returns = (price_df.iloc[-1] > price_df.iloc[0]).sum()
    print(f"Number of stocks with positive returns: {positive_returns}")

    #Count how many stocks have a negative return from the beginning to the end
    negative_returns = (price_df.iloc[-1] < price_df.iloc[0]).sum()
    print(f"Number of stocks with negative returns: {negative_returns}")

    #Mean increace of the stock prices from the beginning to the end
    median_increase = ((price_df.iloc[-1] - price_df.iloc[0]) / price_df.iloc[0]).median()
    print(f"Median increase of stock prices: {100*median_increase:.4f}%")


