import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

# Parameters
n_steps = 1000           # Reduced number of time steps in the simulation
n_runs = 500            # Number of stock price simulations
initial_price = 100      # Initial stock price
sigma = 1.0              # Volatility (std dev of price change)

# Set seed for reproducibility
np.random.seed(0)

@jit(nopython=True, cache=True)
def simulate_stock_prices(initial_price, n_steps, sigma):
    returns = np.random.normal(loc=0, scale=sigma, size=n_steps)
    prices = np.empty(n_steps + 1)  # Preallocate array for performance
    prices[0] = initial_price       # Set the initial price
    for t in range(1, n_steps + 1):
        prices[t] = prices[t - 1] + returns[t - 1]
    return prices

@jit(nopython=True, cache=True)  
def multiple_simulations(n_runs, n_steps, initial_price, sigma):
    all_prices = np.empty((n_runs, n_steps + 1))  # Preallocate array
    for i in range(n_runs):
        all_prices[i] = simulate_stock_prices(initial_price, n_steps, sigma)
    return all_prices

# Simulate stock prices for multiple runs
all_prices = multiple_simulations(n_runs, n_steps, initial_price, sigma)

# Plot 1: Multiple simulated stock price paths
plt.figure(figsize=(14, 6))
for i in range(100):  # Plot only the first 100 paths for better visualization
    plt.plot(all_prices[i], alpha=0.6)
plt.title("Simulation of Stock Prices with gaussian Random Walk", fontsize=16)
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()

# Plot 2: KDE of prices at selected time points (not flipped)
times_to_plot = [0, 250, 500, 750, 1000]  # Adjusted time points to fit within 1000 steps

plt.figure(figsize=(14, 6))
for t in times_to_plot:
    sns.kdeplot(all_prices[:, t], label=f"Time {t}", fill=True, alpha=0.4)

plt.title("Probability Distribution of Stock Price at different Times", fontsize=16)
plt.xlabel("Stock Price", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
