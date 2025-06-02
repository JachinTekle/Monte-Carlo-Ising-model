import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
n_steps = 500           # Number of time steps in the simulation
n_runs = 100            # Number of stock price simulations
initial_price = 100
sigma = 1.0             # Volatility (std dev of price change)

# Set seed for reproducibility
np.random.seed(0)

# Simulate many runs and store prices
all_prices = np.zeros((n_runs, n_steps + 1))
all_prices[:, 0] = initial_price

for i in range(n_runs):
    returns = np.random.normal(loc=0, scale=sigma, size=n_steps)
    prices = initial_price + np.cumsum(returns)
    prices = np.insert(prices, 0, initial_price)  # start at same price
    all_prices[i] = prices

# Plot 1: Multiple simulated stock price paths
plt.figure(figsize=(14, 6))
for i in range(n_runs):
    plt.plot(all_prices[i], alpha=0.2, color='blue')
plt.title("Simulated Stock Prices with Gaussian Steps (Random Walk)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()

# Plot 2: KDE of prices at selected time points
times_to_plot = [0, 100, 250, 500]

plt.figure(figsize=(14, 6))
for t in times_to_plot:
    sns.kdeplot(all_prices[:, t], label=f"Time {t}", fill=True, alpha=0.4)

plt.title("Probability Distribution of Stock Price at Different Times")
plt.xlabel("Stock Price")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
