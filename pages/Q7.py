import pandas as pd
import streamlit as st

st.set_page_config(page_title="Q7 - Portfolio Analysis", layout="wide")

import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import scipy.optimize as sco
import cvxpy as cp

st.title('Q7')

data = pd.read_csv('C:\\Users\\user\\Desktop\\stream_fin\\assets.csv')

assets1 = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "V",
          "JNJ", "WMT", "PG", "DIS", "MA", "NFLX", "XOM", "PFE", "KO", "PEP"]

assets = st.sidebar.multiselect('Choose the Ticks:', options = assets1, default=assets1)

assets2 = list(assets)  # Convert tuple to list if necessary

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data = data[assets2]

st.header('Data')

st.dataframe(data)

returns = np.log(data / data.shift(1)).dropna()

# Number of assets
n = len(assets)

# Equal-weight allocation (1/n)
ew_weights = np.ones(n) / n  # Each stock gets an equal share

# Compute expected return and risk of equal-weighted portfolio
mean_returns = returns.mean()  # Average return of each stock
cov_matrix = returns.cov()

st.subheader('Asset Allocation Per Risk-Aversion & Leverage')

# Function to compute portfolio performance (expected return & volatility)
def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

# Function to minimize volatility for given target return (used in SciPy optimization)
def min_volatility(weights):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# Define leverage levels
leverage_levels = [1, 2,3,4,5 ]  # Different leverage levels

# Store efficient frontiers per leverage level
efficient_frontiers = {}

for leverage in leverage_levels:
    constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},

    {"type": "ineq", "fun": lambda w: leverage - np.sum(np.abs(w))}  # Max leverage constraint
    ]

    bounds = tuple((-leverage, leverage) for _ in range(len(assets)))  # Allow short selling

    target_returns = np.linspace(mean_returns.min(), mean_returns.max() * leverage, 50)
    efficient_frontier = []

    for target in target_returns:
        cons = constraints + [{'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}]


        result = sco.minimize(
            min_volatility, np.ones(len(assets)) / len(assets),
            method='SLSQP', bounds=bounds, constraints=cons
        )
        if result.success:
            efficient_frontier.append((target, portfolio_performance(result.x, mean_returns, cov_matrix)[1]))

    efficient_frontiers[leverage] = np.array(efficient_frontier)

# Plot the efficient frontiers for different leverage levels
plt.figure(figsize=(10, 6))

for leverage, frontier in efficient_frontiers.items():
    plt.plot(frontier[:, 1], frontier[:, 0], label=f"Leverage {leverage}")

plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier for Different Leverage Levels")
plt.legend()
plt.grid(True)
st.pyplot(plt)

st.subheader('Efficient Frontier: Minimized Volatility vs. Maximized Risk-Adjusted Return')

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

# Function to maximize Sharpe Ratio
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    port_return, port_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (port_return - risk_free_rate) / port_volatility  # Negative Sharpe ratio for minimization

# Generate efficient frontiers
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
vols_range, sharpe_returns, sharpe_vols_range = [], [], []

# Min Volatility Efficient Frontier
for target in target_returns:
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}]

    result = sco.minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
                          np.ones(len(assets)) / len(assets),
                          method='SLSQP', bounds=[(0, 1)] * len(assets), constraints=constraints)

    if result.success:
        vols_range.append(result.fun)

# Maximized Sharpe Ratio Efficient Frontier (Scaling for a full curve)
best_sharpe_portfolio = sco.minimize(neg_sharpe_ratio, np.ones(len(assets)) / len(assets),
                                     args=(mean_returns, cov_matrix), method='SLSQP',
                                     bounds=[(0, 1)] * len(assets), constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])

if best_sharpe_portfolio.success:
    opt_weights = best_sharpe_portfolio.x
    base_return, base_vol = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    # Scale the portfolio return and volatility along different leverage levels
    leverage_factors = np.linspace(0.5, 2.5, 50)  # Scaling from 50% leverage to 250%
    for factor in leverage_factors:
        sharpe_returns.append(base_return * factor)
        sharpe_vols_range.append(base_vol * factor)

# Plot Side-by-Side Efficient Frontiers
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(vols_range, target_returns, "g-", linewidth=3)
ax[0].set(title="Efficient Frontier - Minimized Volatility", xlabel="Volatility", ylabel="Expected Returns")

ax[1].plot(sharpe_vols_range, sharpe_returns, "g-", linewidth=3)
ax[1].set(title="Efficient Frontier - Maximized Risk-Adjusted Return", xlabel="Volatility", ylabel="Expected Returns")

plt.tight_layout()
st.pyplot(plt)

st.subheader('Weight Allocation across Leverages')

# Define leverage levels
LEVERAGE_RANGE = [1, 2, 5]  # Max leverage allowed
N_POINTS = 25
gamma_range = np.logspace(-3, 3, num=N_POINTS)

# Store results
weights_ef = np.zeros((len(LEVERAGE_RANGE), N_POINTS, len(assets)))

# CVXPY Setup
weights = cp.Variable(len(assets))
gamma_par = cp.Parameter(nonneg=True)
max_leverage = cp.Parameter()

# Define portfolio return & volatility
portf_rtn_cvx = cp.sum(cp.multiply(mean_returns.values, weights))
portf_vol_cvx = cp.quad_form(weights, cov_matrix)
objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)

# Solve for different leverage levels
for lev_idx, leverage in enumerate(LEVERAGE_RANGE):
    for gamma_idx, gamma in enumerate(gamma_range):
        gamma_par.value = gamma
        max_leverage.value = leverage  # Set leverage constraint

        problem = cp.Problem(
            objective_function,
            [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage]  # Leverage constraint
        )
        problem.solve()

        # Store weights for each leverage level
        weights_ef[lev_idx, gamma_idx, :] = weights.value

# Plot Stacked Bar Charts for Different Leverage Levels
fig, ax = plt.subplots(len(LEVERAGE_RANGE), 1, figsize=(12, 8), sharex=True)

for lev_idx, leverage in enumerate(LEVERAGE_RANGE):
    weights_df = pd.DataFrame(weights_ef[lev_idx], columns=assets, index=np.round(gamma_range, 3))
    weights_df.plot(kind="bar", stacked=True, ax=ax[lev_idx], legend=(lev_idx == 0))
    ax[lev_idx].set_ylabel(f"Leverage {leverage}")

ax[-1].set_xlabel(r"$\gamma$")  # Only set x-axis label on the last subplot
fig.suptitle("Weight Allocation Across Risk-Aversion Levels with Leverage")
st.pyplot(fig)