import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simulation - Portfolio Analysis", layout="wide")

import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import scipy.optimize as sco
import cvxpy as cp

st.title('Simulation')

data = pd.read_csv('../assets.csv')

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

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt

# Random seed for reproducibility
np.random.seed(42)



num_portfolios = st.sidebar.slider('select number of portfolios:', min_value=10, max_value=10000000)  # Reduce number of random portfolios for better visualization
risk = st.sidebar.number_input('adjust risk choice: ', min_value=0, max_value= 100)
risk_free_rate = risk / 100

# Lists to store portfolio metrics
port_returns = []
port_volatilities = []
sharpe_ratios = []
portfolio_weights = []

# Generate random portfolios
for _ in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(assets)), size=1)[0]  # Random weights summing to 1
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility

    port_returns.append(port_return)
    port_volatilities.append(port_volatility)
    sharpe_ratios.append(sharpe_ratio)
    portfolio_weights.append(weights)

# Convert to DataFrame
portf_results_df = pd.DataFrame({
    'returns': port_returns,
    'volatility': port_volatilities,
    'sharpe_ratio': sharpe_ratios
})

# Efficient Frontier Calculation
def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    efficient_portfolios = []
    n_assets = len(avg_rtns)
    args = (avg_rtns, cov_mat)
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.ones(n_assets) / n_assets

    for ret in rtns_range:
        constr = (
            {"type": "eq", "fun": lambda x: get_portf_rtn(x, avg_rtns) - ret},
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        )
        ef_portf = sco.minimize(get_portf_vol, initial_guess, args=args,
                                method="SLSQP", constraints=constr, bounds=bounds)
        efficient_portfolios.append(ef_portf)

    return efficient_portfolios

# Generate Efficient Frontier
rtns_range = np.linspace(mean_returns.min(), mean_returns.max(), 100)
efficient_portfolios = get_efficient_frontier(mean_returns, cov_matrix, rtns_range)
vols_range = [x["fun"] for x in efficient_portfolios]

# Plot Efficient Frontier with Color Gradient
fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap-like effect using a scatter plot of random portfolios
sc = ax.scatter(portf_results_df["volatility"], portf_results_df["returns"],
                c=portf_results_df["sharpe_ratio"], cmap="RdYlGn", edgecolors="black", alpha=0.5)

# Colorbar to indicate Sharpe Ratio
plt.colorbar(sc, label="Sharpe Ratio")

# Plot the efficient frontier in blue
ax.plot(vols_range, rtns_range, "b--", linewidth=3, label="Efficient Frontier")

# Plot individual assets as different markers
for stock in assets:
    ax.scatter(np.sqrt(cov_matrix.loc[stock, stock]), mean_returns[stock], marker="o", s=200, label=stock)

# Labels and Title
ax.set_xlabel("Volatility (Risk)")
ax.set_ylabel("Expected Returns")
ax.set_title("Efficient Frontier")
ax.legend()
plt.grid(True)

# Show the final plot
st.pyplot(plt)
