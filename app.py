import pandas as pd
import streamlit as st
st.set_page_config(page_title="Financial Analytics", layout="wide")
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import scipy.optimize as sco
import cvxpy as cp

data = pd.read_csv('assets.csv')

assets1 = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "V",
          "JNJ", "WMT", "PG", "DIS", "MA", "NFLX", "XOM", "PFE", "KO", "PEP"]

assets = st.sidebar.multiselect('Choose the Ticks:', options = assets1, default=assets1)


st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Q7.py", label="Q7 - Portfolio Analysis", icon="❓")
st.sidebar.page_link("pages/Q7.py", label="Simulation - Portfolio Analysis", icon="📈")

st.title("Welcome to Financial Analytics App")
st.write("Select a page from the sidebar to explore.")

st.title('Home')
st.header('Group 4')
st.write('***')

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
cov_matrix = returns.cov()  # Covariance matrix of returns

ew_portfolio_return = np.dot(mean_returns, ew_weights)  # Portfolio expected return
ew_portfolio_volatility = np.sqrt(np.dot(ew_weights.T, np.dot(cov_matrix, ew_weights)))

p_weights = n * [1/n]
portfolio_returns = pd.Series(
np.dot(p_weights, returns.T),
index=returns.index
)

st.subheader('Data Description')
st.write(data.describe())

st.write('***')

st.header('1/n portfolio')

# Compute cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()

# Compute drawdowns
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns / running_max) - 1

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

# Cumulative Return Plot
axes[0].plot(cumulative_returns, color='blue')
axes[0].set_title("1/n Portfolio's Performance")
axes[0].set_ylabel("Cumulative Return")

# Drawdown Plot
axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.5)
axes[1].set_ylabel("Drawdown")

# Daily Return Plot
axes[2].plot(portfolio_returns.index, portfolio_returns, color='gold', linestyle="-", linewidth=0.8)
axes[2].set_ylabel("Daily Return")

plt.tight_layout()

st.pyplot(fig)

st.write('***')

####################################### Formulations

forms = ['Scipy', 'CVXPY']

choice = st.selectbox('Select Formulation', forms)

#################################################### scipy formulation #####################################################

if choice == 'Scipy':

    st.header('Scipy Formulation')

    # Function to compute portfolio performance
    def portfolio_performance(weights, mean_returns, cov_matrix):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return, port_volatility

    # Minimize volatility for a given return target
    def min_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(assets)))

    # Generate efficient frontier by varying target returns
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    efficient_frontier_scipy = []

    for target in target_returns:
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}]  # Target return constraint

        result = sco.minimize(min_volatility, np.ones(len(assets)) / len(assets), method='SLSQP', bounds=bounds, constraints=cons)
        efficient_frontier_scipy.append((target, portfolio_performance(result.x, mean_returns, cov_matrix)[1]))

    efficient_frontier_scipy = np.array(efficient_frontier_scipy)

    plt.figure(figsize=(10, 6))
    plt.plot(efficient_frontier_scipy[:, 1], efficient_frontier_scipy[:, 0], color="blue", linestyle="dashed", label="Efficient Frontier (SciPy)")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (SciPy)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    ################## weighted with stocks
    st.subheader('Optimized with weights')

    n_assets = len(assets)
    # Portfolio optimization
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Generate Efficient Frontier
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    efficient_frontier = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum of weights = 1
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target}  # Target return
        )

        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
        result = sco.minimize(portfolio_volatility, np.ones(n_assets) / n_assets, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            efficient_frontier.append((result.fun, target))  # Store (volatility, return)

    efficient_frontier = np.array(efficient_frontier)

    # Plot the Efficient Frontier
    plt.figure(figsize=(12, 8))

    # Plot the efficient frontier
    plt.plot(efficient_frontier[:, 0], efficient_frontier[:, 1], linestyle="dashed", color="blue", label="Efficient Frontier")

    # Plot individual assets
    for stock in assets:
        plt.scatter(np.sqrt(cov_matrix.loc[stock, stock]), mean_returns[stock], marker="o", s=100, label=stock)

    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with Optimized Weights")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Frontier with max/min return')

    individual_volatility = np.sqrt(np.diag(cov_matrix))  # Standard deviation of each asset
    individual_returns = mean_returns

    # 1. Minimum Variance Portfolio:
    min_volatility_index = np.argmin(efficient_frontier_scipy[:, 1])  # Volatility is in the second column
    min_volatility_portfolio = efficient_frontier_scipy[min_volatility_index]

    # 2. Maximum Return Portfolio:
    max_return_index = np.argmax(efficient_frontier_scipy[:, 0])  # Return is in the first column
    max_return_portfolio = efficient_frontier_scipy[max_return_index]

    # 3. Minimum Return Portfolio:
    min_return_index = np.argmin(efficient_frontier_scipy[:, 0])  # Return is in the first column
    min_return_portfolio = efficient_frontier_scipy[min_return_index]

    # 4. Maximum Sharpe Ratio Portfolio:
    risk_free_rate = 0.0  # Assuming a risk-free rate of 0
    sharpe_ratios = (efficient_frontier_scipy[:, 0] - risk_free_rate) / efficient_frontier_scipy[:, 1]
    max_sharpe_index = np.argmax(sharpe_ratios)
    max_sharpe_portfolio = efficient_frontier_scipy[max_sharpe_index]

    # Plot Efficient Frontier
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_frontier_scipy[:, 1], efficient_frontier_scipy[:, 0], linestyle="dashed", color="blue", label="Efficient Frontier (SciPy)")

    # Plot individual assets
    plt.scatter(individual_volatility, individual_returns, marker="o", s=100, color="red", label="Individual Assets")
    for i, stock in enumerate(assets):
        plt.annotate(stock, (individual_volatility[i], individual_returns[i]), fontsize=9)

    # Highlight special portfolios
    plt.scatter(min_volatility_portfolio[1], min_volatility_portfolio[0], marker="*", s=500, color="gold", label="Min Variance")  # Note: x and y are swapped
    plt.text(min_volatility_portfolio[1] + 0.001, min_volatility_portfolio[0], "Min Variance", fontsize=10, color="black", zorder=6)  # Note: x and y are swapped
    plt.scatter(max_return_portfolio[1], max_return_portfolio[0], marker="*", s=200, color="darkgreen", label="Max Return")  # Note: x and y are swapped
    plt.scatter(min_return_portfolio[1], min_return_portfolio[0], marker="*", s=200, color="darkred", label="Min Return")  # Note: x and y are swapped
    plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], marker="*", s=200, color="darkblue", label="Max Sharpe Ratio")  # Note: x and y are swapped


    # Labels & title
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with SciPy Optimization")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write('***')

##################################################### CVXPY Formulation ##########################################################

else:
    st.header('CVXPY Formulation')

    w = cp.Variable(len(assets))
    gamma = cp.Parameter(nonneg=True)

    # Objective function (max return - risk penalty)
    objective = cp.Maximize(mean_returns.values @ w - gamma * cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= 0]

    # Solve for different gamma values
    problem = cp.Problem(objective, constraints)
    gamma_values = np.logspace(-2, 3, 50)
    efficient_frontier_cvxpy = []

    for g in gamma_values:
        gamma.value = g
        problem.solve()
        efficient_frontier_cvxpy.append((np.sqrt(w.value.T @ cov_matrix.values @ w.value), mean_returns.values @ w.value))

    efficient_frontier_cvxpy = np.array(efficient_frontier_cvxpy)

    plt.figure(figsize=(10, 6))
    plt.plot(efficient_frontier_cvxpy[:, 0], efficient_frontier_cvxpy[:, 1], color="green", linestyle="solid", label="Efficient Frontier (CVXPY)")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (CVXPY)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('CVXPY weighted')

    # Compute risk (volatility) and return for each individual asset
    individual_volatility = np.sqrt(np.diag(cov_matrix))  # Standard deviation of each asset
    individual_returns = mean_returns  # Mean return of each asset

    # Plot Efficient Frontier
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_frontier_cvxpy[:, 0], efficient_frontier_cvxpy[:, 1], linestyle="dashed", color="blue", label="Efficient Frontier (CVXPY)")

    # Plot individual assets
    plt.scatter(individual_volatility, individual_returns, marker="o", s=100, color="red", label="Individual Assets")
    for i, stock in enumerate(assets):
        plt.annotate(stock, (individual_volatility[i], individual_returns[i]), fontsize=9)

    # Labels & title
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with CVXPY Optimization")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Weights Allocation Per Risk-Aversion Level')
    n_assets = len(assets)
    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    # Calculate portfolio returns from scratch
    portf_rtn_cvx = cp.sum(cp.multiply(mean_returns.values, weights))
    portf_vol_cvx = cp.quad_form(weights, cov_matrix)
    objective_function = cp.Maximize(portf_rtn_cvx - gamma_par*portf_vol_cvx)
    problem = cp.Problem(
    objective_function,
    [cp.sum(weights) == 1, weights >= 0]
    )

    N_POINTS = 25
    portf_rtn_cvx_ef = []
    portf_vol_cvx_ef = []
    weights_ef = []
    gamma_range = np.logspace(-3, 3, num=N_POINTS)
    for gamma in gamma_range:
        gamma_par.value = gamma
        problem.solve()
        portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
        portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
        weights_ef.append(weights.value.copy())

    weights_df = pd.DataFrame(weights_ef,
        columns=assets,
        index=np.round(gamma_range, 3))

    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes
    weights_df.plot(kind="bar", stacked=True, ax=ax)  # Pass `ax` to the plot

    # Customize the plot
    ax.set(title="Weights Allocation per Risk-Aversion Level",
        xlabel=r"$\gamma$",
        ylabel="Weight")
    ax.legend(bbox_to_anchor=(1, 1))

    st.pyplot(fig)

    st.subheader('Frontier with max/min return')



    # 1. Minimum Variance Portfolio:
    min_volatility_index = np.argmin(efficient_frontier_cvxpy[:, 0])
    min_volatility_portfolio = efficient_frontier_cvxpy[min_volatility_index]

    # 2. Maximum Return Portfolio:
    max_return_index = np.argmax(efficient_frontier_cvxpy[:, 1])
    max_return_portfolio = efficient_frontier_cvxpy[max_return_index]

    # 3. Minimum Return Portfolio:
    min_return_index = np.argmin(efficient_frontier_cvxpy[:, 1])
    min_return_portfolio = efficient_frontier_cvxpy[min_return_index]

    # 4. Maximum Sharpe Ratio Portfolio:
    risk_free_rate = 0.0  # Assuming a risk-free rate of 0
    sharpe_ratios = (efficient_frontier_cvxpy[:, 1] - risk_free_rate) / efficient_frontier_cvxpy[:, 0]
    max_sharpe_index = np.argmax(sharpe_ratios)
    max_sharpe_portfolio = efficient_frontier_cvxpy[max_sharpe_index]

    # Plot Efficient Frontier
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_frontier_cvxpy[:, 0], efficient_frontier_cvxpy[:, 1], linestyle="dashed", color="blue", label="Efficient Frontier (CVXPY)")

    # Plot individual assets
    plt.scatter(individual_volatility, individual_returns, marker="o", s=100, color="red", label="Individual Assets")
    for i, stock in enumerate(assets):
        plt.annotate(stock, (individual_volatility[i], individual_returns[i]), fontsize=9)

    # Highlight special portfolios
    plt.scatter(min_volatility_portfolio[0], min_volatility_portfolio[1], marker="*", s=500, color="gold", label="Min Variance")
    plt.text(min_volatility_portfolio[0] + 0.001, min_volatility_portfolio[1],
            "Min Variance", fontsize=10, color="black", zorder=6)
    plt.scatter(max_return_portfolio[0], max_return_portfolio[1], marker="*", s=200, color="darkgreen", label="Max Return")
    plt.scatter(min_return_portfolio[0], min_return_portfolio[1], marker="*", s=200, color="darkred", label="Min Return")
    plt.scatter(max_sharpe_portfolio[0], max_sharpe_portfolio[1], marker="*", s=200, color="darkblue", label="Max Sharpe Ratio")

    # Labels & title
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with CVXPY Optimization")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write('***')

############################################################ Portfolio weights df ######################################################

def get_portfolio_weights(efficient_frontier, index, choice):
    """Helper function to retrieve portfolio weights from efficient frontier."""
    if choice == 'Scipy':
        # For SciPy, weights are stored directly in the result object
        target_return = efficient_frontier[index, 0]
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(len(assets)))
        result = sco.minimize(portfolio_volatility, np.ones(len(assets)) / len(assets), method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
    elif choice == 'CVXPY':
        # For CVXPY, re-create the problem with the specific gamma value
        gamma_param = cp.Parameter(nonneg=True)  # Create a new parameter object
        objective = cp.Maximize(mean_returns.values @ w - gamma_param * cp.quad_form(w, cov_matrix.values))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)

        gamma_param.value = gamma_values[index]  # Set gamma value
        problem.solve()
        weights = w.value
    else:
        weights = np.nan * np.ones(len(assets))
    return weights


if choice == 'Scipy':

    # Get weights for special portfolios (SciPy)
    min_var_weights_scipy = get_portfolio_weights(efficient_frontier_scipy, np.argmin(efficient_frontier_scipy[:, 1]), choice)
    max_ret_weights_scipy = get_portfolio_weights(efficient_frontier_scipy, np.argmax(efficient_frontier_scipy[:, 0]), choice)
    min_ret_weights_scipy = get_portfolio_weights(efficient_frontier_scipy, np.argmin(efficient_frontier_scipy[:, 0]), choice)

    # Since Sharpe ratio calculation involves both risk and return,
    # we need to recalculate it to find the exact index on the efficient frontier
    risk_free_rate = 0.0
    sharpe_ratios_scipy = (efficient_frontier_scipy[:, 0] - risk_free_rate) / efficient_frontier_scipy[:, 1]
    max_sharpe_weights_scipy = get_portfolio_weights(efficient_frontier_scipy, np.argmax(sharpe_ratios_scipy), choice)

    scipy_df = pd.DataFrame({
    'Stock': assets,
    'Min Variance': min_var_weights_scipy,
    'Max Return': max_ret_weights_scipy,
    'Min Return': min_ret_weights_scipy,
    'Max Sharpe': max_sharpe_weights_scipy
    }).reset_index(drop=True)  # Reset index for SciPy table
    
    st.subheader('Portfolio Weights:')
    st.dataframe(scipy_df)

else:
    # Get weights for special portfolios (CVXPY)
    min_var_weights_cvxpy = get_portfolio_weights(efficient_frontier_cvxpy, np.argmin(efficient_frontier_cvxpy[:, 0]), choice)
    max_ret_weights_cvxpy = get_portfolio_weights(efficient_frontier_cvxpy, np.argmax(efficient_frontier_cvxpy[:, 1]), choice)
    min_ret_weights_cvxpy = get_portfolio_weights(efficient_frontier_cvxpy, np.argmin(efficient_frontier_cvxpy[:, 1]), choice)
    sharpe_ratios_cvxpy = (efficient_frontier_cvxpy[:, 1] - risk_free_rate) / efficient_frontier_cvxpy[:, 0]
    max_sharpe_weights_cvxpy = get_portfolio_weights(efficient_frontier_cvxpy, np.argmax(sharpe_ratios_cvxpy), choice)

    cvxpy_df = pd.DataFrame({
    'Stock': assets,
    'Min Variance': min_var_weights_cvxpy,
    'Max Return': max_ret_weights_cvxpy,
    'Min Return': min_ret_weights_cvxpy,
    'Max Sharpe': max_sharpe_weights_cvxpy
    }).reset_index(drop=True)  # Reset index for CVXPY table

    st.subheader('Portfolio Weights:')
    st.dataframe(cvxpy_df)

st.write('***')
