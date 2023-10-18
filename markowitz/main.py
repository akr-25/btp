import pandas as pd
import numpy as np
from scipy.optimize import minimize
from stocks import stocks as stocklist
import matplotlib.pyplot as plt

def get_data(stocks, start_timestamp=None, end_timestamp=None):
    data = {}
    for stock in stocks:
        stock = stock.replace('.NS', '')
        stock = stock.replace('.BO', '')
        try:
            df = pd.read_json('data/{}.json'.format(stock))
            if start_timestamp and end_timestamp:
                df = df[(df['t'] >= start_timestamp) & (df['t'] <= end_timestamp)]
            data[stock] = df
        except:
            print('Error in reading data for {}'.format(stock))
    return data

def expected_returns_and_covariance(data):
    prices = []
    for stock, df in data.items():
        prices.append(df['c'])
    prices_df = pd.concat(prices, axis=1, keys=data.keys())
    returns = prices_df.pct_change()
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    return expected_returns, cov_matrix

def objective(weights, expected_returns, cov_matrix, rf_rate):
    portfolio_return = np.dot(expected_returns, weights)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - rf_rate) / portfolio_stddev

def markowitz_optimization(expected_returns, cov_matrix, rf_rate=0.01):
    num_stocks = len(expected_returns)
    args = (expected_returns, cov_matrix, rf_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for stock in range(num_stocks))
    result = minimize(objective, [1./num_stocks]*num_stocks, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def minimum_variance_portfolio(expected_returns, cov_matrix):
    num_stocks = len(expected_returns)
    args = (expected_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # bounds = tuple((0, 1) for stock in range(num_stocks))
    bounds = None
    result = minimize(objective, [1./num_stocks]*num_stocks, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Efficient Frontier
def efficient_frontier(expected_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((4, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        portfolio_return = np.dot(expected_returns, weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i], results[1,i] = portfolio_return, portfolio_stddev
        results[2,i] = (results[0,i]) / results[1,i]  # Sharpe ratio
        results[3,i] = weights.argmax()
    return results

start_timestamp = 1534204800  
end_timestamp = 1691971200    

data = get_data(stocklist, start_timestamp, end_timestamp)
stocks_available = list(data.keys())
expected_returns, cov_matrix = expected_returns_and_covariance(data)

# Calculate optimized weights and portfolio statistics
result = markowitz_optimization(expected_returns, cov_matrix)
optimized_weights = pd.DataFrame(result.x, index=stocks_available, columns=['Optimized Weights'])

# Generate efficient frontier data
results = efficient_frontier(expected_returns, cov_matrix)

plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier with Color-coded Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
# plt.show()

# Print data
print("Optimized Weights:\n", optimized_weights)
print("\nExpected Returns:\n", expected_returns)
print("\nPortfolio Risks (Standard Deviations):\n", np.sqrt(np.diag(cov_matrix)))


# Portfolio returns and risks
portfolio_return = np.dot(expected_returns, optimized_weights)
portfolio_stddev = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))

print("\nPortfolio Return: ", portfolio_return)
print("\nPortfolio Risk (Standard Deviation): ", portfolio_stddev)
print("\nAnnualized return: ", portfolio_return * 252)