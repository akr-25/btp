import pandas as pd
import numpy as np
from .stocks import stocklist
from scipy.optimize import minimize

# Constants
DATA_PATH = 'C://Users//aman2//Documents//BTProject//btp2.0//data//{}.csv'
DATE_FORMAT = '%d/%m/%Y'

def get_stock_data(stock, start_timestamp=None, end_timestamp=None):
    try:
        stock = stock.replace('.NS', '').replace('.BO', '')
        df = pd.read_csv(DATA_PATH.format(stock))

        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = (df['Date'].astype('int64')//1e9) + 19800

        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        if start_timestamp and end_timestamp:
            df = df[(df['Date'] >= start_timestamp) & (df['Date'] <= end_timestamp)]
        df = df.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        raise ValueError(f"Error in reading data for {stock}. Error: {e}")

def get_data_for_stocks(stocks, start_timestamp=None, end_timestamp=None):
    data = {}
    errors = []
    for stock in stocks:
        try:
            data[stock] = get_stock_data(stock, start_timestamp, end_timestamp)
        except ValueError as ve:
            errors.append(str(ve))
    return data, errors

def prepare_data(stocklist, start, end, return_prices=False):
    start_timestamp = pd.to_datetime(start, format=DATE_FORMAT)
    end_timestamp = pd.to_datetime(end, format=DATE_FORMAT)
    data, errors = get_data_for_stocks(stocklist, start_timestamp, end_timestamp)
    if errors:
        for error in errors:
            print(error)
    prices = [df['Close'] for stock, df in data.items()]
    prices_df = pd.concat(prices, axis=1, keys=data.keys())
    prices_df = prices_df.dropna(axis=1)
    if return_prices:
        return prices_df
    # prices_df = prices_df.fillna(method='bfill').dropna(axis=1)
    return np.log(prices_df / prices_df.shift(1)).dropna()

def calculate_portfolio_weights(returns_df):
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    u = np.ones(len(cov_matrix))
    cov_inv = np.linalg.inv(cov_matrix)
    W_mvp = cov_inv @ u / (u.T @ cov_inv @ u)
    return pd.Series(W_mvp, index=cov_matrix.index)

def portfolio_risk_for_return(cov_matrix, mean_returns, mu): 
    u = np.ones(len(cov_matrix))
    cov_inv = np.linalg.inv(cov_matrix)
    M = np.array([
        [mean_returns @ cov_inv @ mean_returns.T, u @ cov_inv @ mean_returns.T],
        [mean_returns @ cov_inv @ u.T, u @ cov_inv @ u.T]
    ])
    M_inv = np.linalg.inv(M)
    lambda12 = 2 * M_inv @ np.array([mu, 1])
    W_tan = (lambda12[0] * mean_returns @ cov_inv + lambda12[1] * u @ cov_inv)/2
    W_tan = pd.Series(W_tan, index=cov_matrix.index)
    return W_tan

def get_input_for_output(cov_matrix, mean_returns, output, input_range):
    TOL = 1e-6
    MAX_ITER = 1000
    iter_count = 0

    low, high = input_range
    while iter_count < MAX_ITER:
        mid = (low + high) / 2
        W_tan = portfolio_risk_for_return(cov_matrix, mean_returns, mid)
        risk = W_tan.T @ cov_matrix @ W_tan
        if abs(risk - output) < TOL: 
            return mid
        elif risk > output:
            high = mid
        else:
            low = mid + 0.000000001
        iter_count += 1
    return -1

def get_returns(stocklist):
    returns_df = prepare_data(stocklist, '01/01/2016', '01/01/2020')
    W_mvp = calculate_portfolio_weights(returns_df)

    cov_matrix = returns_df.cov()
    mean_returns = returns_df.mean()

    x = get_input_for_output(cov_matrix, mean_returns, 0.008118**2, (0, 20))
    W_tan = portfolio_risk_for_return(cov_matrix, mean_returns, x)

    returns = returns_df.mean() @ W_tan
    volatility = np.sqrt(252 * W_tan.T @ cov_matrix @ W_tan)

    print('Stocks with issues -  ',  set(stocklist) - set(W_tan.index))

    print('Returns for 2016-2020:', (np.exp(returns*252)-1)*100, '%')
    print('Volatility for 2016-2020:', volatility)

    returns_df2 = prepare_data(W_tan.index, '01/01/2020', '30/5/2023')
    mean_returns2 = returns_df2.mean()
    # print(mean_returns2)
    portfolio_return = mean_returns2 @ W_tan
    volatility = np.sqrt(252 * W_tan.T @ cov_matrix @ W_tan)

    return (np.exp(portfolio_return*252)-1)*100, volatility, W_tan

def annualize_returns(returns):
    return np.exp(returns*252) - 1

def maximize_sharpe_ratio(returns_df, R_f=0.06):
    """
    Calculates the portfolio with the maximum Sharpe ratio.

    Parameters:
    - returns_df: DataFrame with historical returns data.
    - R_f: risk-free rate (default is 2% annually).

    Returns:
    - weights for the portfolio with maximum Sharpe ratio.
    - the Sharpe ratio of this portfolio.
    """
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, R_f)
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((-0.1, 0.1) for asset in range(num_assets))
    result = minimize(lambda weights, mean_returns, cov_matrix, R_f: -(((annualize_returns(mean_returns@weights) - R_f) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))*252))), 
                      [1./num_assets]*num_assets, args=args, bounds=bounds, constraints=constraints)
    
    return result.x, -result.fun

def get_sharpe_optimized_returns(stocklist):
    returns_df = prepare_data(stocklist, '01/01/2016', '01/01/2020')
    weights, sharpe_ratio = maximize_sharpe_ratio(returns_df)
    
    portfolio_return = annualize_returns(returns_df.mean() @ weights)
    portfolio_stddev = np.sqrt(weights.T @ returns_df.cov() @ weights*252)
    
    return portfolio_return, portfolio_stddev, sharpe_ratio, weights

def get_returns_given_weights(weights, stocklist, start_date, end_date):
    """
    Calculates the returns and volatility for a given portfolio (specified by weights)
    during the specified period.

    Parameters:
    - weights: Portfolio weights.
    - stocklist: List of stock tickers.
    - start_date: Start date of the period.
    - end_date: End date of the period.

    Returns:
    - Portfolio returns during the specified period.
    - Portfolio volatility during the specified period.
    """
    returns_df = prepare_data(stocklist, start_date, end_date)
    
    # Ensure the stocks in weights are in the same order as in returns_df
    aligned_weights = [weights[stocklist.index(stock)] for stock in returns_df.columns]

    portfolio_return = annualize_returns(returns_df.mean() @ aligned_weights)
    cov_matrix = returns_df.cov()
    portfolio_stddev = np.sqrt(np.array(aligned_weights).T @ cov_matrix @ np.array(aligned_weights)*252)
    
    return portfolio_return, portfolio_stddev

if __name__ == '__main__':
    print('Choose stocklist:')
    for i, stocklistname in enumerate(stocklist.keys()):
        print(f'{i} - {stocklistname}')
    stocklistname = int(input('Enter stocklist index: '))
    choosenone = stocklist[list(stocklist.keys())[stocklistname]]
    returns, volatility, weights = get_returns(choosenone)
    print('Returns:', returns, '%')
    print('Volatility:', volatility)
    # print('Weights:', weights)

    print('Using Sharpe Ratio Optimization')

    portfolio_return, portfolio_stddev, sharpe_ratio, weights = get_sharpe_optimized_returns(choosenone)
    print('Sharpe-optimized Portfolio Returns:', portfolio_return)
    print('Sharpe-optimized Portfolio Volatility:', portfolio_stddev)
    print('Max Sharpe Ratio:', sharpe_ratio)

    portfolio_return, portfolio_stddev = get_returns_given_weights(weights, choosenone, '01/01/2020', '30/5/2023')
    print('Returns:', portfolio_return, '%')
    print('Volatility:', portfolio_stddev)
