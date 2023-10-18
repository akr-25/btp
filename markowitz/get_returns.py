import pandas as pd
import numpy as np
# from scipy.optimize import minimize
# from stocks import stocks as stocklist
# import matplotlib.pyplot as plt

def get_returns_fun(stocklist):

  def get_data(stocks, start_timestamp=None, end_timestamp=None):
      data = {}
      errors = []
      for stock in stocks:
          stock = stock.replace('.NS', '')
          stock = stock.replace('.BO', '')
          try:
              df = pd.read_json('C://Users//aman2//Documents//BTProject//btp2.0//data//{}.json'.format(stock))
              df['t'] = pd.to_datetime(df['t'], unit='s')
              if start_timestamp and end_timestamp:
                  df = df[(df['t'] >= start_timestamp) & (df['t'] <= end_timestamp)]
              df = df.set_index('t')
              df = df[['o', 'h', 'l', 'c', 'v']]
              df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
              
              data[stock] = df
          except Exception as e:
              print('Error in reading data for {}'.format(stock))
              print(e)
              errors.append(stock)
      return data, errors

  # start_timestamp = '01/01/2018'
  # end_timestamp = '30/6/2023'

  start_timestamp = '01/01/2016'
  end_timestamp = '01/01/2020'

  # convert to datetime64[ns] dtype
  start_timestamp = pd.to_datetime(start_timestamp, dayfirst=True)
  end_timestamp = pd.to_datetime(end_timestamp, dayfirst=True)

  data, errors = get_data(stocklist, start_timestamp, end_timestamp)
  print('Errors: {}'.format(errors) if errors else 'No errors')
  print('Stocks available: {}'.format(list(data.keys())))
  stocks_available = list(data.keys())

  prices = []
  for stock, df in data.items():
      prices.append(df['Close'])
  prices_df = pd.concat(prices, axis=1, keys=data.keys())

  # backfill missing values
  prices_df = prices_df.fillna(method='bfill')

  # prices_df.head()
  # get all columns which have atleast one null value
  null_columns = prices_df.columns[prices_df.isna().any()].tolist()

  # make prices_df drop all the columns which have null values
  prices_df = prices_df.dropna(axis=1)
  # prices_df.head()

  # log returns
  returns_df = np.log(prices_df / prices_df.shift(1))
  returns_df = returns_df.dropna()

  mean_returns = returns_df.mean()
  # mean_returns = mean_returns
  mean_returns.sort_values(ascending=False)

  cov_matrix = returns_df.cov()

  # Markowitz portfolio optimization
  u = np.ones(len(cov_matrix))

  W_mvp = np.linalg.inv(cov_matrix) @ u / (u.T @ np.linalg.inv(cov_matrix) @ u)

  W_mvp = pd.Series(W_mvp, index=cov_matrix.index)

  # Portfolio variance and returns

  portfolio_variance = W_mvp.T @ cov_matrix @ W_mvp
  print('Minimum variance - ' , portfolio_variance)

  portfolio_return = mean_returns.T @ W_mvp
  # print(portfolio_return)
  print('Returns of MVP - ', (np.exp(portfolio_return)-1)*100)


  cov_inv = np.linalg.inv(cov_matrix)

  print('Next two lines should be same')
  print(np.matmul(np.transpose(mean_returns), np.linalg.solve(cov_matrix, mean_returns)))
  print(mean_returns @ cov_inv @ mean_returns.T)

  def portfolio_risk_for_return(mu): 
    cov_inv = np.linalg.inv(cov_matrix)
    M = np.array([[mean_returns @ np.linalg.solve(cov_matrix, mean_returns.T), u @ np.linalg.solve(cov_matrix, mean_returns.T)], [mean_returns @ np.linalg.solve(cov_matrix, u.T), u @ np.linalg.solve(cov_matrix, u.T)]])
    M_inv = np.linalg.inv(M)
    lambda12 = 2 * M_inv @ np.array([mu, 1])

    W_tan = (lambda12[0] * mean_returns @ cov_inv + lambda12[1] * u @ cov_inv)/2

    W_tan = pd.Series(W_tan, index=cov_matrix.index)

    portfolio_rrisk = W_tan.T @ cov_matrix @ W_tan
    return portfolio_rrisk, W_tan

    # get the input at which portfolio_risk_for_return returns specified value using binary search
  def get_input_for_output(output, input_range):
    TOL = 1e-6
    low, high = input_range
    while low < high:
      mid = (low + high) / 2
      risk, _ = portfolio_risk_for_return(mid)
      if abs(risk - output) < TOL: 
        return mid
      elif risk > output:
        high = mid
      else:
        low = mid + 0.000000001
    return -1 

  x = get_input_for_output(0.008118**2, (0, 20))
  # print(x)
  print('Returns of Tangency portfolio (Bias) - ', (np.exp(x*252)-1)*100, '%')
  risk, W_tan = portfolio_risk_for_return(x) 

  # part 2

  start_timestamp = '01/01/2020'
  end_timestamp = '30/6/2023'

  # start_timestamp = '01/01/2010'
  # end_timestamp = '01/01/2018'

  # convert to datetime64[ns] dtype
  start_timestamp = pd.to_datetime(start_timestamp, dayfirst=True)
  end_timestamp = pd.to_datetime(end_timestamp, dayfirst=True)

  data2, errors = get_data(W_tan.index, start_timestamp, end_timestamp)
  print('Errors: {}'.format(errors) if errors else 'No errors')

  prices = []
  for stock, df in data2.items():
      prices.append(df['Close'])
  prices_df = pd.concat(prices, axis=1, keys=data2.keys())

  prices_df = prices_df.fillna(method='bfill')

  print(prices_df.isna().sum().sum()==0)
  prices_df = prices_df.dropna(axis=1)
  # prices_df.head()
  returns_df = np.log(prices_df / prices_df.shift(1))
  returns_df = returns_df.dropna()
  mean_returns2 = returns_df.mean()

  portfolio_return = mean_returns2 @ W_tan
  volatility = np.sqrt(252*np.dot(W_tan.T, np.dot(cov_matrix, W_tan)))
  # print(portfolio_return*252)
  print('Returns of Tangency portfolio - ', (np.exp(portfolio_return*252)-1)*100, '%')
  print('Volatility of Tangency portfolio - ', volatility)
  print(W_tan)

  return (np.exp(portfolio_return*252)-1)*100, volatility ,W_tan


from stocks import esg_stocks2 as esg_stocks

get_returns_fun(esg_stocks)