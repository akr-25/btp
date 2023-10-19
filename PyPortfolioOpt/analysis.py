from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
import cvxpy as cp

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('../markowitz/*'))))

from markowitz.stocks import stocklist
from markowitz.refactored import prepare_data
from markowitz.refactored import get_returns_given_weights, annualize_returns
import yfinance as yf

# cacher = {
# }

market_vol = 0.182267


# def prepare_data(choosenone, start, end, return_prices=False):

#   start = pd.to_datetime(start)
#   end = pd.to_datetime(end)

#   start = pd.to_datetime(start, format='%Y-%m-%d')
#   end = pd.to_datetime(end, format='%Y-%m-%d')

#   key = str(choosenone) + str(start) + str(end)
#   if key in cacher:
#     df = cacher[key]

#   else:
#     df = pd.DataFrame()
#     for stock in choosenone:
#       data = yf.download(stock + '.NS', start=start, end=end)
#       df[stock] = data['Adj Close']
#     cacher[key] = df

#   if return_prices:
#     return df
#   else:
#     return np.log(df / df.shift(1)).dropna()

def sharpe_ratio(returns, risk, risk_free_rate = 0.06):
    return (returns - risk_free_rate) / risk

def get_all(choosenone):

  df = pd.DataFrame(columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])

  start_train = '01/01/2021'
  end_train = '30/06/2021'

  start_test = '30/06/2021'
  end_test = '31/12/2021'

  df = prepare_data(choosenone, start_train, end_train, return_prices=True)

  from pypfopt.expected_returns import mean_historical_return
  from pypfopt.risk_models import CovarianceShrinkage

  mu = mean_historical_return(df, False)
  S = CovarianceShrinkage(df).ledoit_wolf()


  def get_returns_given_w(weights, choosenone, start, end):
    returns_df = prepare_data(choosenone, start, end)
    aligned_weights = [weights[stock] for stock in returns_df.columns]
    portfolio_return = 252*(returns_df.mean() @ aligned_weights)
    cov_matrix = returns_df.cov()
    portfolio_stddev = np.sqrt(np.array(aligned_weights).T @ cov_matrix @ np.array(aligned_weights)*252)
    return portfolio_return, portfolio_stddev

  print('****************************MAX SHARPE RATIO ANALYSIS****************************')

  ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
  weights = ef.max_sharpe()
  # prrr = pd.DataFrame()
  # for stock in choosenone:
  #   prrr[stock] = weights[stock]
  # prrr.to_csv('weights.csv', index=False)
  returns, risk = get_returns_given_w(weights, choosenone, start_train, end_train)
  print('****************************Time period: ', start_train, ' - ', end_train, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_train, end_train, sharpe_ratio(returns, risk), returns, risk, 'max_sharpe']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  returns, risk = get_returns_given_w(weights, choosenone, start_test, end_test)
  print('****************************Time period: ', start_test, ' - ', end_test, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_test, end_test, sharpe_ratio(returns, risk), returns, risk, 'max_sharpe']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  print('****************************MIN VOLATILITY ANALYSIS****************************')

  ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
  weights = ef.min_volatility()

  returns, risk = get_returns_given_w(weights, choosenone, start_train, end_train)
  print('****************************Time period: ', start_train, ' - ', end_train, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_train, end_train, sharpe_ratio(returns, risk), returns, risk, 'min_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])


  returns, risk = get_returns_given_w(weights, choosenone, start_test, end_test)
  print('****************************Time period: ', start_test, ' - ', end_test, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_test, end_test, sharpe_ratio(returns, risk), returns, risk, 'min_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  print('****************************MARKET VOLATILITY ANALYSIS****************************')

  ef = EfficientFrontier(mu, S, weight_bounds=(-1,1), solver=cp.ECOS)
  weights = ef.efficient_risk(market_vol)

  returns, risk = get_returns_given_w(weights, choosenone, start_train, end_train)
  print('****************************Time period: ', start_train, ' - ', end_train, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_train, end_train, sharpe_ratio(returns, risk), returns, risk, 'market_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  returns, risk = get_returns_given_w(weights, choosenone, start_test, end_test)
  print('****************************Time period: ', start_test, ' - ', end_test, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start_test, end_test, sharpe_ratio(returns, risk), returns, risk, 'market_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])
  return df


merged_df = pd.DataFrame(columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method', 'stockname'])
# get_all(stocklist['esg_stocks2'])
for stockname in stocklist.keys():
  print('****************************', stockname, '****************************')
  df = get_all(stocklist[stockname])
  # df.to_csv('results/' + stockname + '.csv', index=False)
  # df['stockname'] = stockname
  # merged_df = pd.concat([merged_df, df])

# merged_df.to_csv('results/merged.csv', index=False)




