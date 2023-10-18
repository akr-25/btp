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

def sharpe_ratio(returns, risk, risk_free_rate = 0.06):
    return (returns - risk_free_rate) / risk

def get_all(choosenone):

  df = pd.DataFrame(columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])

  start = '01/01/2016'
  end = '01/01/2020'
  df = prepare_data(choosenone, start, end, return_prices=True)

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

  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'max_sharpe']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  start = '01/01/2020'
  end = '30/05/2023'
  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'max_sharpe']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  print('****************************MIN VOLATILITY ANALYSIS****************************')

  ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))
  weights = ef.min_volatility()

  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'min_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  start = '01/01/2016'
  end = '01/01/2020'
  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'min_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  print('****************************MARKET VOLATILITY ANALYSIS****************************')

  ef = EfficientFrontier(mu, S, weight_bounds=(-1,1), solver=cp.ECOS)
  weights = ef.efficient_risk(0.214816)

  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'market_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])

  start = '01/01/2020'
  end = '30/05/2023'
  returns, risk = get_returns_given_w(weights, choosenone, start, end)
  print('****************************Time period: ', start, ' - ', end, '****************************')
  print('Sharpe ratio: ', sharpe_ratio(returns, risk))
  print('Expected returns: ', returns*100, '%')
  print('Risk: ', risk*100, '%')
  df = pd.concat([df, pd.DataFrame([[start, end, sharpe_ratio(returns, risk), returns, risk, 'market_volatility']], columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method'])])
  return df


merged_df = pd.DataFrame(columns=['start', 'end', 'sharpe_ratio', 'returns', 'risk', 'method', 'stockname'])

for stockname in stocklist.keys():
  df = get_all(stocklist[stockname])
  df.to_csv('results/' + stockname + '.csv', index=False)
  df['stockname'] = stockname
  merged_df = pd.concat([merged_df, df])

merged_df.to_csv('results/merged.csv', index=False)




