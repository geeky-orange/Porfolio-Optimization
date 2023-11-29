import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from scipy.optimize import minimize

"""
Best result found from 2016.8.1 to 2020.8.1
Allen's benchmark:
Tickers: ["VONG", "ONEQ", "ARKK"]
Weight: [0.314, 0.336, 0.35]
Return: 0.315
Volatility: 0.233
Sharp Ratio: 1.351

My portfolio:
Tickers: ["GDX","IBUY","ARKK"]
Weight: [0.11, 0.506, 0.384]
Return: 0.334
Volatility: 0.246
Sharp Ratio: 1.366
"""

# https://people.math.ethz.ch/~jteichma/markowitz_portfolio_optimization.html
# returns results of Monte Carlo simulation
def get_results_frame(data, number_of_simulations, sortino_target_return):
    target = 0
    components = list(data.columns.values)
    # convert daily stock prices into daily returns
    returns = data.pct_change()
    returns = returns.dropna()

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    # cov_matrix_sortino = returns
    # set number of runs of random portfolio weights
    num_portfolios = number_of_simulations

    """    
     set up array to hold results
     '3' is to store returns, standard deviation and sharpe ratio
     len(components) is for storing weights for each component in every portfolio
    """

    results = np.zeros((4 + len(components), num_portfolios))

    for i in list(range(num_portfolios)):
        # select random weights for portfolio holdings
        weights = np.random.random(len(components))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)
        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        daily_portfolio_return = returns * weights

        # add this sum col to results
        sum_col = daily_portfolio_return.sum(axis=1)
        tempseries = sum_col[sum_col < target]
        downside_deviation = tempseries.std()
        # print("downside deviation is {}".format(downside_deviation))

        # filter sum_col to get only negative values and use that to calculate semi-deviation/downside risk
        # use downside risk to calculate sortino ratio = portfolio_return / downside_risk

        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # downside_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # downside_returns = df.loc[df['pf_returns'] < target]
        # store results in results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev

        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = results[0, i] / results[1, i]

        # sortino ratio
        results[3, i] = portfolio_return / downside_deviation

        for j in range(len(weights)):
            # change the 3 in the following line to 4 and add 'sortino' to columns
            results[j + 4, i] = weights[j]

    cols = ['ret', 'stdev', 'sharpe', 'sortino']
    for stock in components:
        cols.append(stock)

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=cols)
    # results_frame.to_csv('datacsv.csv')
    return results_frame


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(mean_daily_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1


def check_sum(weights):
    return np.sum(weights) - 1


if __name__ == "__main__":
    init_guess = [1 / 4] * 4
    test_tup = (0, 1)
    bounds = ((test_tup,) * 4)
    daily_data = pd.read_csv(r"Daily_Data_5_Years_Benchmark.csv", index_col="Date", parse_dates=True)
    cons = ({'type': 'eq', 'fun': check_sum})

    # Get all the tickers from columns
    ticker_list = daily_data.columns.values

    # Check the portfolio
    """# temp_df = daily_data.pct_change().mean()

    # returns = temp_df.pct_change()
    # returns = returns.dropna()

    # calculate mean daily return and covariance of daily returns
    # mean_daily_returns = returns.mean()

    # print(temp_df[temp_df > 0.3/252])
    # for i in range(10000):
    # temp_df = daily_data[['DVY Adj Close', 'SOXX Adj Close', 'CIBR Adj Close', 'ARKK Adj Close']]
    # [0.00000000e+00, 6.20219693e-02, 7.37646144e-17, 9.37978031e-01]
    # temp_df = daily_data[['IJR Adj Close', 'ARKK Adj Close', 'KBWB Adj Close', 'SCHC Adj Close']]
    # [2.77555756e-17, 1.00000000e+00, 2.03830008e-17, 0.00000000e+00]
    # temp_df = daily_data[['ARKK Adj Close', 'MGK Adj Close', 'XLF Adj Close', 'ARKG Adj Close']]
    # [1.00000000e+00, 0.00000000e+00, 1.38777878e-17, 0.00000000e+00]
    # temp_df = daily_data[['IJR Adj Close', 'ARKK Adj Close', 'KBWB Adj Close', 'SCHC Adj Close']]
    # [2.77555756e-17, 1.00000000e+00, 2.03830008e-17, 0.00000000e+00]
    # temp_df = daily_data[['XLV Adj Close', 'SKYY Adj Close', 'IBUY Adj Close', 'FTC Adj Close']]
    # [1.10710296e-16, 1.06786801e-01, 8.93213199e-01, 9.32008287e-17]
    # temp_df = daily_data[['VHT Adj Close', 'VFH Adj Close', 'ARKK Adj Close', 'SLV Adj Close']]
    # [0.00000000e+00, 1.18499546e-16, 1.00000000e+00, 7.63278329e-17]"""

    # Use random guess to select the wanted portfolio
    for i in range(100000):
        # Randomly select tickers
        selected_ticker_list = ticker_list[np.random.randint(1, 333, 4)]
        temp_df = daily_data[selected_ticker_list]

        # print(temp_df)

        returns = temp_df.pct_change()
        returns = returns.dropna()

        # calculate mean daily return and covariance of daily returns
        mean_daily_returns = returns.mean()
        cov_matrix = returns.cov()

        # Use the minimize to accelerate the calculation
        opt_results = minimize(neg_sharpe, init_guess, bounds=bounds, method='SLSQP', constraints=cons)
        # print("Weight of ETF", opt_results.x)

        returns, volatility, sharpe_ratio = get_ret_vol_sr(opt_results.x)

        # select the portfolio return rate greater than 30%
        if returns > 0.3:
            flag = 0
            for weight in opt_results.x:
                if weight <= 0.75:  # Avoid too bias portfolio
                    flag += 1
            if flag == 4:
                if sharpe_ratio >= 1.3:
                    print("--------star---------")  # wanted portfolio
                else:
                    print("-------normal--------")
                print(selected_ticker_list)
                print("Return", returns)
                print("Volatility", volatility)
                print("Sharpe Ratio", sharpe_ratio)
                print("Weight", opt_results.x)

