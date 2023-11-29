import pandas as pd
import numpy as np
from scipy.stats import norm


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    :param r: DataFrame or Series
    :return: float or Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set degree of freedom to be 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    :param r: DataFrame or Series
    :return: float or Series
    """
    delta_mean = r - r.mean()
    # use the population standard deviation, so set degree of freedom to be 0
    sigma_r = r.std(ddof=0)
    exp = (delta_mean ** 4).mean()
    return exp / sigma_r ** 4


def var_historic(r, level=5):
    """
    calculate the var with historic data
    :param r: dataframe or series
    :param level: significance level
    :return: dataframe
    """

    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def var_gaussian(r, level=5, modified=False, time_period=1):
    """

    :param time_period: calculate the time
    :param modified: use cornish-fisher distribution or not
    :param r: DataFrame
    :param level: significance level
    :return: Parametric Gaussian VaR of a Series or DataFrame
    """
    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(r.mean() + z * r.std() * np.sqrt(time_period))


def cvar_historic(r, level=5):
    """

    :param r: DataFrame or Series
    :param level: significance level
    :return: conditional VaR of a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def cvar_gaussian(r, level=5, modified=False, time_period=1):
    """

    :param time_period:
    :param modified:
    :param r: DataFrame or Series
    :param level: significance level
    :return: conditional VaR of a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        annual_r = r.apply(lambda x: (x + 1) ** 252 - 1)
        is_beyond = annual_r <= -var_gaussian(r, level=level, modified=modified, time_period=time_period)
        return -annual_r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def portfolio_return_cov_sharpe(daily_data, weights):
    """
    give the portfolio return
    :param daily_data: all daily data of portfolio
    :param weights: weights of portfolio
    :return: portfolio return, sharpe, volatility
    """
    returns = daily_data.pct_change()
    returns = returns.dropna()
    weights = np.array(weights)
    mean_daily_returns = returns.mean()

    ret = np.sum(mean_daily_returns * weights) * 252
    cov_matrix = returns.cov()
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = ret / vol

    return ret, vol, sharpe


def portfolio_max_drawdown(cum_return):
    """
    get the maximum drawdown of a portfolio
    :param cum_return: the cumulative return of portfolio return
    :return: the max drawdown
    """
    index_j = np.argmax((np.maximum.accumulate(cum_return) - cum_return) / np.maximum.accumulate(cum_return))
    if index_j == 0:
        max_drawdown = 0
    else:
        index_i = np.argmax(cum_return[:index_j])  # start
        max_drawdown = (cum_return[index_j] - cum_return[index_i]) / cum_return[index_i]

    return max_drawdown


def portfolio_mean_daily_return(daily_data):
    return daily_data.pct_change().dropna().mean()


def portfolio_cum_total_return(daily_data, weights):
    returns = daily_data.pct_change()
    returns = returns.dropna()
    daily_returns = (returns * weights).sum(axis=1)
    daily_total_returns = daily_returns + 1
    cum_return = daily_total_returns.cumprod().dropna()
    return cum_return


def asset_cum_total_return(daily_data):
    total_return = daily_data.div(daily_data.shift(1)).dropna()
    return total_return.cumprod().dropna()


def get_risk_profile():
    """
    use this risk profile to control all of the risk profile of whole project,
    possible choices: "conservative","moderate_conservative", "moderate", "moderate_aggressive", "aggressive"
    :return:
    """
    return "moderate"
