import numpy as np
import pandas as pd
from portfolio.common import AbstractAssetList, AbstractAssetAction
from scipy.optimize import minimize
from util.portfolio_property import portfolio_max_drawdown, portfolio_cum_total_return, portfolio_mean_daily_return, \
    portfolio_return_cov_sharpe, get_risk_profile

global daily_data
risk_profile = get_risk_profile()
risk_profile_metric_dict = {
    "conservative": {"return": 0.18, "sharpe": 1.0, "max drawdown": 0.25, "filename": "conservative"},
    "moderate_conservative": {"return": 0.15, "sharpe": 1.1, "max drawdown": 0.2, "filename": "moderate"},
    "moderate": {"return": 0.12, "sharpe": 1.2, "max drawdown": 0.15, "filename": "moderate"},
    "moderate_aggressive": {"return": 0.08, "sharpe": 1.3, "max drawdown": 0.10, "filename": "moderate"},
    "aggressive": {"return": 0.04, "sharpe": 1.4, "max drawdown": 0.05, "filename": "aggressive"}
}

all_daily_data = pd.read_csv(r"data/" + risk_profile + "_daily_data4.csv", index_col=0, parse_dates=True)  # windows file location

# all_daily_data = pd.read_csv(r"~/MCTS_moderate_conservative_v2.3/util/moderate_daily_data4.csv", index_col=0, parse_dates=True)  # linux file location
all_asset_list = list(all_daily_data.columns)

etf_info = pd.read_csv(r"data/etf_info.csv", index_col=0)  # windows file location


# etf_info = pd.read_csv(r"~/MCTS_moderate_conservative_v2.3/util/etf_info_add_industry.csv", index_col=0)  # linux file location

def get_return_over_max_drawdown(weights, requirement=1.3):
    """
    get the return over max drawdown and minus 1.3 (the requirement)
    :param requirement: the requirement of return over max drawdown
    :param weights: the weights of given assets
    :return: the return over max drawdown - 1.3
    """
    mean_daily_returns = portfolio_mean_daily_return(daily_data)  # get the mean daily return
    ret = np.sum(mean_daily_returns * weights) * 252  # get the portfolio annual return
    cum_return = portfolio_cum_total_return(daily_data, weights)  # get the cumulative return of portfolio
    drawdown = portfolio_max_drawdown(cum_return)  # get the maximum drawdown of portfolio
    return ret / (-drawdown) - requirement


def check_commodities_weight(weights, requirement=0.2):
    """
    check if the weight of commodities is less than 0.2
    :param requirement: the requirement of maximum weight of commodities
    :param weights: the weights of given assets
    :return: 0.2 - commodities weight
    """
    tickers = list(daily_data.columns.values)  # get the tickers e.g. QQQ Adj Close
    commodities_weight = 0  # initialize the commodities weight

    for ticker in tickers:
        # get the asset industry
        asset_industry = list(etf_info[etf_info['Ticker'] == ticker[:-10]]['Asset Class'].values)[0]

        # get the weights of all commodities assets
        if asset_industry == 'Commodities':
            index = tickers.index(ticker)
            commodities_weight += weights[index]

    return requirement - commodities_weight


def get_ret_vol_sr(weights):
    """
    get the annual return, volatility and sharpe ratio of given portfolio
    :param weights: the weights of given assets
    :return: return an array contains return, volatility and sharpe ratio
    """
    [ret, vol, sr] = portfolio_return_cov_sharpe(daily_data, weights)
    return np.array([ret, vol, sr])


def neg_sharpe(args):
    """
    minimize function
    :param args: the weights of given assets
    :return: negative sharpe ratio
    """
    return get_ret_vol_sr(args)[2] * -1


def check_sum(weights):
    """
    check if the sum of the weights is 1
    :param weights: the weights of given assets
    :return: the weights of given assets - 1
    """
    return np.sum(weights) - 1


def check_max_weight(weights):
    """
    check if the maximum weight is less than 0.4
    :param weights: the weights of given assets
    :return: 0.4 - the weights of given assets
    """
    return 0.4 - max(weights)


def check_no_of_assets(weights):
    """
    check the number of assets whose weight is greater than 0.02
    :param weights: the weights of given assets
    :return: the number of assets whose weight is greater than 0.02 - 3
    """
    temp = np.array(weights)
    return len(temp[temp >= 0.02]) - 3


def check_return(weights):
    """
    check if return satisfy the requirement of different risk profile
    :param weights: the weights of given assets
    :return: the portfolio return - requirement
    """
    ret, _, _ = portfolio_return_cov_sharpe(daily_data, weights)

    return ret - risk_profile_metric_dict[risk_profile]['return']


def get_result(selected_asset_list, selected_daily_data):
    n = len(selected_asset_list)  # get the number of asset
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples
    # construct the constraints
    cons1 = ({'type': 'eq', 'fun': check_sum},
             {'type': 'ineq', 'fun': get_return_over_max_drawdown},
             {'type': 'ineq', 'fun': check_return},
             {'type': 'ineq', 'fun': check_no_of_assets},
             {'type': 'ineq', 'fun': check_max_weight},
             {'type': 'ineq', 'fun': check_commodities_weight})

    # set the global variable
    global daily_data
    daily_data = selected_daily_data
    # print(selected_daily_data)
    opt_results = minimize(neg_sharpe, init_guess, bounds=bounds, method='SLSQP', constraints=cons1)
    returns, volatility, sharpe_ratio = get_ret_vol_sr(opt_results.x)
    weights = opt_results.x

    drawdown_over_return = get_return_over_max_drawdown(weights)
    # print("success!")
    # print(selected_asset_list)
    # print("Returns", returns)
    # print("Volatility", volatility)
    # print("Sharpe", sharpe_ratio)
    # print("Drawdown over return", drawdown_over_return)
    # print(weights)

    if returns >= risk_profile_metric_dict[risk_profile]['return'] \
            and sharpe_ratio >= risk_profile_metric_dict[risk_profile]['sharpe'] \
            and drawdown_over_return >= 0:  # /?
        # double check the result

        # check the weight of assets
        temp = np.array(weights)
        if any(temp >= 0.5) or len(temp[temp >= 0.02]) <= 3:
            return None

        new_asset_list = []
        asset_industry_list = []
        total_broad_no = 0

        # check the number of categories of assets
        for asset in selected_asset_list:
            ticker = asset[:-10]
            new_asset_list.append(ticker)
            # print(ticker)
            asset_industry = etf_info[etf_info['Ticker'] == ticker]['Industry'].values[0]
            if asset_industry == "Total" or "Broad":
                total_broad_no += 1
                asset_industry_list.append(asset_industry)

            else:
                asset_industry_list.append(asset_industry)

        if total_broad_no < 3 and len(set(asset_industry_list)) < 3:
            return None

        # generate the successful results
        results = {"Assets": [list(np.array(selected_asset_list)[temp >= 0.02])],
                   "Industry": [list(np.array(asset_industry_list)[temp >= 0.02])],
                   "Returns": returns, "Volatility": volatility, "Sharpe": sharpe_ratio,
                   "Return over drawdown": drawdown_over_return,
                   "Weights": [list(temp[temp >= 0.02])], "Level": risk_profile}

        print(results)

        return results


class AssetMove(AbstractAssetAction):
    def __init__(self, asset_name, selected_daily_data):
        self.asset_name = asset_name
        self.selected_daily_data = selected_daily_data

    def __repr__(self):
        return "asset_name:{0}".format(
            self.asset_name,
        )


class PortfolioAssetList(AbstractAssetList):
    def __init__(self, selected_asset_list, selected_daily_data):
        self.selected_asset_list = selected_asset_list
        self.selected_daily_data = selected_daily_data
        self.selected_asset_weight = None

    @property
    def portfolio_result(self):
        """
        check the result of the selected asset
        :return: None if the result is not satisfied, else 'aggressive', 'moderate_aggressive', ...
        """

        # screen the asset list that has less than 2 assets
        if len(self.selected_asset_list) < 3:
            return None

        if len(self.selected_asset_list) > 7:
            return "fail"
        # if the asset list has more than 5 asset, then fail

        # get the result from selected asset list
        results = get_result(self.selected_asset_list, self.selected_daily_data)

        if results is None:
            return None

        # if result is equal to risk profile, return the result
        if results['Level'] == risk_profile:
            # print(self.selected_asset_list)
            return results

        # if not over - no result
        return None

    def is_satisfied(self):
        # return if the portfolio result is None or equal to risk profile
        return self.portfolio_result is not None

    def move(self, move):
        """
        add the new asset into current PortfolioAssetList
        :param move: AssetMove
        :return: return the PortfolioAssetList with new asset list
        """

        # test if the selected_asset_list is None, then create
        # else append to list
        if self.selected_asset_list is None:
            new_asset_list = [move.asset_name]
        else:
            new_asset_list = self.selected_asset_list.copy()
            new_asset_list.append(move.asset_name)

        # test if the selected_daily_data, then set the dataframe
        # else concat the dataframe
        if self.selected_daily_data is None:
            new_daily_data = move.selected_daily_data.to_frame()
        else:
            new_daily_data = pd.concat([self.selected_daily_data, move.selected_daily_data.to_frame()], axis=1)

        return PortfolioAssetList(new_asset_list, new_daily_data)

    def get_possible_asset(self):
        """
        get the possible asset except for selected asset list
        :return: the possible asset move list contains the untried asset
        """
        remain_asset_list = list(set(all_asset_list) - set(self.selected_asset_list))  # /? need to be modified
        return [
            AssetMove(asset, all_daily_data[asset])
            for asset in remain_asset_list
        ]
