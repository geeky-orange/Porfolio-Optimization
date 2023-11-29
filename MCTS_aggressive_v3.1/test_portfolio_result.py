import pandas as pd
from tree.nodes import PortfolioConstructionTreeSearchNode
from tree.search import MonteCarloTreeSearch
from time import time
from portfolio.portfolio_construction import PortfolioAssetList
from util.portfolio_property import get_risk_profile

risk_profile = get_risk_profile()
pd.set_option('display.max_columns', None)
# windows file location
portfolio_daily_data = pd.read_csv("data/" + risk_profile + "_daily_data4.csv", index_col=0, parse_dates=True)


def test_if_reach_max_len():
    tickers = ["SPY Adj Close", "QQQ Adj Close", "IVV Adj Close", "VTI Adj Close",
               "VOO Adj Close", "VEA Adj Close", "IWD Adj Close", "ITOT Adj Close"]
    portfolio = PortfolioAssetList(tickers, portfolio_daily_data[tickers])
    result = portfolio.portfolio_result
    print(result)


def test_max_len_case():
    tickers = ["SPY Adj Close", "QQQ Adj Close", "IVV Adj Close", "VTI Adj Close",
               "VOO Adj Close"]
    portfolio = PortfolioAssetList(tickers, portfolio_daily_data[tickers])
    result = portfolio.portfolio_result
    print(result)


def test_init_case():
    tickers = []
    portfolio = PortfolioAssetList(tickers, portfolio_daily_data[tickers])
    result = portfolio.portfolio_result
    print(result)


def test_fail_case():
    tickers = ["SPY Adj Close", "QQQ Adj Close", "IVV Adj Close"]
    portfolio = PortfolioAssetList(tickers, portfolio_daily_data[tickers])
    result = portfolio.portfolio_result
    print(result)


def best_best_children():
    tickers = []
    start_asset = PortfolioAssetList(tickers, None)

    root = PortfolioConstructionTreeSearchNode(asset_list=start_asset,
                                               parent=None
                                               )

    mcts = MonteCarloTreeSearch(root)
    result = mcts.best_action(total_simulation_seconds=172800)
    root._portfolios_info.to_csv("final_" + risk_profile + "_portfolio.csv")
    # print(root._portfolios_info)


def test_drawback_case():
    tickers = ['SPY Adj Close']
    portfolio = PortfolioAssetList(tickers, portfolio_daily_data[tickers])
    result = portfolio.portfolio_result
    print(result)


if __name__ == "__main__":
    # test_if_reach_max_len()

    # test_max_len_case()

    # test_init_case()

    # test_fail_case()

    # test_drawback_case()

    # test_if_reach_max_len()

    # test_max_len_case()

    # test_init_case()

    # test_drawback_case()

    start = time()
    print("Start: " + str(start))
    best_best_children()
    stop = time()
    print("Stop: " + str(stop))
    print(str(stop - start) + "s")
