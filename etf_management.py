"""
etf_management.py contains methods that provide information on ETF prices and the cost to buy required number of securities.
"""

from portfolio_analysis import get_latest_adj_close
import numpy as np
import yfinance as yf
import datetime as dt
from technical_analysis import ibsync_integration as ib


def get_latest_adj_close(ticker):

    # Setting the date of today
    Current_Date = dt.datetime.today()
    enddate = Current_Date.strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=enddate)
        latest_adj_close = df[['Adj Close']].tail(1).values[0][0]
        print(latest_adj_close)
        return latest_adj_close

    except:
        print("Problem with yfinance download.")


def buy_etf(conn,selected_etfs):
    # conn = ib.connect()
    print("Connected to Trader Workstation")
    cnt = 0
    for etf in selected_etfs:
        print("Buying for {}".format(etf))
        try:
            contract = etf['contract']
            order = ib.create_buy_order('BUY', etf['shares'], etf['closing'],0)
            ib.place_order(conn,contract,order)
            cnt += 1
        except Exception as e:
            print(e.with_traceback())
    # conn.disconnect()

def connect():
    conn = ib.connect()
    return conn

def disconnect(ib):
    ib.disconnect()

def generate_portfolio_contracts(conn, portfolio,weight):
    contracts = []
    """
    :param conn: ib connection object
    :param portfolio: list of tickers
    :return: list of qualified contracts from ib
    """

    cnt = 0
    for ticker in portfolio:
        dict = {}
        tmp = ib.qualify_stock(conn, ticker)
        dict['ticker'] = tmp.symbol
        dict['closing'] = get_latest_adj_close(ticker)
        dict['contract'] = tmp
        dict['weight'] = weight[cnt]
        contracts.append(dict)
        cnt+=1
        # dict.clear()

    # print("DICTIONARY")
    # print(contracts)
    return contracts


def get_closing_prices(selected_etfs):
    print('no')

if __name__ == '__main__':

    conn = connect()
    investment_amount = 1000
    # selected_etfs = ['CWB', 'SHV', 'QQQ', 'IAU', 'SHY']
    selected_etfs = ['FTEC', 'ARKG', 'XLG', 'IAU']

    portfolio_weights = [0.3999944612478112, 0.39999999999999997, 0.050687953431871866, 0.1493175853203167]
    contracts = generate_portfolio_contracts(conn, selected_etfs,portfolio_weights)
    disconnect(conn)
        # [0.035571, 0.334979, 0.062914, 0.093326, 0.473210]
    # print("Printing sum of weights {}".format(np.sum(portfolio_weights)))
    # dict_of_contracts = {}
    #
    # latest_prices = [get_latest_adj_close(i) for i in selected_etfs]
    #
    # x = np.multiply(portfolio_weights, latest_prices)
    # print("x is :")
    # print(x)
    # required_amount = np.sum(x)
    # print("Investment amount is {} and required amount is {}".format(investment_amount, required_amount))
    #
    # number_of_portfolio = np.ceil(investment_amount/required_amount)
    #
    # print("Buy {} portfolios.".format(number_of_portfolio))
    #
    # i = 0
    # amount = []
    # for w in portfolio_weights:
    #     price = latest_prices[i]
    #     allocation_amount = w*investment_amount
    #     print("{} Price = {}, Weight = {}, Allocation amount = {} * {} = {}".format(selected_etfs[i], price,w,w,investment_amount, allocation_amount))
    #     print("Number of ETFs of {} that they get is {} and we need to buy is {}".format(selected_etfs[i], allocation_amount/price, np.ceil(allocation_amount/price)))
    #     amount.append(np.ceil(allocation_amount/price))
    #     i = i + 1
    #
    # # buy_etf(selected_etfs,amount,latest_prices)
    # print(amount)

    for contract in contracts:
        price = contract['closing']

        allocation_amount =contract['weight'] *investment_amount
        contract['allocation_amount'] = allocation_amount
        number_to_buy = np.ceil(allocation_amount/price)
        contract['shares'] = np.ceil(allocation_amount/price)
    print('final')
    print(contracts)
