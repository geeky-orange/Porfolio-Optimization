import time

import ibapi
from ib_insync import *


def connect():
    # util.startLoop()
    ib = IB()
    # change port for real - trading
    ib.connect(port=7497)
    return ib


def qualify_stock(ib,stock, exchange = 'SMART',currency = 'USD'):
    contract = Stock(stock, exchange,currency=currency)
    ib.qualifyContracts(contract)
    return contract


def create_buy_order(action, amount, limit_price, stop_price = 0):

    if stop_price == 0:
        stopPrice = limit_price - (0.1 * limit_price)

    """
    :param type_order: type of order - > MarketOrder()
                                         LimitOrder()
                                         StopLimitOrder
                                         StopOrder
    :param stop_price: if 0 then set as 2% of limit price
    :param amount: amount for execution -> int
    :param limit_price:
    :return: order object

    PS: ADD OTHER TYPES OF ORDERS WHEN NEEDED
    """
    try:
        # return StopLimitOrder(action,amount,limit_price, stop_price)
        order = LimitOrder(action,amount,limit_price)

        return order
    except Exception as e:
        print(e.with_traceback())
        print('problem in ib_sync')


def get_last_close(ib,contract):
    util.startLoop()
    ib.reqMarketDataType(4)
    stock = ib.reqMktData(contract)
    print("The Closing price is: {}".format(stock))
    stock.close
    return stock


def get_sell_order(amount,stop_price,action = "Buy"):
    """
    :param action: 'sell'
    :param amount: amount of shares -> int
    :param stop_price: -> integer
    :return: StopOrder object for selling
    """
    return StopOrder(action, amount, stop_price)


def place_order(ib, contract, order):
    print("Contract {}".format(contract))
    print("order {}".format(order))

    dps = str(ib.reqContractDetails(contract)[0].minTick + 1)[::-1].find('.') - 1
    if order.action == 'SELL':
        order.lmtPrice = round(order.lmtPrice - ib.reqContractDetails(contract)[0].minTick * 2, dps)
    elif order.action == 'BUY':
        order.lmtPrice = round(order.lmtPrice + ib.reqContractDetails(contract)[0].minTick * 2, dps)

    print("ORDER is {}".format(order))
    x = ib.placeOrder(contract, order)

    """
    :param ib: ib connected
    :param contract: contract for order
    :param order:  type -> order object
    :return: if success 
    """

    return 'Success'




#
# Stock('AMD', 'SMART', 'USD')
# Stock('INTC', 'SMART', 'USD', primaryExchange='NASDAQ')
# Forex('EURUSD')
# CFD('IBUS30')
# Future('ES', '20180921', 'GLOBEX')
# Option('SPY', '20170721', 240, 'C', 'SMART')
# Bond(

# if __name__ == '__main__':
#     ib = connect()
#     contract = qualify_stock(ib,'TSLA','NYSE')
#     print(contract)
#     ib.disconnect()
    # ib = connect()
    # place_order()