from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator

import numpy as np
import pandas
from ta.volatility import BollingerBands


class Company:
    def __init__(self, symbol):
        self.symbol = symbol
        self.technical_indicators = None
        self.prices = None


def generate_buy_sell_signals(condition_buy, condition_sell, dataframe, strategy):
    last_signal = None
    indicators = []
    buy = []
    sell = []
    for i in range(0, len(dataframe)):
        # if buy condition is true and last signal was not Buy
        if condition_buy(i, dataframe) and last_signal != 'Buy':
            last_signal = 'Buy'
            indicators.append(last_signal)
            buy.append(dataframe['Close'].iloc[i])
            sell.append(np.nan)
        # if sell condition is true and last signal was Buy
        elif condition_sell(i, dataframe) and last_signal == 'Buy':
            last_signal = 'Sell'
            indicators.append(last_signal)
            buy.append(np.nan)
            sell.append(dataframe['Close'].iloc[i])
        else:
            indicators.append(last_signal)
            buy.append(np.nan)
            sell.append(np.nan)

    dataframe[f'{strategy}_Last_Signal'] = np.array(last_signal)
    dataframe[f'{strategy}_Indicator'] = np.array(indicators)
    dataframe[f'{strategy}_Buy'] = np.array(buy)
    dataframe[f'{strategy}_Sell'] = np.array(sell)


def set_technical_indicators(config, company):
    company.technical_indicators = pandas.DataFrame()
    company.technical_indicators['Close'] = company.prices["bid Close"]

    get_macd(config, company)
    get_rsi(config, company)
    get_bollinger_bands(config, company)
    get_stoch_rsi(config, company)
    get_stoch(config, company)
    get_adx(config, company)


def get_macd(config, company):
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators
    window_slow = 26  # need to be modified
    signal = 9  # need to be modified
    window_fast = 12  # need to be modified
    macd = MACD(close_prices, window_slow, window_fast, signal)
    dataframe['MACD'] = macd.macd()
    dataframe['MACD_Histogram'] = macd.macd_diff()
    dataframe['MACD_Signal'] = macd.macd_signal()

    generate_buy_sell_signals(
        # lambda x, df: df['MACD'].values[x] < df['MACD_Signal'].iloc[x] and df['MACD'].values[x] < 0,
        # lambda x, df: df['MACD'].values[x] > df['MACD_Signal'].iloc[x] and df['MACD'].values[x] > 0,
        lambda x, df: x - 1 >= 0 and df['MACD'].values[x - 1] < df['MACD_Signal'].iloc[x - 1] and
                      df['MACD_Signal'].iloc[x] < df['MACD'].values[x] < 0,
        lambda x, df: x - 1 >= 0 and df['MACD'].values[x - 1] > df['MACD_Signal'].iloc[x - 1] and
                      df['MACD_Signal'].iloc[x] > df['MACD'].values[x] > 0,
        dataframe,
        'MACD')
    return dataframe


def get_rsi(config, company):
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators
    # rsi_time_period = 20
    rsi_time_period = 14  # change the rsi

    rsi_indicator = RSIIndicator(close_prices, rsi_time_period)
    dataframe['RSI'] = rsi_indicator.rsi()

    # low_rsi = 40
    low_rsi = 30
    high_rsi = 70

    generate_buy_sell_signals(
        lambda x, df: df['RSI'].values[x] < low_rsi,
        lambda x, df: df['RSI'].values[x] > high_rsi,
        dataframe, 'RSI')

    return dataframe


def get_bollinger_bands(config, company):
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators

    window = 20

    indicator_bb = BollingerBands(close=close_prices, window=window, window_dev=2)

    # Add Bollinger Bands features
    dataframe['Bollinger_Bands_Middle'] = indicator_bb.bollinger_mavg()
    dataframe['Bollinger_Bands_Upper'] = indicator_bb.bollinger_hband()
    dataframe['Bollinger_Bands_Lower'] = indicator_bb.bollinger_lband()

    generate_buy_sell_signals(
        lambda x, df: df['Close'].values[x] < df['Bollinger_Bands_Lower'].values[x],
        lambda x, df: df['Close'].values[x] > df['Bollinger_Bands_Upper'].values[x],
        dataframe, 'Bollinger_Bands')

    return dataframe


def get_stoch_rsi(config, company):
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators

    indicator_stochrsi = StochRSIIndicator(close_prices)

    dataframe['STOCH_RSI'] = indicator_stochrsi.stochrsi()  # windows = 14
    indicator_sma = SMAIndicator(close_prices, window=10)
    dataframe['SMA_10'] = indicator_sma.sma_indicator()
    low_stoch_rsi = 0.1
    high_stoch_rsi = 0.9

    generate_buy_sell_signals(
        lambda x, df: df['STOCH_RSI'].iloc[x] < low_stoch_rsi and close_prices.values[x] < dataframe['SMA_10'].iloc[x],
        lambda x, df: df['STOCH_RSI'].iloc[x] > high_stoch_rsi and close_prices.values[x] > dataframe['SMA_10'].iloc[x],
        dataframe, 'STOCH_RSI'
    )

    return dataframe


def get_stoch(config, company):
    high_prices = company.prices['bid High']
    low_prices = company.prices['bid Low']
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators

    indicator_stoch = StochasticOscillator(close=close_prices, low=low_prices, high=high_prices)

    dataframe['STOCH'] = indicator_stoch.stoch()
    dataframe['STOCH_Signal'] = indicator_stoch.stoch_signal()
    low_stoch = 20
    high_stoch = 80

    generate_buy_sell_signals(
        lambda x, df: x + 1 < df.shape[0] and df['STOCH'].values[x] < low_stoch < df['STOCH'].values[x + 1],
        lambda x, df: x + 1 < df.shape[0] and df['STOCH'].values[x] > high_stoch > df['STOCH'].values[x + 1],
        dataframe, 'STOCH'
    )

    return dataframe


def get_adx(config, company):
    high_prices = company.prices['bid High']
    low_prices = company.prices['bid Low']
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators

    indicator_adx = ADXIndicator(close=close_prices, low=low_prices, high=high_prices)

    dataframe['ADX'] = indicator_adx.adx()
    dataframe['ADX_Pos'] = indicator_adx.adx_pos()
    dataframe['ADX_Neg'] = indicator_adx.adx_neg()
    signal_line = 20

    generate_buy_sell_signals(
        lambda x, df: df['ADX'].values[x] > signal_line and x - 1 > 0 and df['ADX_Neg'].values[x - 1] >
                      df['ADX_Pos'].values[x - 1] and df['ADX_Pos'].values[x] > df['ADX_Neg'].values[x],
        lambda x, df: df['ADX'].values[x] > signal_line and x - 1 > 0 and df['ADX_Neg'].values[x - 1] <
                      df['ADX_Pos'].values[x - 1] and df['ADX_Pos'].values[x] < df['ADX_Neg'].values[x],
        dataframe, 'ADX'
    )

    return dataframe


def get_williams(config, company):
    high_prices = company.prices['bid High']
    low_prices = company.prices['bid Low']
    close_prices = company.prices["bid Close"]
    dataframe = company.technical_indicators

    indicator_adx = WilliamsRIndicator(close=close_prices, low=low_prices, high=high_prices)

    dataframe['William'] = indicator_adx.williams_r()

    low_readings = -80
    high_readings = -20
    signal_line = -50

    generate_buy_sell_signals(
        lambda x, df: x - 20 > 0 and df['William'].values[x - 20] < low_readings and df['William'].values[x - 1] < signal_line < df['William'].values[x],
        lambda x, df: x - 20 > 0 and df['William'].values[x - 20] > high_readings and df['William'].values[x - 1] > signal_line > df['William'].values[x],
        dataframe, 'William'
    )


def get_return_by_indicator(strategy, company):
    holding = False
    total = 0.0
    last_buy = 0.0
    for index, row in company.technical_indicators.iterrows():
        if not np.isnan(row[f'{strategy}_Buy']) and not holding:
            # print("Buy in {}".format(row[f'{strategy}_Buy']))
            # print("Buy", row[f'{strategy}_Buy'])
            total -= row[f'{strategy}_Buy']
            last_buy = row[f'{strategy}_Buy']
            holding = True

        elif not np.isnan(row[f'{strategy}_Sell']) and holding:
            # print("Sell out {}".format(row[f'{strategy}_Sell']))
            # print("Sell", row[f'{strategy}_Sell'])
            total += row[f'{strategy}_Sell']
            holding = False

    if holding:
        total += last_buy

    return total
