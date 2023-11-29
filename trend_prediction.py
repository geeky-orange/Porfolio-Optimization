import datetime
import time
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import talib
# from candlestick import candlestick
import yfinance as yf
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
from matplotlib import pyplot as plt

candle_rankings = {
    "CDL3LINESTRIKE_Bull": 1,
    "CDL3LINESTRIKE_Bear": 2,
    "CDL3BLACKCROWS_Bull": 3,
    "CDL3BLACKCROWS_Bear": 3,
    "CDLEVENINGSTAR_Bull": 4,
    "CDLEVENINGSTAR_Bear": 4,
    "CDLTASUKIGAP_Bull": 5,
    "CDLTASUKIGAP_Bear": 5,
    "CDLINVERTEDHAMMER_Bull": 6,
    "CDLINVERTEDHAMMER_Bear": 6,
    "CDLMATCHINGLOW_Bull": 7,
    "CDLMATCHINGLOW_Bear": 7,
    "CDLABANDONEDBABY_Bull": 8,
    "CDLABANDONEDBABY_Bear": 8,
    "CDLBREAKAWAY_Bull": 10,
    "CDLBREAKAWAY_Bear": 10,
    "CDLMORNINGSTAR_Bull": 12,
    "CDLMORNINGSTAR_Bear": 12,
    "CDLPIERCING_Bull": 13,
    "CDLPIERCING_Bear": 13,
    "CDLSTICKSANDWICH_Bull": 14,
    "CDLSTICKSANDWICH_Bear": 14,
    "CDLTHRUSTING_Bull": 15,
    "CDLTHRUSTING_Bear": 15,
    "CDLINNECK_Bull": 17,
    "CDLINNECK_Bear": 17,
    "CDL3INSIDE_Bull": 20,
    "CDL3INSIDE_Bear": 56,
    "CDLHOMINGPIGEON_Bull": 21,
    "CDLHOMINGPIGEON_Bear": 21,
    "CDLDARKCLOUDCOVER_Bull": 22,
    "CDLDARKCLOUDCOVER_Bear": 22,
    "CDLIDENTICAL3CROWS_Bull": 24,
    "CDLIDENTICAL3CROWS_Bear": 24,
    "CDLMORNINGDOJISTAR_Bull": 25,
    "CDLMORNINGDOJISTAR_Bear": 25,
    "CDLXSIDEGAP3METHODS_Bull": 27,
    "CDLXSIDEGAP3METHODS_Bear": 26,
    "CDLTRISTAR_Bull": 28,
    "CDLTRISTAR_Bear": 76,
    "CDLGAPSIDESIDEWHITE_Bull": 46,
    "CDLGAPSIDESIDEWHITE_Bear": 29,
    "CDLEVENINGDOJISTAR_Bull": 30,
    "CDLEVENINGDOJISTAR_Bear": 30,
    "CDL3WHITESOLDIERS_Bull": 32,
    "CDL3WHITESOLDIERS_Bear": 32,
    "CDLONNECK_Bull": 33,
    "CDLONNECK_Bear": 33,
    "CDL3OUTSIDE_Bull": 34,
    "CDL3OUTSIDE_Bear": 39,
    "CDLRICKSHAWMAN_Bull": 35,
    "CDLRICKSHAWMAN_Bear": 35,
    "CDLSEPARATINGLINES_Bull": 36,
    "CDLSEPARATINGLINES_Bear": 40,
    "CDLLONGLEGGEDDOJI_Bull": 37,
    "CDLLONGLEGGEDDOJI_Bear": 37,
    "CDLHARAMI_Bull": 38,
    "CDLHARAMI_Bear": 72,
    "CDLLADDERBOTTOM_Bull": 41,
    "CDLLADDERBOTTOM_Bear": 41,
    "CDLCLOSINGMARUBOZU_Bull": 70,
    "CDLCLOSINGMARUBOZU_Bear": 43,
    "CDLTAKURI_Bull": 47,
    "CDLTAKURI_Bear": 47,
    "CDLDOJISTAR_Bull": 49,
    "CDLDOJISTAR_Bear": 51,
    "CDLHARAMICROSS_Bull": 50,
    "CDLHARAMICROSS_Bear": 80,
    "CDLADVANCEBLOCK_Bull": 54,
    "CDLADVANCEBLOCK_Bear": 54,
    "CDLSHOOTINGSTAR_Bull": 55,
    "CDLSHOOTINGSTAR_Bear": 55,
    "CDLMARUBOZU_Bull": 71,
    "CDLMARUBOZU_Bear": 57,
    "CDLUNIQUE3RIVER_Bull": 60,
    "CDLUNIQUE3RIVER_Bear": 60,
    "CDL2CROWS_Bull": 61,
    "CDL2CROWS_Bear": 61,
    "CDLBELTHOLD_Bull": 62,
    "CDLBELTHOLD_Bear": 63,
    "CDLHAMMER_Bull": 65,
    "CDLHAMMER_Bear": 65,
    "CDLHIGHWAVE_Bull": 67,
    "CDLHIGHWAVE_Bear": 67,
    "CDLSPINNINGTOP_Bull": 69,
    "CDLSPINNINGTOP_Bear": 73,
    "CDLUPSIDEGAP2CROWS_Bull": 74,
    "CDLUPSIDEGAP2CROWS_Bear": 74,
    "CDLGRAVESTONEDOJI_Bull": 77,
    "CDLGRAVESTONEDOJI_Bear": 77,
    "CDLHIKKAKEMOD_Bull": 82,
    "CDLHIKKAKEMOD_Bear": 81,
    "CDLHIKKAKE_Bull": 85,
    "CDLHIKKAKE_Bear": 83,
    "CDLENGULFING_Bull": 84,
    "CDLENGULFING_Bear": 91,
    "CDLMATHOLD_Bull": 86,
    "CDLMATHOLD_Bear": 86,
    "CDLHANGINGMAN_Bull": 87,
    "CDLHANGINGMAN_Bear": 87,
    "CDLRISEFALL3METHODS_Bull": 94,
    "CDLRISEFALL3METHODS_Bear": 89,
    "CDLKICKING_Bull": 96,
    "CDLKICKING_Bear": 102,
    "CDLDRAGONFLYDOJI_Bull": 98,
    "CDLDRAGONFLYDOJI_Bear": 98,
    "CDLCONCEALBABYSWALL_Bull": 101,
    "CDLCONCEALBABYSWALL_Bear": 101,
    "CDL3STARSINSOUTH_Bull": 103,
    "CDL3STARSINSOUTH_Bear": 103,
    "CDLDOJI_Bull": 104,
    "CDLDOJI_Bear": 104
}


def draw(df):
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)

    trace = go.Candlestick(
        open=o,
        high=h,
        low=l,
        close=c)
    layout = {
        'title': '2019 Feb - 2020 Feb Bitcoin Candlestick Chart',
        'yaxis': {'title': 'Price'},
        'xaxis': {'title': 'Index Number'},
    }

    data = [trace]
    fig = dict(data=data, layout=layout)
    plot(fig, filename='btc_candles')


def get_all_patterns(df):
    open = df.Open
    high = df.High
    low = df.Low
    close = df.Close
    candle_names = talib.get_function_groups()['Pattern Recognition']
    exclude_items = ('CDLCOUNTERATTACK',
                     'CDLLONGLINE',
                     'CDLSHORTLINE',
                     'CDLSTALLEDPATTERN',
                     'CDLKICKINGBYLENGTH')

    candle_names = [candle for candle in candle_names if candle not in exclude_items]
    print("candle names")
    print(candle_names)
    for candle in candle_names:
        df[candle] = getattr(talib, candle)(open, high, low, close)

    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan
    for index, row in df.iterrows():
        # no pattern found
        if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index, 'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                else:
                    container.append(pattern + '_Bear')
            try:
                rank_list = [candle_rankings[p] for p in container]
                if len(rank_list) == len(container):
                    rank_index_best = rank_list.index(min(rank_list))
                    df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                    df.loc[index, 'candlestick_match_count'] = len(container)
            except Exception as e:
                print(e)
    # clean up candle columns
    df.drop(candle_names, axis=1, inplace=True)
    print("final")
    print(df)

    df = create_candlestick_signals(df)
    # df.to_csv('data.csv')
    # draw(df.tail(50))
    return df


# def create_bollinger_band_signals(df):


def create_candlestick_signals(df):
    df = pd.DataFrame(df)
    df['signal'] = "None"
    for index, row in df.iterrows():
        print("pattern")
        print(row['candlestick_pattern'])
        if row['candlestick_pattern'] is None or row['candlestick_pattern'] == 'NO_PATTERN' or row[
            'candlestick_pattern'] == 'nan':
            continue
        elif 'bull' in str(row['candlestick_pattern']).lower():
            print("BULL")
            # if int(row['candlestick_match_count']) > 3:
            #     row['signal'] = 'Strong Buy'
            # elif int(row['candlestick_match_count']) <= 3:
            #     row['signal'] = 'Buy'
            row['signal'] = 'Buy'
            df.at[index, 'signal'] = 'Buy'
        elif 'bear' in str(row['candlestick_pattern']).lower():
            print('bear')
            # if int(row['candlestick_match_count']) > 3:
            #     row['signal'] = 'Strong Sell'
            # elif int(row['candlestick_match_count']) <= 3:
            #     row['signal'] = 'Sell'
            df.at[index, 'signal'] = 'Sell'

    print(df)
    # df.to_csv("candle_with_signal.csv")
    # print(df.signal.value_counts())
    # print(list(df.signal))
    # final_signal(df)
    # signal_show(df.tail(50))
    return df


def signal_show(df):
    # %matplotlib qt
    df['Date'] = df['Datetime']
    #     time = list(df['Date'])
    #     xi = list(range(len(time)))
    #     plt.xticks(xi, time)
    plt.plot(df.Date, df['close'], linewidth=0.5, color='black')
    plt.scatter(df.loc[df['final_signal'] == 'Buy', 'Datetime'].values,
                df.loc[df['final_signal'] == 'Buy', 'close'].values,
                label='skitscat', color='green', s=25, marker="^")
    plt.scatter(df.loc[df['final_signal'] == 'Sell', 'Datetime'].values,
                df.loc[df['final_signal'] == 'Sell', 'close'].values,
                label='skitscat', color='red', s=25, marker="v")

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('HSI stock price with buy and sell signal')

    # Saving image
    plt.show()
    plt.savefig('HDFC with SMA 20-100 Buy sell.png')


def pivot(close, high, low):
    pivot = (high + low + close) / 3
    print('PIVOT IS {}'.format(pivot))
    return pivot


def final_signal(df):
    df['final_signal'] = 'Hold'

    flag = False
    total = 0.0
    df.reset_index(inplace=True)
    # print(df)
    buy_price = 0
    pnl_list = []
    pnl = 0
    for index, row in df.iterrows():
        # print("index {}".format(index))
        # print('buying price {}'.format(buy_price) )
        if flag and ((row['close'] - buy_price)/buy_price) * 100 > 0.5:
            print('Final Signal Sell for cap')
            df.at[index, 'final_signal'] = 'Sell'
            total += row['close']
            pnl = row['close'] - buy_price
            pnl_list.append(pnl)
            flag = False
        elif str(row['Datetime'].time()) == '16:08:00' and flag:
            print('Final Signal Sell for end of day')
            df.at[index, 'final_signal'] = 'Sell'
            total += row['close']
            pnl = row['close'] - buy_price
            pnl_list.append(pnl)
            flag = False
        elif (index > 0 and buy_price > 1 * pivot(df.at[index, 'close'], df.at[index, 'high'],
                                                df.at[index, 'low'])) and flag:
            print('Final Signal Sell for pivot')
            df.at[index, 'final_signal'] = 'Sell'
            total += row['close']
            pnl = row['close'] - buy_price
            pnl_list.append(pnl)
            flag = False
        elif (row['signal'] is None) or (row['signal'] == 'Buy' and flag) or (row['signal'] == 'Sell' and not flag) or (
                str(row['signal']) == 'None'):
            print('Continue')
            continue
        elif row['signal'] == 'Buy' and not flag:
            print('Final /signal Buy')
            df.at[index, 'final_signal'] = 'Buy'
            buy_price = row['close']
            total -= row['close']
            flag = True
        elif row['signal'] == 'Sell' and flag:
            print('Final Signal Sell')
            df.at[index, 'final_signal'] = 'Sell'
            total += row['close']
            pnl = row['close'] - buy_price
            pnl_list.append(pnl)
            flag = False

    print("PNL LIST")
    print(pnl_list)
    print(len(pnl_list))
    print(sum(pnl_list))
    only_pos = [num for num in pnl_list if num >= 1]
    pos_count = len(only_pos)
    print("Positive numbers in the list: ", pos_count)
    print("Negative numbers in the list: ", len(pnl_list) - pos_count)
    print(df.final_signal.value_counts())
    print("Total Revenue: {}".format(total))
    signal_show(df)


def sma(data, window):
    sma = data.rolling(window=window).mean()
    return sma


def bb(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb


def create_bollinger_signals(df):
    df['sma_20'] = sma(df['close'], 20)
    df['upperband'], df['lowerband'] = bb(df['close'], df['sma_20'], 20)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['signal'] = None
    for index, row in df.iterrows():
        if index == 0:
            continue
        if df.at[index - 1, 'close'] > df.at[index - 1, 'lowerband'] and row['close'] < row['lowerband']:
            print('low')
            df.at[index, 'signal'] = 'Buy'
        elif df.at[index - 1, 'close'] < df.at[index - 1, 'upperband'] and row['close'] > row['upperband']:
            print('high')
            df.at[index, 'signal'] = 'Sell'
    #     final_signal(df)
    print(df['signal'].value_counts())
    return df


def create_rsi_signal(df):
    df['rsi'] = talib.RSI(df['close'])
    for index, row in df.iterrows():
        if row['rsi'] < 30:
            df.at[index, 'signal'] = 'Buy'
        if row['rsi'] > 70:
            df.at[index, 'signal'] = 'Sell'
    #     final_signal(df)
    return df



if __name__ == '__main__':
    interval = [2, 5]
    print("Number  : {} ".format(3))
    df = yf.download('^HSI', interval='{}m'.format(2), period='7D')
    # df = yf.download('AAPL', interval='{}D'.format(), period='max')
    print(df)
    df.rename(columns={'Open': 'open', 'Close': 'close', 'Low': 'low', 'High': 'high'})
    # print(df)
    df = get_all_patterns(df)
    df = df.rename(columns={'Open': 'open', 'Close': 'close', 'Low': 'low', 'High': 'high'})
    print("DF before bollinger")
    print(df.columns)

    # df = create_bollinger_signals(df)
    df = create_rsi_signal(df)
    # print(df)
    final_signal(df)

    print("working")

    # df = df.rename(columns={'Open':'open','Close':'close', 'Low':'low','High':'high'})
    # df_copy = df.copy()
    # df_copy.reset_index(drop=True, inplace=True)
    # target = 'InvertedHammers'
    # df_copy = candlestick.inverted_hammer(df_copy, target='RESULT')
    # print("done")
    #
    #
    # # df_copy = candlestick.doji_star(df_copy, target='doji')
    # df_copy = candlestick.bearish_harami(df_copy, target = 'bearish_harami')
    # df['bearish_harami'] = list(df_copy['bearish_harami'])
    #
    # df_copy = candlestick.bullish_harami(df_copy,target='bullish_harami')
    #
    # df_copy = candlestick.dark_cloud_cover(df_copy,target='dark_cloud_cover')
    #
    # df_copy = candlestick.doji(df_copy,target='doji')
    #
    # df_copy = candlestick.dragonfly_doji(df_copy,target='dragonfly_doji')
    #
    # df_copy = candlestick.hanging_man(df_copy,target='hanging_man')
    #
    # df_copy = candlestick.gravestone_doji(df_copy,target='gravestone_doji')
    #
    # df_copy = candlestick.bearish_engulfing(df_copy,target='bearish_engulfing')
    #
    # df_copy = candlestick.bullish_engulfing(df_copy,target='bullish_engulfing')
    #
    # df_copy = candlestick.hammer(df_copy,target='hammer')
    # df['hammer'] = list(df_copy['hammer'])
    #
    # df_copy = candlestick.morning_star(df_copy,target='morning_star')
    #
    # df_copy = candlestick.morning_star_doji(df_copy,target='morning_star_doji')
    #
    # df_copy = candlestick.piercing_pattern(df_copy,target='piercing_pattern')
    #
    # df_copy = candlestick.rain_drop(df_copy,target='rain_drop')
    # df['rain_drop'] = list(df_copy['rain_drop'])
    # # candles_df = candlestick.rain_drop_doji(candles_df)
    #
    # df_copy = candlestick.star(df_copy,target= 'star')
    # df['star'] = list(df_copy['star'])
    #
    # df_copy = candlestick.shooting_star(df_copy, target= 'shooting_star')
    # df['shooting_star'] = list(df_copy['shooting_star'])
    #
    # print(df_copy)
    # df_copy['time'] = list(df.index)
    #
    # df_copy.to_csv('{}_{}_number.csv'.format(datetime.datetime.today().date(),i))
    # print(df_copy)
