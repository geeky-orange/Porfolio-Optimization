import os
from matplotlib import pyplot as plt
import numpy as np


class TechnicalIndicatorsChartPlotter:
    def plot_price_and_signals(self, fig, company, data, strategy, axs):
        last_signal_val = data[f'{strategy}_Last_Signal'].values[-1]
        last_signal = 'Unknown' if not last_signal_val else last_signal_val
        title = f'Close Price Buy/Sell Signals using {strategy}.  Last Signal: {last_signal}'
        fig.suptitle(f'Top: {company.symbol} Stock Price. Bottom: {strategy}')

        if not data[f'{strategy}_Buy'].isnull().all():
            axs[0].scatter(data.index, data[f'{strategy}_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if not data[f'{strategy}_Sell'].isnull().all():
            axs[0].scatter(data.index, data[f'{strategy}_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        axs[0].plot(company.prices["bid Close"], label='Close Price', color='blue', alpha=0.35)

        plt.xticks(rotation=45)
        axs[0].set_title(title)
        axs[0].set_xlabel('Date', fontsize=18)
        axs[0].set_ylabel('Close Price', fontsize=18)
        axs[0].legend(loc='upper left')
        axs[0].grid()

    def plot_macd(self, company):
        image = f'images/{company.symbol}_macd.png'
        macd = company.technical_indicators

        # Create and plot the graph
        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
        self.plot_price_and_signals(fig, company, macd, 'MACD', axs)

        axs[1].plot(macd['MACD'], label=company.symbol + ' MACD', color='blue')
        axs[1].plot(macd['MACD_Signal'], label='Signal Line', color='orange')
        positive = macd['MACD_Histogram'][(macd['MACD_Histogram'] >= 0)]
        negative = macd['MACD_Histogram'][(macd['MACD_Histogram'] < 0)]
        axs[1].bar(positive.index, positive, color='green')
        axs[1].bar(negative.index, negative, color='red')
        axs[1].legend(loc='upper left')
        axs[1].grid()
        print(os.path.abspath(image))
        plt.show()

    def plot_rsi(self, company):
        image = f'images/{company.symbol}_rsi.png'
        rsi = company.technical_indicators
        low_rsi = 30
        high_rsi = 70

        # plt.style.use('default')
        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
        self.plot_price_and_signals(fig, company, rsi, 'RSI', axs)
        axs[1].fill_between(rsi.index, y1=low_rsi, y2=high_rsi, color='#adccff', alpha=0.3)
        axs[1].plot(rsi.index, [50]*len(rsi.index), linestyle='--', linewidth=3, color='purple', alpha=0.3)
        axs[1].plot(rsi['RSI'], label='RSI', color='blue', alpha=0.35)
        axs[1].legend(loc='upper left')
        axs[1].grid()
        plt.show()

    def plot_bollinger_bands(self, company):
        image = f'images/{company.symbol}_bb.png'
        bollinger_bands = company.technical_indicators

        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))

        self.plot_price_and_signals(fig, company, bollinger_bands, 'Bollinger_Bands', axs)

        axs[1].plot(bollinger_bands['Bollinger_Bands_Middle'], label='Middle', color='blue', alpha=0.35)
        axs[1].plot(bollinger_bands['Bollinger_Bands_Upper'], label='Upper', color='green', alpha=0.35)
        axs[1].plot(bollinger_bands['Bollinger_Bands_Lower'], label='Lower', color='red', alpha=0.35)
        axs[1].fill_between(bollinger_bands.index, bollinger_bands['Bollinger_Bands_Lower'],
                            bollinger_bands['Bollinger_Bands_Upper'], alpha=0.1)
        axs[1].legend(loc='upper left')
        axs[1].grid()

        plt.show()

    def plot_stoch_rsi(self, company):
        image = f'images/{company.symbol}_stoch_rsi.png'
        stoch_rsi = company.technical_indicators
        low_stoch_rsi = 0.2
        high_stoch_rsi = 0.8

        # plt.style.use('default')
        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
        self.plot_price_and_signals(fig, company, stoch_rsi, 'STOCH_RSI', axs)
        axs[1].fill_between(stoch_rsi.index, y1=low_stoch_rsi, y2=high_stoch_rsi, color='#adccff', alpha=0.3)
        axs[1].plot(stoch_rsi.index, [0.5]*len(stoch_rsi.index), linestyle='--', linewidth=3, color='purple', alpha=0.3)
        axs[1].plot(stoch_rsi['STOCH_RSI'], label='STOCH_RSI', color='blue', alpha=0.5)
        # axs[1].plot(stoch_rsi['SMA_10'], label="SMA(10)", color='orange', alpha=0.5)
        axs[1].legend(loc='upper left')
        axs[1].grid()
        plt.show()

    def plot_stoch(self, company):
        image = f'images/{company.symbol}_stoch.png'
        stoch_rsi = company.technical_indicators
        low_stoch_rsi = 20
        high_stoch_rsi = 80

        # plt.style.use('default')
        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
        self.plot_price_and_signals(fig, company, stoch_rsi, 'STOCH', axs)
        axs[1].fill_between(stoch_rsi.index, y1=low_stoch_rsi, y2=high_stoch_rsi, color='#adccff', alpha=0.3)
        axs[1].plot(stoch_rsi.index, [50] * len(stoch_rsi.index), linestyle='--', linewidth=3, color='purple',
                    alpha=0.3)
        axs[1].plot(stoch_rsi['STOCH'], label='STOCH', color='blue', alpha=0.3)
        # axs[1].plot(stoch_rsi['SMA_10'], label="SMA(10)", color='orange', alpha=0.5)
        axs[1].legend(loc='upper left')
        axs[1].grid()
        plt.show()

    def plot_adx(self, company):
        image = f'images/{company.symbol}_adx.png'
        adx = company.technical_indicators
        signal = 20

        # plt.style.use('default')
        fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))
        self.plot_price_and_signals(fig, company, adx, 'ADX', axs)

        axs[1].plot(adx.index, [20] * len(adx.index), linestyle='--', linewidth=3, color='purple',
                    alpha=0.3)
        axs[1].plot(adx['ADX'], label='ADX', color='black', alpha=0.5)
        axs[1].plot(adx['ADX_Pos'], label='ADX_Pos', color='green', alpha=0.3)
        axs[1].plot(adx['ADX_Neg'], label='ADX_Neg', color='red', alpha=0.3)
        # axs[1].plot(stoch_rsi['SMA_10'], label="SMA(10)", color='orange', alpha=0.5)
        axs[1].legend(loc='upper left')
        axs[1].grid()
        plt.show()
