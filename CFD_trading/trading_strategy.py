from technical_analysis.technical_indicators_calculator \
    import set_technical_indicators, Company, get_return_by_indicator
from technical_analysis.technical_indicators_chart_plotting import TechnicalIndicatorsChartPlotter
import pandas as pd
import numpy as np
from matplotlib.dates import AutoDateLocator
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    cfd_hs50 = pd.read_excel(r"data/07_2021/hs50.xlsx", index_col=0, parse_dates=False)

    test = cfd_hs50[:"2021-7-2"]

    company = Company("^HSI")
    config = {}
    company.prices = test
    set_technical_indicators(config, company)

    tacp = TechnicalIndicatorsChartPlotter()
    tacp.plot_macd(company)
    tacp.plot_rsi(company)
    tacp.plot_bollinger_bands(company)
    tacp.plot_stoch_rsi(company)
    tacp.plot_stoch(company)
    tacp.plot_adx(company)

    print("Return of using {} indicator:".format("MACD"), get_return_by_indicator("MACD", company))
    print("Return of using {} indicator:".format("RSI"), get_return_by_indicator("RSI", company))
    print("Return of using {} indicator:".format("Bollinger_Bands"), get_return_by_indicator("Bollinger_Bands", company))
    print("Return of using {} indicator:".format("STOCH_RSI"), get_return_by_indicator("STOCH_RSI", company))
    print("Return of using {} indicator:".format("STOCH"), get_return_by_indicator("STOCH", company))
    # print("Return of using {} indicator:".format("ADX"), get_return_by_indicator("ADX", company))

    # print(company.technical_indicators.head())

    """fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    autodate = AutoDateLocator()
    ax1.xaxis.set_major_locator(autodate)
    index = []
    for date in cfd_hs50_close.index.values:
        index.append(np.datetime_as_string(date, unit='m'))
    # print(index)
    plt.plot(index, cfd_hs50_close.values, label="HS50")
    plt.legend()
    plt.grid()
    plt.show()

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.xaxis.set_major_locator(autodate)
    exp1 = cfd_hs50_close.ewm(span=12, adjust=False).mean()
    exp2 = cfd_hs50_close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    plt.plot(index, macd, label='MACD', color='#EBD2BE')
    plt.plot(index, exp3, label='Signal Line', color='#E5A4CB')
    plt.legend(loc='upper left')
    plt.show()"""
