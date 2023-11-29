import yfinance as yf
import pandas as pd
import datetime as dt


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# gets adj. close historical data for ETFs for a specified number of years
def get_data(ETF_list, start_year, start_month, start_day, end_year, end_month, end_day):
    start_date = dt.datetime(year=start_year, month=start_month, day=start_day).strftime("%Y-%m-%d")
    end_date = dt.datetime(year=end_year, month=end_month, day=end_day).strftime("%Y-%m-%d")

    print("Printing ETF list: {}".format(ETF_list))
    df_list = []

    for etf in ETF_list:
        try:
            df = yf.download(etf, start=start_date, end=end_date)
            df = df[['Adj Close']]
            df.rename(columns={'Adj Close': '{} Adj Close'.format(str(etf))}, inplace=True)
            df_list.append(df)

        except Exception as e:
            print(e)
            print("Problem with yfinance download.")

    final_df = pd.concat(df_list, axis=1)
    final_df = final_df.dropna(axis='columns')
    # print("Final df is:")
    print(final_df)

    return final_df


def add_asset_class(df):
    asset_class1 = []
    asset_class2 = []

    for index, row in df.iterrows():
        if "Equity" in row["Segment"]:
            asset_class1.append("Equity")
            if "Large" in row["Segment"]:
                asset_class2.append("Large")
            elif "Mid" in row["Segment"]:
                asset_class2.append("Mid")
            elif "Small" in row["Segment"]:
                asset_class2.append("Small")
            elif "Total Market" in row["Segment"]:
                asset_class2.append("Total Market")
            else:
                asset_class2.append("Null")
        elif "Fixed Income" in row["Segment"]:
            asset_class1.append("Fixed Income")
            if "Government" in row["Segment"]:
                asset_class2.append("Government")
            elif "Ultra-Short Term" in row["Segment"]:
                asset_class2.append("Ultra-Short Term")
            elif "Broad Market" in row["Segment"]:
                asset_class2.append("Broad Market")
            elif "Corporate" in row["Segment"]:
                asset_class2.append("Corporate")
            else:
                asset_class2.append("Null")
        elif "Commodities" in row["Segment"]:
            asset_class1.append("Commodities")
            if "Precious Metals" in row["Segment"]:
                asset_class2.append("Precious Metals")
            elif "Broad Market" in row["Segment"]:
                asset_class2.append("Broad Market")
            else:
                asset_class2.append("Null")
        else:
            asset_class1.append("Null")

    df["Asset Class1"] = asset_class1
    df["Asset Class2"] = asset_class2

    return df


def start_download(download_category='conservative', start_year=2018, start_month=7, start_day=1, end_year=2021,
                   end_month=7, end_day=1):
    """
    download the data of given category and timeslot
    :param end_day:
    :param end_month:
    :param end_year:
    :param start_day:
    :param start_month:
    :param start_year:
    :param download_category: five categories: conservative, moderate_conservative, moderate, moderate_aggressive, portfolio
    :return: the download dataframe
    """
    etf_info = pd.read_csv("../data/etf_info.csv", index_col=0)

    download_dict = {"conservative": ['Fixed Income', 'Commodities'],
                     "moderate_conservative": ['Fixed Income', 'Commodities', 'Equity'],
                     "moderate": ['Fixed Income', 'Commodities', 'Equity'],
                     "moderate_aggressive": ['Fixed Income', 'Commodities', 'Equity'],
                     "portfolio": ['Equity', 'Commodities']}

    concat_df = []

    for category in download_dict[download_category]:
        concat_df.append(etf_info[etf_info['Asset Class'] == category])

    ticker_list = list(etf_info['Ticker'].values)
    daily_data = get_data(ticker_list, start_year, start_month, start_day, end_year, end_month, end_day)

    return daily_data
