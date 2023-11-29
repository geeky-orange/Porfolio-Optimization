"""
portfolio_analysis.py contains methods for performing mean variance analysis and portfolio optimization.
"""

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import discrete_allocation
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from io import BytesIO
import base64
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import time
import os
import datetime
from scipy.optimize import minimize
# from etf_categorize import get_small_cap,get_real_estate,get_int_large_cap,get_government_bonds,get_emerging_markets,get_large_cap,get_ultra_term_bonds,get_gov_short_term,get_gold,get_corporate_bonds

from etf_categorize_new import get_commodities, get_corporate_bonds, get_emerging_markets, get_global_non_US_equities, \
    get_government_bonds, get_large_cap, get_real_estate, get_small_mid, get_total_bonds, get_total_market_equities

"""
# date
mylist = []
today = datetime.date.today()
mylist.append(today)
# results directory
x = os.path.join(os.getcwd(), 'Results_'+str(mylist[0]))
print(x)
print(os.path.isdir(x))
# check if path exists and is directory
if os.path.exists(x) and os.path.isdir(x):
    print()
else:
    os.makedirs(x)
"""


# https://people.math.ethz.ch/~jteichma/markowitz_portfolio_optimization.html
# returns results of Monte Carlo simulation
def get_results_frame(data, number_of_simulations, sortino_target_return):
    target = 0
    components = list(data.columns.values)
    # convert daily stock prices into daily returns
    returns = data.pct_change()
    returns = returns.dropna()

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    # cov_matrix_sortino = returns
    # set number of runs of random portfolio weights
    num_portfolios = number_of_simulations

    """    
     set up array to hold results
     '3' is to store returns, standard deviation and sharpe ratio
     len(components) is for storing weights for each component in every portfolio
    """

    results = np.zeros((4 + len(components), num_portfolios))

    for i in list(range(num_portfolios)):
        # select random weights for portfolio holdings
        weights = np.random.random(len(components))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)
        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        daily_portfolio_return = returns * weights

        # add this sum col to results
        sum_col = daily_portfolio_return.sum(axis=1)
        tempseries = sum_col[sum_col < target]
        downside_deviation = tempseries.std()
        # print("downside deviation is {}".format(downside_deviation))

        # filter sum_col to get only negative values and use that to calculate semi-deviation/downside risk
        # use downside risk to calculate sortino ratio = portfolio_return / downside_risk

        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # downside_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # downside_returns = df.loc[df['pf_returns'] < target]
        # store results in results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev

        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = results[0, i] / results[1, i]

        # sortino ratio
        results[3, i] = portfolio_return / downside_deviation

        for j in range(len(weights)):
            # change the 3 in the following line to 4 and add 'sortino' to columns
            results[j + 4, i] = weights[j]

    cols = ['ret', 'stdev', 'sharpe', 'sortino']
    for stock in components:
        cols.append(stock)

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=cols)
    # results_frame.to_csv('datacsv.csv')
    return results_frame


# visualize weights of a portfolio in the form of a pie chart that is returned as image_base64
def visualize_portfolio(portfolio, title):
    weights = portfolio[3:]
    plt.title(title)
    plt.pie(weights.values, labels=list(weights.keys()), autopct='%.2f%%')
    plt.savefig('{}.png'.format(title))
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()
    plt.clf()
    plt.cla()
    plt.close()
    return image_base64


# takes ETF data as input and returns top number_of_tickers that have the highest weights from the category after Monte Carlo simulation
def portfolio_manager(input_data, minimum_target_return=0, number_of_simulations=25000,
                      number_of_tickers_returned=10):
    # clean input data
    data = input_data.dropna()
    print(data)

    """
    # testing with library
    max_sharpe_chart, max_sharpe_performance, sharpe_weights = optimize_max_sharpe(data)
    print("\n------------------------")
    print("\nMax sharpe from library:")
    print(max_sharpe_performance)
    print(sharpe_weights)
    print('\n')
    """

    # monte carlo simulation function is called
    results_frame = get_results_frame(data=data, number_of_simulations=number_of_simulations, sortino_target_return=0)
    print("results frame obtained is:")
    print(results_frame)

    if results_frame.empty:
        print("Please check data source.")
        return 0, 0

    # condition for minimum target return
    ret_condition = results_frame['ret'] >= minimum_target_return
    rf = results_frame[ret_condition]

    if rf.empty:
        print("Minimum target return criteria is not met as input value is too high.")
        return 0, 0

    print("Printing the filtered df:")
    print(rf)
    rf = rf.sort_values(by=['stdev'])
    print("Sorted by stdev:")
    print(rf)

    current_df = rf
    if current_df.empty:
        print("Please check the split function for errors.")
        return 0, 0

    else:
        max_sharpe = current_df.loc[int(current_df['sharpe'].idxmax())]
        max_ret = current_df.loc[int(current_df['ret'].idxmax())]
        min_var = current_df.loc[int(current_df['stdev'].idxmin())]
        max_sortino = current_df.loc[int(current_df['sortino'].idxmax())]
        print("\n------------------------")
        print("\nMax sharpe ratio from code:")
        print(max_sharpe)

        all_portfolios = [max_sharpe, min_var, max_ret, max_sortino]
        portfolio_names = ["Max. Sharpe Allocation", "Min. Volatility Allocation",
                           "Max. Returns Allocation", "Max. Sortino"]
        i = 0
        array_of_images = []
        for portfolio in all_portfolios:
            image = visualize_portfolio(portfolio, portfolio_names[i])
            array_of_images.append(image)
            i = i + 1

        dict = max_sharpe.to_dict()
        print(dict)
        print(type(dict))
        sort_orders = sorted(dict.items(), key=lambda x: x[1], reverse=True)

        # remove sharpe, stdev and ret
        k = 0
        tupledata = []
        for i in sort_orders:
            print(i[0], i[1])
            if k <= (number_of_tickers_returned - 1):
                if (i[0] != 'ret' and i[0] != 'sharpe' and i[0] != 'stdev' and i[0] != 'sortino'):
                    tupledata.append((str(i[0]).split(' ')[0], i[1]))
                    k = k + 1

        print("Tuple data is: {}".format(tupledata))

        # iterate over dict and return top number_of_tickers_returned values
        return array_of_images, all_portfolios, tupledata


def portfolio_manager2(input_data, minimum_target_return=0, number_of_simulations=25000,
                       number_of_tickers_returned=10):
    # clean input data
    data = input_data.dropna()
    print(data)

    """
    # testing with library
    max_sharpe_chart, max_sharpe_performance, sharpe_weights = optimize_max_sharpe(data)
    print("\n------------------------")
    print("\nMax sharpe from library:")
    print(max_sharpe_performance)
    print(sharpe_weights)
    print('\n')
    """

    # monte carlo simulation function is called
    results_frame = get_results_frame(data=data, number_of_simulations=number_of_simulations,
                                      sortino_target_return=minimum_target_return)

    print("results frame obtained is:")
    print(results_frame)
    # New tickers for moderately aggressive: ['CWB', 'IAU', 'SCHO', 'QQQ', 'XSOE', 'ARKK', 'FREL', 'SCZ']
    """
    int_filter = results_frame['IAU Adj Close'] <= 0.2
    cp_filter = results_frame['SHV Adj Close'] >= 0.2
    cpp_filter = results_frame['CWB Adj Close'] >= 0.2
    gld_filter = results_frame['SHY Adj Close'] >= 0.2
    results_frame = results_frame[int_filter & cp_filter & cpp_filter & gld_filter]
    print("After filter:")
    print(results_frame)
    """
    if results_frame.empty:
        print("Please check data source.")
        return 0, 0

    # condition for minimum target return
    ret_condition = results_frame['ret'] >= minimum_target_return
    rf = results_frame[ret_condition]
    print(rf)
    print(rf.index)
    index_remove = []
    for ind in rf.index:
        for column in rf:
            if (column != 'ret' and column != 'stdev' and column != 'sharpe' and column != 'sortino'):

                # eliminate rows that have any single weight more than 0.35
                if rf[column][ind] > 0.35 and rf[column][ind] < 0.01:
                    # print(rf[column][ind])
                    index_remove.append(ind)

    print("Removing the following indicies:")
    print(index_remove)
    rf = rf.drop(index_remove)

    print("Check for no weights more than 0.5.")
    print(rf)
    if rf.empty:
        print("Minimum target return criteria is not met as input value is too high.")
        return 0, 0

    print("Printing the filtered df:")
    print(rf)
    rf = rf.sort_values(by=['stdev'])
    print("Sorted by stdev:")
    print(rf)

    current_df = rf
    if current_df.empty:
        print("Please check the split function for errors.")
        return 0, 0

    else:
        max_sharpe = current_df.loc[int(current_df['sharpe'].idxmax())]
        max_ret = current_df.loc[int(current_df['ret'].idxmax())]
        min_var = current_df.loc[int(current_df['stdev'].idxmin())]
        max_sortino = current_df.loc[int(current_df['sortino'].idxmax())]
        print("\n------------------------")
        print("\nPortfolio max sharpe ratio with target return {}%:".format(str(minimum_target_return)))
        print(max_sharpe)

        all_portfolios = [max_sharpe, min_var, max_ret, max_sortino]
        portfolio_names = ["Max. Sharpe Allocation", "Min. Volatility Allocation",
                           "Max. Returns Allocation", "Max. Sortino"]
        i = 0
        array_of_images = []
        for portfolio in all_portfolios:
            image = visualize_portfolio(portfolio, portfolio_names[i])
            array_of_images.append(image)
            i = i + 1

        dict = max_sharpe.to_dict()
        print(dict)
        print(type(dict))
        sort_orders = sorted(dict.items(), key=lambda x: x[1], reverse=True)

        # remove sharpe, stdev and ret
        k = 0
        tupledata = []
        for i in sort_orders:
            print(i[0], i[1])
            if k <= (number_of_tickers_returned - 1):
                if (i[0] != 'ret' and i[0] != 'sharpe' and i[0] != 'stdev' and i[0] != 'sortino'):
                    tupledata.append((str(i[0]).split(' ')[0], i[1]))
                    k = k + 1

        print("Tuple data is: {}".format(tupledata))

        # iterate over dict and return top number_of_tickers_returned values
        return array_of_images, all_portfolios, tupledata

        # create scatter plot coloured by Sharpe Ratio
        """
        get_efficient_frontier(results_frame)
        # plot red star to highlight position of portfolio with highest Sharpe Ratio
        plt.scatter(max_sharpe[1], max_sharpe[0], marker="8", color='r', s=500)
        # plot green star to highlight position of minimum variance portfolio
        plt.scatter(min_var[1], min_var[0], marker=(5, 1, 0), color='g', s=500)
        # plot yellow star to highlight position of minimum variance portfolio
        plt.scatter(max_sharpe_with_target[1], max_sharpe_with_target[0], marker=(5, 1, 0), color='y', s=500)
        # plot yellow star to highlight position of minimum variance portfolio
        plt.scatter(max_returns_with_target[1], max_returns_with_target[0], marker=(5, 1, 0), color='coral', s=500)
        regression_correlation.make_directory('./Portfolio_allocation')
        plt.savefig('./Portfolio_allocation/Efficient frontier.png')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()
        plt.clf()
        plt.cla()
        plt.close()
        return image_base64
        """


# gets adj. close historical data for ETFs for a specified number of years
def get_data(ETF_list, number_of_years=5):
    # Setting the date of today
    Current_Date = dt.datetime.today()

    # Setting the historical date (starting date)
    Historical_Date = dt.datetime.now() - dt.timedelta(days=number_of_years * 365)
    # You can set the date here, say like "2020-07-23"
    startdate = Historical_Date.strftime("%Y-%m-%d")
    enddate = Current_Date.strftime("%Y-%m-%d")

    print("Printing ETF list: {}".format(ETF_list))
    dflist = []
    for etf in ETF_list:
        try:
            df = yf.download(etf, start=startdate, end=enddate)
            df = df[['Adj Close']]
            df.rename(columns={'Adj Close': '{} Adj Close'.format(str(etf))}, inplace=True)
            dflist.append(df)
        except:
            print("Problem with yfinance download.")

    finaldf = pd.concat(dflist, axis=1)
    finaldf = finaldf.dropna(axis='columns')
    print("Final df is:")
    print(finaldf)
    return finaldf


# initial ETF screening rules based on AUM, expense ratio and grade whilst removing inverse and leveraged ETFs
def screen_etfs(dataset):
    # Dataset is now stored in a Pandas Dataframe
    accepted_grading = ['A', 'B']
    # return_volatility_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-']
    invalidIndex = dataset[(dataset['Expense Ratio'] == '--') | (dataset['AUM'] == '--')].index
    dataset.drop(invalidIndex, inplace=True)
    # eliminate dollar sign
    dataset['AUM'] = dataset['AUM'].str.replace('$', '')

    # convert to number with float
    for ind in dataset.index:
        if dataset['AUM'][ind][-1] == 'B':
            dataset['AUM'][ind] = dataset['AUM'][ind][:-1]
            dataset['AUM'][ind] = float(dataset['AUM'][ind][:-1]) * 1000000000

        elif dataset['AUM'][ind][-1] == 'M':
            dataset['AUM'][ind] = dataset['AUM'][ind][:-1]
            dataset['AUM'][ind] = float(dataset['AUM'][ind][:-1]) * 1000000

        elif dataset['AUM'][ind][-1] == 'K':
            dataset['AUM'][ind] = dataset['AUM'][ind][:-1]
            dataset['AUM'][ind] = float(dataset['AUM'][ind][:-1]) * 1000

    # remove inverse and leveraged
    for ind in dataset.index:
        if ('Inverse' in dataset['Segment'][ind]) or ('Leveraged' in dataset['Segment'][ind]):
            dataset = dataset.drop([ind])

    dataset['Expense Ratio'] = dataset['Expense Ratio'].str.replace('%', '')
    dataset['Expense Ratio'] = dataset['Expense Ratio'].astype(float)

    # rules
    AUM_test = dataset['AUM'].astype(float) >= 1000000000
    ER_test = dataset['Expense Ratio'] <= 0.75
    grade_test = dataset['Grade'].isin(accepted_grading)
    print(len(dataset))
    dataset = dataset[grade_test]
    dataset = dataset[AUM_test]
    dataset = dataset[ER_test]

    return dataset


# returns a list of top n tickers from a given category
def get_category_data(listticker, number_of_years, minimum_target_return, number_of_simulations,
                      number_of_tickers_returned):
    if listticker:
        getdata = get_data(ETF_list=listticker, number_of_years=number_of_years)

        output_images_gold, gold_portfolios, tupledata = portfolio_manager(input_data=getdata,
                                                                           minimum_target_return=minimum_target_return,
                                                                           number_of_simulations=number_of_simulations,
                                                                           number_of_tickers_returned=number_of_tickers_returned)

        corporatebond_ticker = [i[0] for i in tupledata]
        return corporatebond_ticker

    else:
        return None


# returns the latest adjusted closing price for a given ticker
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


# returns sum of latest adjusted closing prices of 'tickers' that are provided as a list
def get_total_close(tickers):
    sum = 0
    for ticker in tickers:
        price = get_latest_adj_close(ticker)
        sum = sum + price

    print("Total amount = {}".format(sum))
    return sum


# creates columns for SPY and AGG,ACWI composite banchmark for comparison and returns final dataframe
def create_benchmark(etf_returns, startdate, enddate, agg_weight, acwi_weight):
    spydf = yf.download('SPY', start=startdate, end=enddate)
    spydf = spydf[['Adj Close']]
    spydf.rename(columns={'Adj Close': 'SPY Adj Close'}, inplace=True)
    spydf = spydf.pct_change()
    spydf = spydf.dropna()
    print("S&P data:")
    print(spydf)

    merged_df1 = pd.merge(etf_returns, spydf, how='inner', left_index=True, right_index=True)
    print(merged_df1)
    print(merged_df1.index)

    acwi = yf.download('ACWI', start=startdate, end=enddate)
    acwi = acwi[['Adj Close']]
    acwi.rename(columns={'Adj Close': 'ACWI Adj Close'}, inplace=True)
    acwi = acwi.pct_change()
    acwi = acwi.dropna()
    print("acwi data:")
    print(acwi)

    agg = yf.download('AGG', start=startdate, end=enddate)
    agg = agg[['Adj Close']]
    agg.rename(columns={'Adj Close': 'AGG Adj Close'}, inplace=True)
    agg = agg.pct_change()
    agg = agg.dropna()
    print("agg data:")
    print(agg)

    merged_df2 = pd.merge(agg, acwi, how='inner', left_index=True, right_index=True)
    print("merged_df2 is:")
    print(merged_df2)

    finalmerged_df = pd.merge(merged_df1, merged_df2, how='inner', left_index=True, right_index=True)
    finalmerged_df['Weighted benchmark'] = finalmerged_df['AGG Adj Close'] * agg_weight + finalmerged_df[
        'ACWI Adj Close'] * acwi_weight

    return finalmerged_df


# library function for optimizing sharpe ratio
def optimize_max_sharpe(stock_csv):
    # mean historical return
    mu = expected_returns.mean_historical_return(stock_csv)
    print(mu)
    print(type(mu))

    S = risk_models.sample_cov(stock_csv)
    print(S)
    print(type(S))

    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    # cleaned_weights = ef.clean_weights()
    cleaned_weights = raw_weights
    ef.save_weights_to_file("optimum allocation weights.csv")
    performance = ef.portfolio_performance(verbose=True, risk_free_rate=0)

    Expected_annual_return = round(float(performance[0]) * 100, 2)
    Annual_volatility = round(float(performance[1]) * 100, 2)
    Sharpe_Ratio = round(float(performance[2]), 2)

    performance = [Expected_annual_return, Annual_volatility, Sharpe_Ratio]

    # get keys and values for the pie chart
    keys, values = zip(*cleaned_weights.items())

    plt.pie(values, labels=keys, autopct='%.2f%%')
    plt.title("Optimal Portfolio")
    plt.savefig('optimum_allocation.png')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()

    plt.clf()
    plt.cla()
    plt.close()
    return image_base64, performance, cleaned_weights


def get_ef(prices, n, ticker_list=[]):
    # get data for tickers from start_date to end_date
    # prices = ffn.get(ticker_list, common_dates=True, start=start_date)
    print(prices)
    print(type(prices))
    returns = prices.pct_change()

    # Expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Optimise portfolio for maximum Sharpe Ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(raw_weights)
    print(cleaned_weights)
    print(type(cleaned_weights))
    ef.portfolio_performance(verbose=True)

    """
    latest_prices = discrete_allocation.get_latest_prices(prices)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000)
    allocation, leftover = da.lp_portfolio()
    print(allocation)
    print("Funds remaining: ${:.2f}".format(leftover))
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    ef.efficient_return(target_return=0.1, market_neutral=True)
    """
    # return top n tickers with weights


"""
    risk_profile = 4
    number_of_years = 5
    number_of_simulations = 25000
    finalportfoliotickers = []
"""


def simulate(data, risk_profile, number_of_years, number_of_simulations, finalportfoliotickers):
    # Setting the dates and weights for acwi and agg
    Current_Date = dt.datetime.today()
    Historical_Date = dt.datetime.now() - dt.timedelta(days=number_of_years * 365)
    startdate = Historical_Date.strftime("%Y-%m-%d")
    enddate = Current_Date.strftime("%Y-%m-%d")

    if risk_profile == 1:
        min_ret = 5;
        filename = 'Conservative';
        acwi_weight = 0.2;
        agg_weight = 0.8
    elif risk_profile == 2:
        min_ret = 8;
        filename = 'Moderately Conservative';
        acwi_weight = 0.2;
        agg_weight = 0.8
    elif risk_profile == 3:
        min_ret = 12;
        filename = 'Moderate';
        acwi_weight = 0.6;
        agg_weight = 0.4
    elif risk_profile == 4:
        min_ret = 15;
        filename = 'Moderately Aggressive';
        acwi_weight = 0.8;
        agg_weight = 0.2
    elif risk_profile == 5:
        min_ret = 20;
        filename = 'Aggressive';
        acwi_weight = 0.8;
        agg_weight = 0.2

    # Conservative: get_large_cap, get_corporate_bonds, get_gold, get_gov_short_term, get_ultra_term_bonds
    if risk_profile == 1:

        # old categories
        """
        ####### Corporate Bonds #########
        finalportfoliotickers.extend(get_category_data(listticker=get_corporate_bonds(data=data),
                                                           number_of_years=number_of_years,minimum_target_return=0,
                                                           number_of_simulations=number_of_simulations, number_of_tickers_returned=1))
        ####### Gov Ultra ST bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_ultra_term_bonds(data=data),
                                                       number_of_years=number_of_years,minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations, number_of_tickers_returned=1))
        ####### Large Cap Equity #######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gold ###################
        finalportfoliotickers.extend(get_category_data(listticker=get_gold(data=data),
                                                           number_of_years=number_of_years,minimum_target_return=0,
                                                           number_of_simulations=number_of_simulations, number_of_tickers_returned=1))
        ####### Gov ST bonds ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_gov_short_term(data=data),
                                                           number_of_years=number_of_years,minimum_target_return=0,
                                                           number_of_simulations=number_of_simulations, number_of_tickers_returned=1))
        """

        ####### Corporate Bonds #########
        finalportfoliotickers.extend(get_category_data(listticker=get_corporate_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Gov Bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_government_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Total Bonds #######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Commodities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_commodities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

    # Moderately conservative: get_emerging_markets, get_government_bonds, get_int_large_cap, get_corporate_bonds, get_gold, get_large_cap
    elif risk_profile == 2:
        # old categories
        """
        ####### Corporate Bonds #########
        finalportfoliotickers.extend(get_category_data(listticker=get_corporate_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gov bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_government_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Large Cap Equity #######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gold ###################
        finalportfoliotickers.extend(get_category_data(listticker=get_gold(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Emerging markets ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_emerging_markets(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### International Large Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_int_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """

        # New categories
        # get_commodities, get_corporate_bonds, get_emerging_markets,get_global_non_US_equities, \
        #     get_government_bonds, get_large_cap, get_real_estate, get_small_mid, get_total_bonds, get_total_market_equities

        ####### Small-Mid Equities #########
        """
        finalportfoliotickers.extend(get_category_data(listticker=get_small_mid(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """

        ####### Total Bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Large cap ######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Global non-US equities ######
        finalportfoliotickers.extend(get_category_data(listticker=get_global_non_US_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Commodities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_commodities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Total market equities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_market_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

    # Moderate: get_emerging_markets, get_government_bonds, get_int_large_cap, get_corporate_bonds, get_gold, get_large_cap, get_small_cap, get_real_estate
    elif risk_profile == 3:

        """
        ####### Corporate Bonds #########
        finalportfoliotickers.extend(get_category_data(listticker=get_corporate_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gov bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_government_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Large Cap Equity #######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gold ###################
        finalportfoliotickers.extend(get_category_data(listticker=get_gold(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Emerging markets ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_emerging_markets(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### International Large Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_int_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Real Estate ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_real_estate(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Small Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_small_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """
        ####### Small-Mid Equities #########
        """
        finalportfoliotickers.extend(get_category_data(listticker=get_small_mid(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """

        ####### Total Bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Large cap ######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Global non-US equities ######
        finalportfoliotickers.extend(get_category_data(listticker=get_global_non_US_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Commodities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_commodities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Total market equities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_market_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

    # Moderately aggressive: get_emerging_markets, get_government_bonds, get_int_large_cap, get_corporate_bonds, get_large_cap, get_small_cap, get_real_estate, get_gold
    elif risk_profile == 4:

        """
        ####### Corporate Bonds #########
        finalportfoliotickers.extend(get_category_data(listticker=get_corporate_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gold #########
        finalportfoliotickers.extend(get_category_data(listticker=get_gold(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gov bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_government_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Large Cap Equity #######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Emerging markets ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_emerging_markets(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### International Large Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_int_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Real Estate ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_real_estate(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Small Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_small_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """
        ####### Small-Mid Equities #########
        """
        finalportfoliotickers.extend(get_category_data(listticker=get_small_mid(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """

        ####### Total Bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_bonds(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Large cap ######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Global non-US equities ######
        finalportfoliotickers.extend(get_category_data(listticker=get_global_non_US_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Commodities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_commodities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Total market equities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_market_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

    # Aggressive: get_emerging_markets, get_int_large_cap, get_gold, get_large_cap, get_small_cap, get_real_estate
    elif risk_profile == 5:
        # old
        """
        ####### Gold bonds ######
        finalportfoliotickers.extend(get_category_data(listticker=get_gold(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Large Cap Equity #######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Emerging markets ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_emerging_markets(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### International Large Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_int_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Real Estate ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_real_estate(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Small Cap ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_small_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        ####### Gov ST ###########
        finalportfoliotickers.extend(get_category_data(listticker=get_gov_short_term(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """
        ####### Small-Mid Equities #########
        """
        finalportfoliotickers.extend(get_category_data(listticker=get_small_mid(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))
        """

        ####### Large cap ######
        finalportfoliotickers.extend(get_category_data(listticker=get_large_cap(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Global non-US equities ######
        finalportfoliotickers.extend(get_category_data(listticker=get_global_non_US_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

        ####### Total market equities #######
        finalportfoliotickers.extend(get_category_data(listticker=get_total_market_equities(data=data),
                                                       number_of_years=number_of_years, minimum_target_return=0,
                                                       number_of_simulations=number_of_simulations,
                                                       number_of_tickers_returned=1))

    print("Final finalportfoliotickers = {} with type {}".format(finalportfoliotickers, type(finalportfoliotickers)))
    finalportfoliotickers = [i for i in finalportfoliotickers if i]
    print(finalportfoliotickers)

    # finalportfoliotickers = ['CWB', 'SHV', 'QQQ', 'IAU', 'SHY']
    if finalportfoliotickers:
        getdata = get_data(ETF_list=finalportfoliotickers, number_of_years=number_of_years)

        print(getdata)

        output_images_gold, gold_portfolios, tupledata = portfolio_manager2(input_data=getdata,
                                                                            minimum_target_return=min_ret / 100,
                                                                            number_of_simulations=40000,
                                                                            number_of_tickers_returned=len(
                                                                                finalportfoliotickers))

        print("Printing tuple data")
        print(tupledata)

        # get etf returns
        etf_returns = getdata.pct_change()
        etf_returns = etf_returns.dropna()
        etf_returns = etf_returns.rename(columns={col: col.split(' ')[0] for col in etf_returns.columns})
        print(etf_returns)

        # multiply weights with returns
        weighted_list = []
        for column in etf_returns:
            columnSeriesObj = etf_returns[column]
            # print('Column Name : ', column)
            # print('Column Contents : ', columnSeriesObj.values)
            for tuple in tupledata:
                if tuple[0] == column:
                    etf_returns['{} Weighted'.format(tuple[0])] = tuple[1] * etf_returns[column]
                    weighted_list.append('{} Weighted'.format(tuple[0]))
        # add weighted returns
        etf_returns['Sum'] = etf_returns[weighted_list].sum(axis=1)

        # plotting returns
        plt.hist(etf_returns.Sum, bins=40)
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('Portfolio returns chart.png')

        # var calculation from Sum column
        tempdf = etf_returns
        tempdf.sort_values('Sum', inplace=True, ascending=True)
        var90 = tempdf['Sum'].quantile(0.1)
        print(tabulate([['90%', var90]], headers=['Confidence Level', 'Value at Risk']))

        # create SPY and ACWI-AGG benchmark and write to csv
        finalmergeddf = create_benchmark(etf_returns=etf_returns, startdate=startdate, enddate=enddate,
                                         agg_weight=agg_weight, acwi_weight=acwi_weight)
        print("Printing finalmergeddf")
        print(finalmergeddf)
        filepath = os.path.join(os.getcwd(), 'Results')
        finalmergeddf.to_csv('{}.csv'.format(filename))
        return finalmergeddf
        # sharpe_tickers1 = finalportfoliotickers


def get_category(data, ticker):
    return data.loc[data['Ticker'] == ticker, 'Segment'].iloc[0]


if __name__ == '__main__':

    dataset = pd.read_csv('NewETFdata.csv', header=0,
                          encoding='unicode_escape')

    data = screen_etfs(dataset=dataset)
    print(data)
    print("After preliminary screening, we have {} tickers: {}".format(len(data), data))

    """
    # tickers to get categories for
    ticker_list1 = ['CWB', 'BIL', 'GSY', 'IAU']
    ticker_list2 = ['MINT', 'VONG', 'QCLN', 'IAU', 'IYY']
    ticker_list3 = ['MINT', 'MGK', 'ARKK', 'GLD', 'MOAT']
    ticker_list4 = ['LMBS', 'VONG', 'ICLN', 'GLD', 'IUSG']
    ticker_list5 = ['QQQ', 'ARKK', 'MTUM']
    all_ticker_list = [ticker_list1, ticker_list2, ticker_list3, ticker_list4, ticker_list5]
    for li in all_ticker_list:
        for tick in li:
            print(tick, get_category(data, tick))
        print("\n")
    """

    risk_profile = 5
    number_of_years = 5
    number_of_simulations = 25000
    finalportfoliotickers = []
    """
    for risk_profile in range(1,6):
        try:
            simulate(data=data, risk_profile=risk_profile, number_of_years=number_of_years, number_of_simulations=number_of_simulations, finalportfoliotickers=finalportfoliotickers)
        except Exception as e:
            print("Problem with generating portfolio {}.".format(str(risk_profile)))
    """
    # Setting the dates and weights for acwi and agg
    Current_Date = dt.datetime.today()
    Historical_Date = dt.datetime.now() - dt.timedelta(days=number_of_years * 365)
    startdate = Historical_Date.strftime("%Y-%m-%d")
    enddate = Current_Date.strftime("%Y-%m-%d")

    if risk_profile == 1:
        min_ret = 5; filename = 'Conservative'; acwi_weight = 0.2; agg_weight = 0.8
    elif risk_profile == 2:
        min_ret = 8; filename = 'Moderately Conservative'; acwi_weight = 0.2; agg_weight = 0.8
    elif risk_profile == 3:
        min_ret = 12; filename = 'Moderate'; acwi_weight = 0.6; agg_weight = 0.4
    elif risk_profile == 4:
        min_ret = 15; filename = 'Moderately Aggressive'; acwi_weight = 0.8; agg_weight = 0.2
    elif risk_profile == 5:
        min_ret = 20; filename = 'Aggressive'; acwi_weight = 0.8; agg_weight = 0.2

    sharpe_tickers1 = ['VONG', 'ONEQ', 'ARKK']
    # sortino_tickers1 = ['CWB','BIL','IVW','IAU','SCHO']
    sharpe_weights1 = np.array([0.313919, 0.336196, 0.349885])
    # sortino_weights1 = np.array([0.021912, 0.318732, 0.067047, 0.136778, 0.455532])

    sharpedata = get_data(ETF_list=sharpe_tickers1, number_of_years=5)
    # sortinodata = get_data(ETF_list=sortino_tickers1, number_of_years=5)
    print(sharpedata)
    # print(sortinodata)
    sharpereturns = sharpedata.pct_change()
    sharpereturns = sharpereturns.dropna()

    # sortinoreturns = sortinodata.pct_change()
    # sortinoreturns = sortinoreturns.dropna()

    sharpe_daily_portfolio_return = sharpereturns * sharpe_weights1
    sum_sharpe = sharpe_daily_portfolio_return.sum(axis=1)

    # sortino_daily_portfolio_return = sortinoreturns*sortino_weights1
    # sum_sortino = sortino_daily_portfolio_return.sum(axis=1)

    print("Printing now:")
    print(sum_sharpe)
    print(type(sum_sharpe), sum_sharpe.shape)
    # print(sum_sortino)
    sum_sharpe = sum_sharpe.to_frame()
    print(type(sum_sharpe), sum_sharpe.shape)
    # final_data = pd.concat([sum_sharpe, sum_sortino], axis=1)
    finalmergeddf = create_benchmark(etf_returns=sum_sharpe, startdate=startdate, enddate=enddate,
                                     agg_weight=agg_weight, acwi_weight=acwi_weight)
    print("Finalmergeddf:")
    print(finalmergeddf)
    finalmergeddf.to_csv('{} portfolio performance.csv'.format(filename))

    """
    1. Backtesting module
    2. Risk asses
    3. Different risk metrics
    4. Neural network with different inputs
    """

    # MAXIMUM SHARPE RATIO USING MINIMIZE ALGO

def get_ret_vol_sr(weights, log_return):
    weights = np.array(weights)
    ret = np.sum(log_return.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights, log_return):
    return get_ret_vol_sr(weights, log_return)[2] * -1


# check allocation sums to 1
def check_sum(weights):
    return np.sum(weights) - 1


def get_Sharpe():
    dataset = pd.read_csv('NewETFdata.csv', header=0,
                          encoding='unicode_escape')

    data = screen_etfs(dataset=dataset)

    new_data = get_government_bonds(data)

    stocks = get_data(new_data)
    stocks.pct_change(1).mean()

    log_return = np.log(stocks / stocks.shift(1))
    test_tup = (0, 1)
    bounds = ((test_tup,) * len(log_return.columns))
    cons = ({'type': 'eq', 'fun': check_sum})

    init_guess = [1 / len(log_return.columns)] * len(log_return.columns)

    opt_results = minimize(neg_sharpe, init_guess, bounds=bounds, method='SLSQP', constraints=cons,
                           args=(log_return))

    arr = get_ret_vol_sr(opt_results.x, log_return)
    maximum_Value = arr[-1]

    return maximum_Value