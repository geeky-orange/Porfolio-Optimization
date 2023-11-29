import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime
from itertools import combinations
import multiprocessing
from multiprocessing import Pool, Manager, Value


def screen_etfs(dataset):
    # Dataset is now stored in a Pandas Dataframe
    accepted_grading = ['A', 'B']
    # return_volatility_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-']
    invalidIndex = dataset[(dataset['Expense Ratio'] == '--') | (dataset['AUM'] == '--')].index

    dataset.drop(invalidIndex, inplace=True)

    # dataset[invalidIndex2] = '0'
    # dataset[invalidIndex2] = dataset[invalidIndex2].astype(float)
    # dataset[invalidIndex3] = '0%'
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

        if dataset["Dividend"][ind] == "--":
            dataset.loc[ind, "Dividend"] = "0%"
        # print("Dividend")

        if dataset['P/E'][ind] == "--":
            dataset.loc[ind, 'P/E'] = "0"
            # print("P/E")

        if dataset['P/B'][ind] == "--":
            dataset.loc[ind, 'P/B'] = "0"
            # print("P/B")

        if dataset['1 Month'][ind] == "--":
            dataset.loc[ind, '1 Month'] = "0%"
            # print("1 Month")

        if dataset['3 Month'][ind] == "--":
            dataset.loc[ind, '3 Month'] = "0%"
            # print("3 Month")

        if dataset['YTD'][ind] == "--":
            dataset.loc[ind, "YTD"] = "0%"

        if dataset['1 Year'][ind] == "--":
            dataset.loc[ind, '1 Year'] = "0%"
            # print("5 Years")

        if dataset['3 Years'][ind] == "--":
            dataset.loc[ind, '3 Years'] = "0%"
            # print("5 Years")

        if dataset['5 Years'][ind] == "--":
            dataset.loc[ind, '5 Years'] = "0%"
            # print("5 Years")

        if dataset['10 Years'][ind] == "--":
            dataset.loc[ind, '10 Years'] = "0%"
            # print("10 Years")

    # remove inverse and leveraged
    for ind in dataset.index:
        if ('Inverse' in dataset['Segment'][ind]) or ('Leveraged' in dataset['Segment'][ind]):
            dataset = dataset.drop([ind])

    dataset['Expense Ratio'] = dataset['Expense Ratio'].str.replace('%', '')
    dataset['Expense Ratio'] = dataset['Expense Ratio'].astype(float)

    dataset['P/E'] = dataset['P/E'].str.replace(',', '')
    dataset['P/E'] = dataset['P/E'].astype(float)

    dataset['P/B'] = dataset['P/B'].str.replace(',', '')
    dataset['P/B'] = dataset['P/B'].astype(float)

    dataset['1 Month'] = dataset['1 Month'].str.replace('%', '')
    dataset['1 Month'] = dataset['1 Month'].astype(float)

    dataset['3 Month'] = dataset['3 Month'].str.replace('%', '')
    dataset['3 Month'] = dataset['3 Month'].astype(float)

    dataset['YTD'] = dataset['YTD'].str.replace('%', '')
    dataset['YTD'] = dataset['YTD'].astype(float)

    dataset['1 Year'] = dataset['1 Year'].str.replace('%', '')
    dataset['1 Year'] = dataset['1 Year'].astype(float)

    dataset['3 Years'] = dataset['3 Years'].str.replace('%', '')
    dataset['3 Years'] = dataset['3 Years'].astype(float)

    dataset['5 Years'] = dataset['5 Years'].str.replace('%', '')
    dataset['5 Years'] = dataset['5 Years'].astype(float)

    dataset['10 Years'] = dataset['10 Years'].str.replace('%', '')
    dataset['10 Years'] = dataset['10 Years'].astype(float)

    dataset['Dividend'] = dataset['Dividend'].str.replace('%', '')
    dataset['Dividend'] = dataset['Dividend'].astype(float)

    # rules
    AUM_test = dataset['AUM'].astype(float) >= 1000000000
    ER_test = dataset['Expense Ratio'] <= 0.75
    grade_test = dataset['Grade'].isin(accepted_grading)
    print(len(dataset))
    dataset = dataset[grade_test]
    dataset = dataset[AUM_test]
    dataset = dataset[ER_test]

    return dataset


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(mean_daily_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1


def check_sum(weights):
    return np.sum(weights) - 1


def get_results(data, namespace, final_weight):
    components = list(data.columns.values)

    # convert daily stock prices into daily returns
    returns = data.pct_change()
    returns = returns.dropna()

    # a copy of the returns to keep it constant during adjustment
    initial_returns = data.pct_change().dropna()
    total_return = data.div(data.shift(1)).dropna()
    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # cov_matrix_sortino = returns
    # set number of runs of random portfolio weights

    """    
     set up array to hold results
     '5' is to store returns, standard deviation, sharpe ratio, value at risk(default = 90%) an cvar(default = 90)
     len(components) is for storing weights for each component in every portfolio
    """

    # no of columns in the results -- change if more data is needed
    number_of_columns = 4
    level = None
    results = np.zeros((number_of_columns + len(components), len(final_weight)))

    for index, weights in enumerate(final_weight):
        # select random weights for portfolio holdings
        weights = np.array(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        daily_portfolio_return = initial_returns * weights
        total_daily_portfolio_return = total_return * weights
        total_sum_col = total_daily_portfolio_return.sum(axis=1)
        cum_return = total_sum_col.cumprod().dropna()
        index_j = np.argmax(np.maximum.accumulate(cum_return) - cum_return)  # end
        index_i = np.argmax(cum_return[:index_j])  # start
        drawdown = (cum_return[index_j] - cum_return[index_i]) / cum_return[index_i]

        # print(data.iloc[index_i+1:index_j+2, :])

        # print(drawdown)

        # add this sum col to results
        # sum_col = daily_portfolio_return.sum(axis=1)

        # adding the sum columns and sum pct change in the results DF
        # returns["sum"] = sum_col
        # returns["sum_pct_change"] = returns["sum"].pct_change()

        # filter sum_col to get only negative values and use that to calculate semi-deviation/downside risk

        # confidence levels for the VAR & CVAR -- if needed make a list of confidence levels to have 90,95,99
        # confidence = 90
        # value_loc_for_percentile = round(len(returns) * (1 - (confidence / 100)))
        # sorted_returns = returns.sort_values(by=["sum_pct_change"])
        # var = sorted_returns.iloc[value_loc_for_percentile, len(sorted_returns.columns) - 1]
        # cvar = sorted_returns["sum_pct_change"].head(value_loc_for_percentile).mean(axis=0)

        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # store results in results array
        results[0, index] = portfolio_return
        results[1, index] = portfolio_std_dev

        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, index] = results[0, index] / results[1, index]

        results[3, index] = drawdown
        # results[5, index] = cvar

        for j in range(len(weights)):
            # change the 3 in the following line to 4 and add 'sortino' to columns
            results[j + 4, index] = weights[j]
    cols = ['ret', 'stdev', 'sharpe', 'drawdown']
    # cols = ['ret', 'stdev', 'sharpe', 'sortino', 'var', 'cvar']
    for stock in components:
        cols.append(stock)

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=cols)
    # print(results_frame['cvar'])
    # print(results_frame)
    if not results_frame[results_frame['sharpe'] > 1].empty:
        results_frame = results_frame[results_frame['sharpe'] >= 1]
        if not results_frame[results_frame['ret'] >= 0.22].empty:
            results_frame = results_frame[results_frame['ret'] >= 0.22]
            if not results_frame[results_frame["drawdown"] < -0.2].empty:
                results_frame = results_frame[results_frame["drawdown"] < -0.2]
                level = "Level5"

        elif not results_frame[results_frame['ret'] >= 0.15].empty:
            results_frame = results_frame[results_frame['ret'] >= 0.15]
            if not results_frame[results_frame["drawdown"] >= -0.2].empty:
                results_frame = results_frame[results_frame["drawdown"] >= -0.2]
                level = "level4"

        elif not results_frame[results_frame['ret'] >= 0.12].empty:
            results_frame = results_frame[results_frame['ret'] >= 0.12]
            if not results_frame[results_frame["drawdown"] >= -0.15].empty:
                results_frame = results_frame[results_frame["drawdown"] >= -0.15]
                level = "Level3"

        elif not results_frame[results_frame['ret'] >= 0.08].empty:
            results_frame = results_frame[results_frame['ret'] >= 0.08]
            if not results_frame[results_frame["drawdown"] >= -0.1].empty:
                results_frame = results_frame[results_frame["drawdown"] >= -0.1]
                level = "Level2"

        elif not results_frame[results_frame['ret'] >= 0.04].empty:
            results_frame = results_frame[results_frame['ret'] >= 0.04]
            if not results_frame[results_frame["drawdown"] >= -0.05].empty:
                results_frame = results_frame[results_frame["drawdown"] >= -0.05]
                level = "Level1"

    if level is not None:
        column = namespace.counter
        # print(column)
        weights = results_frame.loc[int((results_frame['sharpe'].idxmax()))][4:7].values
        returns = results_frame.loc[int((results_frame['sharpe'].idxmax())), 'ret']
        sharpe = results_frame['sharpe'].max()
        ori_df = namespace.df

        ori_df.loc[column] = [components[0], components[1], components[2], weights[0], weights[1], weights[2],
                              returns, sharpe, drawdown, level]
        namespace.df = ori_df
        column += 1
        namespace.counter = column
        # print(namespace.df)


if __name__ == "__main__":
    # print(multiprocessing.cpu_count()) 12
    # /? Create the ticker list
    """init_guess = [1 / 3] * 3
    test_tup = (0, 1)
    bounds = ((test_tup,) * 3)
    daily_data = pd.read_csv(r"Daily_Data_5_Years_Benchmark.csv", index_col="Date", parse_dates=True)
    cons = ({'type': 'eq', 'fun': check_sum})

    df1 = pd.read_csv("NewETFData2.csv")
    df2 = pd.read_csv("NewETFdata.csv")
    df3 = pd.read_csv("NewETFDataRegion.csv")

    df1.drop(columns=['Ticker', 'Name', 'As Of Date'], inplace=True)

    result = pd.concat([df2, df1], axis=1)
    result = pd.merge(result, df3, on="Ticker")

    result = screen_etfs(result)

    # add the asset class
    asset_class = []

    for index, row in result.iterrows():
        if "Equity" in row["Segment"]:
            asset_class.append("Equity")
        elif "Fixed Income" in row["Segment"]:
            asset_class.append("Fixed Income")
        elif "Commodities" in row["Segment"]:
            asset_class.append("Commodities")
        else:
            asset_class.append("Null")

    result["Asset Class"] = asset_class

    result['Grade'] = result['Grade'].str.replace('A', '2')  # not sure 2 or 1 to be used
    result['Grade'] = result['Grade'].str.replace('B', '1')  # not sure 1 or 0 to be used"""

    # /? read the result.csv
    result = pd.read_csv("Result.csv")
    daily_data = pd.read_csv(r"Daily_Data_5_Years_Benchmark.csv", index_col="Date", parse_dates=True)
    ticker_list = result["Ticker"].values
    new_ticker_list = []

    for ticker in ticker_list:
        new_ticker_list.append("{} Adj Close".format(ticker))

    # equity_ticker_list = list(set(new_ticker_list).intersection(set(daily_data.columns.values)))
    # equity_data = daily_data.loc[:, equity_ticker_list]
    # print(equity_data)
    daily_returns = daily_data.pct_change()
    # print(equity_returns)
    daily_returns = daily_returns.dropna()
    # print(equity_returns)

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = daily_returns.mean()
    mean_daily_returns.sort_values(inplace=True, ascending=False)  # 333
    # print(mean_daily_returns)
    # screen the return greater than 0
    mean_daily_returns = mean_daily_returns[mean_daily_returns > 0]  # 324
    # print(mean_daily_returns)
    # /? random simulation

    new_ticker_list = mean_daily_returns.index
    # print(new_ticker_list)
    print(new_ticker_list)
    weight_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
    final_weight = []

    """candidate_ticker1 = []
    candidate_ticker2 = []
    candidate_ticker3 = []

    init_guess_weight1 = []
    init_guess_weight2 = []
    init_guess_weight3 = []

    candidate_return = []
    candidate_sharpe = []
    candidate_level = []
    candidate_drawdown = []
    start = datetime.datetime.now()
    result = pd.DataFrame()"""

    for weight_a in weight_list:
        for weight_b in weight_list:
            for weight_c in weight_list:
                if weight_a + weight_b + weight_c == 1:
                    final_weight.append([weight_a, weight_b, weight_c])

    # print(final_weight)
    # print(len(final_weight))
    f = combinations(new_ticker_list, 3)

    result_df = pd.DataFrame(
        columns=["Ticker1", "Ticker2", "Ticker3", "Weight1", "Weight2", "Weight3", "Return", "Sharpe", "Drawdown",
                 "Level"])

    counter = 0
    save_count = 0
    iteration = 0
    manager = Manager()
    ns = manager.Namespace()
    ns.df = result_df
    ns.counter = counter

    while True:
        try:
            print("----start iteration{}----".format(iteration))
            pool = multiprocessing.Pool(processes=10)

            for i in range(10):
                tickers = list(next(f))
                pool.apply(get_results, (daily_data[tickers], ns, final_weight))

            pool.close()
            pool.join()

            print("----end iteration{}----".format(iteration))
            iteration += 1
            print(ns.df)

            if iteration % 1000 == 0:
                print("Saving....")
                ns.df.to_csv("Test{}.csv".format(save_count))
                save_count += 1
                new_result_df = pd.DataFrame(
                    columns=["Ticker1", "Ticker2", "Ticker3", "Weight1", "Weight2", "Weight3", "Return", "Sharpe",
                             "Drawdown",
                             "Level"])
                ns.df = new_result_df
                new_counter = 0
                ns.counter = new_counter

        except StopIteration:
            print("Finish!")
            break

    """for i in range(len(new_ticker_list) - 2):
        for j in range(i + 1, len(new_ticker_list) - 1):
            for k in range(j + 1, len(new_ticker_list)):
                temp_ticker1 = new_ticker_list[i]
                temp_ticker2 = new_ticker_list[j]
                temp_ticker3 = new_ticker_list[k]

                selected_ticker_list = [temp_ticker1, temp_ticker2, temp_ticker3]
                temp_df = daily_data[selected_ticker_list]

                cal_result = get_results(temp_df)

                if cal_result is not None:
                    level, drawdown, returns, sharpe, asset_weights = cal_result
                    print(level)
                    candidate_ticker1.append(selected_ticker_list[0])
                    candidate_ticker2.append(selected_ticker_list[1])
                    candidate_ticker3.append(selected_ticker_list[2])

                    init_guess_weight1.append(asset_weights[0])
                    init_guess_weight2.append(asset_weights[1])
                    init_guess_weight3.append(asset_weights[2])

                    candidate_drawdown.append(drawdown)
                    candidate_return.append(returns)
                    candidate_sharpe.append(sharpe)
                    candidate_level.append(level)

                if datetime.datetime.now() >= start + datetime.timedelta(hours=1):
                    print("Saving")
                    start = datetime.datetime.now()

                    result["Ticker1"] = candidate_ticker1
                    result["Ticker2"] = candidate_ticker2
                    result["Ticker3"] = candidate_ticker3

                    result["Weight1"] = init_guess_weight1
                    result["Weight2"] = init_guess_weight2
                    result["Weight3"] = init_guess_weight3

                    result["Return"] = candidate_return
                    result["Sharpe"] = candidate_sharpe
                    result["Drawdown"] = candidate_drawdown
                    result["Level"] = candidate_level
                    result.to_csv("Candidate Tickers and Level{}.csv".format(count))

                    result = pd.DataFrame()
                    candidate_ticker1 = []
                    candidate_ticker2 = []
                    candidate_ticker3 = []

                    init_guess_weight1 = []
                    init_guess_weight2 = []
                    init_guess_weight3 = []

                    candidate_level = []
                    candidate_return = []
                    candidate_sharpe = []
                    candidate_drawdown = []

                    count += 1"""
