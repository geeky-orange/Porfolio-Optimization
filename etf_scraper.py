"""
etf_scraper.py contains methods for scraping data from etf.com
"""
import numpy as np
from selenium import webdriver
from getpass import getpass
import re
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
import csv
from requests import get
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from collections import Counter
from selenium.webdriver.common.action_chains import ActionChains
# import yfinance as yf
import datetime as dt


# get tickers from a single page
def page_tickers(driver, tickers):
    # get elements
    elements = driver.find_elements_by_class_name('linkTickerName')
    for element in elements:
        tickers.append(element.text)
        print(element.text)


# get tickers from all pages
def get_all_tickers():
    # list to store tickers
    tickers = []

    # Open chrome and go to link
    driver = webdriver.Chrome(r'chromedriver')
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    options=chrome_options
    options = Options()
    options.headless = True
    """
    driver.get('https://www.etf.com/etfanalytics/etf-finder')
    driver.maximize_window()
    time.sleep(10)

    # display hundred ETFs per page
    displayhundred = WebDriverWait(driver, 20).until(
        ec.presence_of_all_elements_located((By.CLASS_NAME, "inactiveResult")))
    hundred = None
    for x in displayhundred:
        if (x.text) == "100":
            hundred = x
    driver.execute_script("arguments[0].click();", hundred)
    time.sleep(5)

    # get tickers on the first page
    page_tickers(driver, tickers)

    # get total amount of pages to be scraped
    totalpages = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "totalPages")))
    totalpages = re.sub("[^0-9]", "", str(totalpages.text))
    print(totalpages)
    totalpages = int(totalpages) - 1

    # go to every page and get ETFs
    for i in range(0, totalpages):
        nextbutton = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "nextPage")))
        driver.execute_script("arguments[0].click();", nextbutton)
        time.sleep(10)

        page_tickers(driver, tickers)

    return tickers


# read tickers from .txt file
def read_tickers():
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f]
        return tickers


"""
scrapes the Ticker,Name,Segment,Issuer,Expense Ratio,AUM,Grade of all ETFs from https://www.etf.com/etfanalytics/etf-finder
and saves it into ETFdata.csv
"""


def get_ETF():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    # https://www.etf.com/etfanalytics/etf-finder
    driver = webdriver.Chrome(r'chromedriver',
                              options=chrome_options)
    driver.get('https://www.etf.com/etfanalytics/etf-finder')
    driver.maximize_window()
    time.sleep(10)
    # input("Enter etf.com manually!")

    displayhundred = driver.find_elements_by_class_name("inactiveResult")
    # https://www.etf.com/etfanalytics/etf-finder/?sfilters=eyJlNTkyODA4MjYwZTIiOnsiZ3RlIjoxMDAwMDAwLCJsdGUiOjMwMDAwMDAwMDAwLCJvcmRlciI6MH19
    # display hundred ETFs per page
    # displayhundred = WebDriverWait(driver, 20).until(ec.presence_of_all_elements_located((By.CLASS_NAME, "inactiveResult")))
    hundred = None
    for x in displayhundred:
        if x.text == "100":
            hundred = x
    driver.execute_script("arguments[0].click();", hundred)
    time.sleep(5)

    # input("Waiting for page change:")

    # final header and main_rows for df
    main_headers = []
    main_rows = []
    performance_header = []
    performance_rows = []
    grades = []
    dividends = []
    pe = []
    pb = []

    # find the main table on the first page and add to array
    finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
    # print(finderTable)
    for row in finderTable.find_elements_by_xpath(".//tr"):
        for header in row.find_elements_by_xpath('./th'):
            # print(header.text)
            main_headers.append(header.text)
        temp_row = []
        for cell in row.find_elements_by_xpath('./td'):
            # print(cell.text)
            temp_row.append(cell.text)
        if temp_row:
            main_rows.append(temp_row)

    # click on analysis
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
    tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
        .until(expected_conditions.presence_of_element_located(
        (By.XPATH, "//*[@id='table-tabs']/li[4]/span")))  # /? is this number represent the tab
    driver.execute_script("arguments[0].click();", tabletabs)

    # find the main table on the page again and get grade for each one
    finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
    for row in finderTable.find_elements_by_xpath(".//tr"):
        count = 0
        for cell in row.find_elements_by_xpath('./td'):
            if count == 3:
                grades.append(cell.text)
            count = count + 1

    # click on performance
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
    tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
        .until(expected_conditions.presence_of_element_located((By.XPATH, "//*[@id='table-tabs']/li[2]/span")))
    driver.execute_script("arguments[0].click();", tabletabs)

    # find the main table on the page again and get 1 Month, 3 Month, YTD, 1 Year, 5 Years, 10 Years, for each one
    finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
    # print(finderTable)
    for row in finderTable.find_elements_by_xpath(".//tr"):
        for header in row.find_elements_by_xpath('./th'):
            print(header.text)
            performance_header.append(header.text)
        temp_row = []
        for cell in row.find_elements_by_xpath('./td'):
            # print(cell.text)
            temp_row.append(cell.text)
        if temp_row:
            performance_rows.append(temp_row)

    # click on fundamentals
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
    tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
        .until(expected_conditions.presence_of_element_located((By.XPATH, "//*[@id='table-tabs']/li[5]/span")))
    driver.execute_script("arguments[0].click();", tabletabs)

    # find the main table on the page again and get dividend, p/e, p/r for each one
    finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
    for row in finderTable.find_elements_by_xpath(".//tr"):
        count = 0
        for cell in row.find_elements_by_xpath('./td'):
            if count == 2:
                dividends.append(cell.text)
            if count == 3:
                pe.append(cell.text)
            if count == 4:
                pb.append(cell.text)

            count = count + 1

    # get total amount of pages to be scraped
    totalpages = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "totalPages")))
    totalpages = re.sub("[^0-9]", "", str(totalpages.text))
    print(totalpages)
    totalpages = int(totalpages) - 1

    # go to every page and get ETFs
    for i in range(0, totalpages):

        # click on fund basics
        """
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        fundbasics = WebDriverWait(driver, 50, ignored_exceptions=ignored_exceptions) \
            .until(expected_conditions.presence_of_element_located((By.XPATH, "/html/body/div[7]/section/div/div[3]/section/div/div/div/div/div[2]/section[2]/div[2]/ul/li[1]/span")))
        driver.execute_script("arguments[0].click();", fundbasics)
        """
        fundbasics = ""
        spans = driver.find_elements_by_tag_name("span")
        for span in spans:
            if span.text == "Fund Basics":
                fundbasics = span
                fundbasics.click()
                break

        # go to next page
        nextbutton = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "nextPage")))
        driver.execute_script("arguments[0].click();", nextbutton)
        time.sleep(10)

        # find the main table on the page
        finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))

        for row in finderTable.find_elements_by_xpath(".//tr"):
            temp_row = []
            for cell in row.find_elements_by_xpath('./td'):
                print(cell.text)
                temp_row.append(cell.text)
            if temp_row:
                main_rows.append(temp_row)

        # click on analysis
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
            .until(expected_conditions.presence_of_element_located((By.XPATH, "//*[@id='table-tabs']/li[4]/span")))
        driver.execute_script("arguments[0].click();", tabletabs)

        # find the main table on the page again and get grade for each one
        finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
        for row in finderTable.find_elements_by_xpath(".//tr"):
            count = 0
            for cell in row.find_elements_by_xpath('./td'):
                if count == 3:
                    grades.append(cell.text)
                count = count + 1

        # click on performance
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
            .until(expected_conditions.presence_of_element_located((By.XPATH, "//*[@id='table-tabs']/li[2]/span")))
        driver.execute_script("arguments[0].click();", tabletabs)

        # find the main table on the page again and get 1 Month, 3 Month, YTD, 1 Year, 5 Years, 10 Years, for each one
        finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))

        for row in finderTable.find_elements_by_xpath(".//tr"):
            temp_row = []
            for cell in row.find_elements_by_xpath('./td'):
                print(cell.text)
                temp_row.append(cell.text)
            if temp_row:
                performance_rows.append(temp_row)

        # click on fundamentals
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        tabletabs = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
            .until(expected_conditions.presence_of_element_located((By.XPATH, "//*[@id='table-tabs']/li[5]/span")))
        driver.execute_script("arguments[0].click();", tabletabs)

        # find the main table on the page again and get dividend, p/e, p/r for each one
        finderTable = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.ID, "finderTable")))
        for row in finderTable.find_elements_by_xpath(".//tr"):
            count = 0
            for cell in row.find_elements_by_xpath('./td'):
                if count == 2:
                    dividends.append(cell.text)
                if count == 3:
                    pe.append(cell.text)
                if count == 4:
                    pb.append(cell.text)

                count = count + 1

    print("grades")
    print(grades[0])
    print("main_rows")
    print(main_rows[0])
    print("Length of main rows", len(main_rows))
    df = pd.DataFrame(main_rows, columns=main_headers)
    df2 = pd.DataFrame(performance_rows,columns=performance_header)
    df['Grade'] = grades
    df['Dividend'] = dividends
    df['P/E'] = pe
    df['P/B'] = pb
    print(df.head())
    df.to_csv(r'NewETFdata.csv', index=False)
    df2.to_csv(r"NewETFData.csv2", index=False)



# get underlying holding data for ticker
def get_ETF_holdings(ticker):
    driver = webdriver.Firefox(executable_path='C:/Users/HP/PycharmProjects/trade_backtest/geckodriver.exe')
    driver.get('https://www.etf.com/{}#overview'.format(ticker))
    driver.maximize_window()
    time.sleep(10)
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)

    # trying different possible places to find holdings table
    try:

        # for IWY
        h4 = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions). \
            until(ec.presence_of_element_located(
            (By.XPATH, "/html/body/div[7]/section/div/div/div[3]/div[1]/div[1]/div[7]/div/h4/span")))
        driver.execute_script("arguments[0].click();", h4)

    except:

        try:
            h4 = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions). \
                until(ec.presence_of_element_located(
                (By.XPATH, "/html/body/div[7]/section/div/div/div[3]/div[4]/div[1]/div[1]/div/h4/span")))
            driver.execute_script("arguments[0].click();", h4)

        except:

            try:
                h4 = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions). \
                    until(ec.presence_of_element_located(
                    (By.XPATH, "/html/body/div[7]/section/div/div/div[3]/div[4]/div[1]/div[2]/div/h4/span")))
                driver.execute_script("arguments[0].click();", h4)
            except:

                h4 = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions). \
                    until(ec.presence_of_element_located(
                    (By.XPATH, "/html/body/div[6]/section/div/div/div[3]/div[4]/div[1]/div[2]/div/h4/span")))
                driver.execute_script("arguments[0].click();", h4)

    main_table = driver.find_elements_by_tag_name('tbody')
    print("Main table")
    print(len(main_table))
    for t in main_table:
        print(type(t))
        print(t)
        print(t.text)

    view_all_box = WebDriverWait(driver, 20).until(
        ec.presence_of_element_located((By.XPATH, "/html/body/div[11]/div[1]/div/div[2]/table/tbody")))
    # print(view_all_box)

    col1 = view_all_box.find_elements_by_class_name("view_all_column1")
    col1 = [i.text for i in col1]
    # print(col1)

    col2 = view_all_box.find_elements_by_class_name("view_all_column2")
    col2 = [float(i.text.strip('%')) / 100 for i in col2]
    # print(col2)

    return dict(zip(col1, col2))


def latest_prices(ticker):
    driver = webdriver.Chrome(r'chromedriver')
    driver.get('https://www.etf.com/{}'.format(ticker))
    driver.maximize_window()
    time.sleep(10)
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
    closing_price = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions) \
        .until(expected_conditions.presence_of_element_located(
        (By.XPATH, "//*[@id='closing-prices-header']/div[1]/div[1]/span[2]")))
    print("{} closing price is {}".format(ticker, closing_price.text))
    return closing_price.text


if __name__ == '__main__':
    """
    sum = 0
    tickerlist = ['CWB', 'SHV', 'QQQ', 'IAU', 'SHY']
    for ticker in tickerlist:
        ticker_price = latest_prices(ticker)
        price = float(ticker_price[1:])
        sum = sum + price

    print(sum)
    
    """
    get_ETF()

    """
    initial_investment = 1000
    sharpe_weights1 = np.array([0.035571, 0.334979, 0.062914, 0.093326, 0.473210])
    tickerlist = ['CWB', 'SHV', 'QQQ', 'IAU', 'SHY']

    ticker_weights = dict(zip(tickerlist, sharpe_weights1))
    print(ticker_weights)

    total = {}
    total = Counter(total)

    for ticker in tickerlist:

        time.sleep(5)

        if ticker == 'IAU':
            holdings = {'Gold': 1.0}

        else:
            holdings = get_ETF_holdings(ticker)

        print("{} is".format(ticker))
        print(holdings)

        # multiply all values of holdings dict with weight
        w = float(ticker_weights.get(ticker))
        holdings.update((x, y * w) for x, y in holdings.items())
        print("Holdings after multiplying weight:")
        print(holdings)
        holdings = Counter(holdings)
        total = total + holdings

    print("Final.")
    print(total)

    d = {}
    for key, value in total.items():
        d[key] = value

    print(d)
    print(sum(d.values()))

    with open('conservative_output.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        for key1, value1 in ticker_weights.items():
            writer.writerow([key1, value1])
        for key, value in d.items():
            writer.writerow([key, value])
    
    """

    """
    time.sleep(10)
    B = get_ETF_holdings('VOO')
    print("VOO is")
    print(B)

    keys_a = set(A.keys())
    keys_b = set(B.keys())
    intersection = keys_a & keys_b
    print("Intersection is")
    print(intersection)

    A = Counter(A)
    B = Counter(B)
    print(A+B)
    #print(get_all_tickers())
    """
    """
    tempdict = {}
    tickerlist = ['IAU', 'MGK', 'XSOE', 'IQLT', 'FREL', 'VBK']
    sharpe_weights1 = np.array([0.445985, 0.471063, 0.015484, 0.032816, 0.005673, 0.028979])
    ticker_weights = dict(zip(tickerlist, sharpe_weights1))
    total = {}
    total = Counter(total)
    driver = webdriver.Firefox(executable_path='C:/Users/HP/PycharmProjects/trade_backtest/geckodriver.exe')

    for ticker in tickerlist:
        print("Ticker is {}".format(ticker))
        w = float(ticker_weights.get(ticker))
        driver.get('https://www.etf.com/{}#overview'.format(ticker))
        driver.maximize_window()
        time.sleep(10)
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)

        try:
            h4 = WebDriverWait(driver, 20, ignored_exceptions=ignored_exceptions). \
                until(ec.presence_of_element_located(
                (By.XPATH, "//*[@id='totalTop10Holdings2']")))

            for x in h4.find_elements_by_xpath('.//ul'):

                for li in x.find_elements_by_xpath('.//li'):
                    span = li.find_elements_by_tag_name('span')
                    tempdict = {span[1].text : float(span[2].text.strip('%'))*w/100}


                    #tempdict.update((x, y * w) for x, y in tempdict.items())
                    print("temp dict")
                    print(tempdict)
                    tempdict = Counter(tempdict)
                    total = total + tempdict
                    print(total)
        except:
            print("Error in getting geographical data for {}".format(ticker))

    print(total)
    d = {}
    for key, value in total.items():
        d[key] = value

    print(d)
    print(sum(d.values()))

    with open('aggressive_geography.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        for key1, value1 in ticker_weights.items():
            writer.writerow([key1, value1])
        for key, value in d.items():
            writer.writerow([key, value])

    """
    # print(h4)
    # print(h4.text)
