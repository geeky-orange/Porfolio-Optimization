"""
etf_categorize.py contains methods for categorizing ETFs based on Segment
"""


#1
def get_commodities(data):
    input_1 = "Commodities"

    series = data[data['Segment'].str.contains(input_1)]
    commodities = list(series['Ticker'])
    return commodities


#2
def get_global_non_US_equities(data):
    input_1 = "Equity"

    contains = ["Developed", "China", "Australia", "Russia", "South Korea", "Switzerland", "Taiwan", "U.K.", "Global"]
    not_contain = ": U.S. -"
    dev_markets = "Equity: Developed Markets Ex-U.S. - Total Market"
    # contains Equity and one of the string from "contains" array and does not contain ": U.S. -"
    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains('|'.join(contains)) & ~data['Segment'].str.contains(not_contain) & ~data['Segment'].str.contains(dev_markets)]
    global_equities = list(series['Ticker'])

    return global_equities


#3
def get_emerging_markets(data):
    input_1 = "Equity"
    input_2 = "Emerging Markets"
    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    emerging_market_equities = list(series['Ticker'])

    return emerging_market_equities


#4
def get_large_cap(data):
    input_1 = "Equity"
    input_2 = "Large Cap"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds


#5
def get_small_mid(data):
    input_1 = "Equity"
    input_2 = "Small Cap"
    input_3 = "Mid Cap"
    series = data[(data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)) | (data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_3))]
    real_estate_equities = list(series['Ticker'])

    return real_estate_equities

#6
def get_total_market_equities(data):
    input_1 = "Equity"
    input_2 = ": U.S. - Total Market"
    # , ": U.S. Aerospace & Defense", ": U.S. Banks", ": U.S. Basic Materials", ": U.S. Biotech", ": U.S. Consumer", ": U.S. Energy", ": U.S. Financial", ": U.S. Health", ": U.S. Homebuilding", ": U.S. Industrials", ": U.S. Internet", ": U.S. Technology", ": U.S. Telecommunications", ": U.S. Transportation", ": U.S. Utilities",
    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds

#7
def get_real_estate(data):
    input_1 = "Equity"
    input_2 = "Real Estate"
    input_3 = "REITs"
    series = data[(data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)) | (data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_3))]
    real_estate_equities = list(series['Ticker'])

    return real_estate_equities


#8
def get_total_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Global"
    input_3 = "U.S. - Broad Market"

    series = data[(data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)) | (data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_3))]
    stbonds = list(series['Ticker'])

    return stbonds


#9
def get_corporate_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Corporate"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds


#10
def get_government_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Government"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds

#
# 11
def get_int_dev_markets(data):
    input_1 = "Equity: Developed Markets Ex-U.S. - Total Market"

    series = data[data['Segment'].str.contains(input_1)]
    stbonds = list(series['Ticker'])

    return stbonds


