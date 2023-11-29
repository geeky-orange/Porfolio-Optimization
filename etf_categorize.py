"""
etf_categorize.py contains methods for categorizing ETFs based on Segment
"""

#Categories:
#1. Short Term Government Bonds +
#2. Ultra Short Term Government Bonds +
#3. Ultra Short Term Corporate Bonds -
#4. All government bonds +
#5. All corporate bonds +
#6. Large Cap stocks +
#7. Small Cap stocks +
#8. International Large Cap
#9. Emerging markets equities +
#10. Real estate stocks +
#11. Gold ETFs +


def get_gov_short_term(data):
    input_1 = "Government"
    input_2 = "Short-Term"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_ultra_term_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Ultra"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_large_cap(data):
    input_1 = "Equity"
    input_2 = "Large Cap"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_small_cap(data):
    input_1 = "Equity"
    input_2 = "Small Cap"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_real_estate(data):
    input_1 = "Equity"
    input_2 = "Real Estate"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_government_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Government"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_corporate_bonds(data):
    input_1 = "Fixed Income"
    input_2 = "Corporate"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_emerging_markets(data):
    input_1 = "Equity"
    input_2 = "Emerging Markets"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_gold(data):
    input_1 = "Commodities"
    input_2 = "Gold"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds




def get_int_large_cap(data):
    input_1 = "Equity"
    input_2 = "Developed Markets Ex-U.S. - Total Market"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    int_lc = list(series['Ticker'])

    input_3 = "Global"
    input_4 = "Total Market"
    series2 = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_3) & data['Segment'].str.contains(input_4)]
    total_lc = list(series2['Ticker'])

    final_list = int_lc + total_lc
    return final_list




def get_ultra_term_bonds(data):
    input_1 = "Government"
    input_2 = "Ultra-Short"

    series = data[data['Segment'].str.contains(input_1) & data['Segment'].str.contains(input_2)]
    stbonds = list(series['Ticker'])

    return stbonds


