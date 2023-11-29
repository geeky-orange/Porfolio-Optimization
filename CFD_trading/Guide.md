Get the epic of market index (could be Chinese):

```python
ig_service.search_markets(search_term="DAX")
```

A table of common epic

|     Index      |         EPIC         |
| :------------: | :------------------: |
|   Hang Seng    | IX.D.HANGSENG.IFA.IP |
|   Dow Jones    |   IX.D.DOW.IFA.IP    |
|     Nasdaq     |  IX.D.NASDAQ.IFA.IP  |
|      S&P       |  IX.D.SPTRD.IFA.IP   |
|      DAX       |   IX.D.DAX.IFA.IP    |
|   Nikkei 225   |  IX.D.NIKKEI.IFA.IP  |
| FTSE China A50 |  IX.D.XINHUA.IFA.IP  |

Trading Strategy:

1. Decision of Long or Short

   By technical indicators analysis. 75% of the same action indicators decides to buy or sell.

   | Name                | Action             |
   | ------------------- | ------------------ |
   | RSI(14)             | Buy/ Sell/ Neutral |
   | STOCH(9,6)          |                    |
   | STOCHRSI(14)        |                    |
   | MACD(12,26)         | Buy/ Sell/ Neutral |
   | ADX(14)             | Buy/ Sell/ Neutral |
   | Williams %R         |                    |
   | CCI(14)             | Buy/ Sell/ Neutral |
   | ATR(14)             |                    |
   | Highs/Lows(14)      | Buy/ Sell/ Neutral |
   | Ultimate Oscillator | Buy/ Sell/ Neutral |
   | ROC                 | Buy/ Sell/ Neutral |
   | Bull/Bear Power     | Buy/ Sell/ Neutral |

   

2. Decide the sell point (long) or buy point (short)

   By moving average

3. Questions:

   1. use bid or ask price
   2. 
