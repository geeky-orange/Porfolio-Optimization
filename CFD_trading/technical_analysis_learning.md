1. Useful Links

   | Indicator           | Links                                                        |
   | ------------------- | ------------------------------------------------------------ |
   | RSI(14)             | https://www.investopedia.com/terms/r/rsi.asp                 |
   | STOCH(9,6)          | https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full |
   | STOCHRSI(14)        | https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi |
   | MACD(12,26)         | https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd |
   | ADX(14)             |                                                              |
   | Williams %R         |                                                              |
   | CCI(14)             |                                                              |
   | ATR(14)             |                                                              |
   | Highs/Lows(14)      |                                                              |
   | Ultimate Oscillator |                                                              |
   | ROC                 |                                                              |
   | Bull/Bear Power     |                                                              |
   |                     |                                                              |
   |                     |                                                              |

   

2. Package

   technical analysis:

   https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#

   

3. **Bullish / Bearish Indicators**

   1. Moving Average Convergence Divergence

      Bullish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close > Daily SMA(200,Daily Close)] 
      AND [Yesterday's Daily MACD Line(12,26,9,Daily Close) < Daily MACD Signal(12,26,9,Daily Close)] 
      AND [Daily MACD Line(12,26,9,Daily Close) > Daily MACD Signal(12,26,9,Daily Close)] 
      AND [Daily MACD Line(12,26,9,Daily Close) < 0]
      ```

      Bearish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close < Daily SMA(200,Daily Close)] 
      AND [Yesterday's Daily MACD Line(12,26,9,Daily Close) > Daily MACD Signal(12,26,9,Daily Close)] 
      AND [Daily MACD Line(12,26,9,Daily Close) < Daily MACD Signal(12,26,9,Daily Close)] 
      AND [Daily MACD Line(12,26,9,Daily Close) > 0]
      ```

   2. Relative Strength Index

      Oversold

      

      Overbought

   3. Stochastic Relative Strength Index

      Oversold with uptrend

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 10] 
      
      AND [Daily SMA(10,Daily Close) > Daily SMA(60,Daily Close)] 
      AND [Daily Stoch RSI(14) < 0.1] 
      AND [Daily Close < Daily SMA(10,Daily Close)]
      ```

      Overbought with downtrend

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 10] 
      
      AND [Daily SMA(10,Daily Close) < Daily SMA(60,Daily Close)] 
      AND [Daily Stoch RSI(14) > 0.9] 
      AND [Daily Close > Daily SMA(10,Daily Close)]
      ```

   4. Stochastic

      Bullish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close > Daily SMA(200,Daily Close)] 
      AND [Yesterday's Daily Slow Stoch %K(14,3) < 20] 
      AND [Daily Slow Stoch %K(14,3) > 20]
      ```

      Bearish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close < Daily SMA(200,Daily Close)] 
      AND [Yesterday's Daily Slow Stoch %K(14,3) > 80] 
      AND [Daily Slow Stoch %K(14,3) < 80]
      ```

   5. Bollinger Band

      Bullish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 5] 
      
      AND [Daily Close x Daily Upper BB(20,2.0)] 
      ```

      Bearish

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 5] 
      
      AND [Daily Lower BB(20,2.0) x Daily Close] 
      ```

   6. Average Directional Index

      Overall Uptrend

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 100000] 
      AND [Daily SMA(60,Daily Close) > 10] 
      
      AND [Daily ADX Line(14) > 20] 
      AND [Daily Plus DI(14) crosses Daily Minus DI(14)] 
      AND [Daily Close > Daily SMA(50,Daily Close)]
      ```

      Overall Downtrend

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 100000] 
      AND [Daily SMA(60,Daily Close) > 10] 
      
      AND [Daily ADX Line(14) > 20] 
      AND [Daily Minus DI(14) crosses Daily Plus DI(14)] 
      AND [Daily Close < Daily SMA(50,Daily Close)]
      ```

   7. Williams %R

      Turn Up From Oversold

      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close > Daily SMA(200,Daily Close)] 
      AND [20 days ago Daily Williams %R(14) < -80] 
      AND [Daily Williams %R(14) crosses -50]
      ```
   
      Turn Down From Overbought
   
      ```pseudocode
      [type = stock] AND [country = US] 
      AND [Daily SMA(20,Daily Volume) > 40000] 
      AND [Daily SMA(60,Daily Close) > 20] 
      
      AND [Daily Close < Daily SMA(200,Daily Close)] 
      AND [20 days ago Daily Williams %R(14) > -20] 
      AND [-50 crosses Daily Williams %R(14)]
      ```
   
   8. Commodity Channel Index
   
   9. Average True Range
   
   10. Mass Index
   
   11. Ultimate Oscillator
   
   12. Rate of Change
   
4. Rules

   1. Cleared in one day