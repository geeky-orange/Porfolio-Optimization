# Program Start

1. start from the test_portfolio_result.py
2. for different risk level, change the get_risk_profile() function in portfolio_property.py



# Version 1

1. Overview

   This is the most base version and the structure follows the Monte Carlo Tree (MCT). Our goal is to search the good portfolios from historical data. Instead of randomly search in over billions of assets pools, the MCT search can decide which path to search with certain confidence interval. As a result, the search speed is much faster and get different results by changing the a) portfolio policies: evaluation function, b) search preference c value of Upper Confidence Bound, c) success conditions: backpropagate rules.

2. Difficulties:
   1. Asset Selection - Too many assets: there are over **2400** ETFs in the U.S. markets and the portfolios contains the those assets has over $10^{27}$​​​ of combination. It is impossible to solve it with brute-force method.
   2. Asset Screening - Too many metric: for one asset, they have different properties that can be used to evaluate the asset. it is the same for the portfolios. So, how to define what is a good asset or what is a good portfolios.
   3. Asset Allocation - Too many combination: even though we have selected the assets, how to determine the weight of each assets in the portfolio. there are nearly infinite choices. With the help of modern portfolio theory, we can minimize the risk as well as satisfy the requirement of return.
   4. Asset Prediction - Too many uncertainty: if we have found a portfolio that performs well, it is based on historical data. There is no guarantee that the portfolio will keep provide with excellent return in the future. So, we applied the back testing and use deep learning to predict the future performance of our portfolios (not completed yet).

3. Reason of Monte Carlo Tree

   1. unbelievable performance of AlphaGo: About five years ago, the Go, a kind of Chinese chess, was regarded as the most impossible tasks for A.I., because wining this game requires multiple layers of strategic thinking and there are an astonishing 10 to the power of 170 possible board configurations - more than the number of atoms in the known universe. However, five years ago, AlphaGo beat the Lee Sedol, one of the greatest players in the world and just one year later, AlphaGo Zero beats the No.1 player Ke Jie. That is the charming and powerful aspect of A.I because it has infinite possibility.
   2. similarities between our problems and AlphaGo: 
   3. expandability of MCT: to get the portfolios of different risk profile, we just need to modify the policy (evaluation function), also if we have experts’ insight, we can add it into the policy, most importantly if we need the future performance, a deep learning prediction model can be applied to get the score.
   4. parallelization: the architecture of MCT can make the searching process be paralleled easily. We can use multiprocessing when expand the child nodes of one node and multithreading when rolling out to get results.

4. Detail of the model

   1. in our model, we use the same backbone as AlphaGo, the Monte Carlo Tree, with our own design evaluation methods with the help of professional investors’ insights, historical data analysis and deep learning model to predict the future performance. With our model, we can quickly search and find out the most proper assets that suits customers’ different risk level. Then, we applied the modern portfolio theory to optimize our portfolios to get the highest return while minimize the risk.




# Version 2.0

new feature: 

1. add the save function when backpropagation to the root node.
2. the new time slot from 2016-01-01 to 2020-01-01 was feed into the model to get result.
3. add the daily_data_downloader.py to the util and rebuild the structure of project.

# Version 2.1

new feature:

1. change the success condition of the backpropagation method in nodes.py, in this condition, the tree preferred results with good return over drawdown.

   ```python
   (current_sharpe > last_sharpe and current_returns > last_returns) or \
           current_sharpe * current_returns >= last_sharpe * last_returns:
   ```

    &rightarrow; 

   ```python
   if (current_sharpe > last_sharpe or current_returns > last_returns) and \
           current_return_over_drawdown > last_return_over_drawdown:
   ```

   

2. add two constraints (the weight of an asset cannot be greater than 40%, the number of assets whose weight is greater than 0.02 should be greater than 3) in the get_result function of portfolio construction. 

3. 



# Version 2.2

new feature:

1. the new time slot from 2018-07-01 to 2021-06-30 was feed into the model to get result.
2. fixed the bug in the portfolio_construction.py that return the asset classes that has weight less than 0.02
3. add one new constrains (the number of asset classes should be greater than 3) in the get_result function of portfolio construction. 

# Version 2.3

new feature:

1. change the structure of the program, put some useful function into the util directory and store all data in the data directory
2. add the visualization function that show the line chart, pie chart and bar chart in one graph.
3. maximize the return subject to different risk level: add one constraint to the optimization process

| Risk Level            | Max Drawdown |
| --------------------- | ------------ |
| conservative          | <= 5%        |
| moderate conservative | 5% ~ 10%     |
| moderate              | 10% ~ 20%    |
| moderate aggressive   | 20% ~ 25%    |
| aggressive            | >= 25%       |

4. the return over drawdown rate is greater than 1.3: change the if statement in the get_result function.
5. the percentage of commodities is less than 20%: add one constraint to the optimization process.
6. fixed the bug that name and level of different programs.
7. fixed the bug that the asset with weight less than 2% 



# Version 3.0

New Feature:

1. Apply multiprocessing in expand process, apply multithreading in rollout process.

   **Fail and Revision** :

   Directly applying the multiprocessing method is not suitable, because different processes use their own memory space. Thus the class attributes cannot be accessed correctly.

   Then some revision about multiprocessing and multithreading of python was done. In python, if the program has many IO operations, multithreading is preferred, if the program has many CPU calculations, then multiprocessing is better. However, the shared memory needs be managed with lock when applying multiprocessing.   

# Version 3.1 (Future Plan)

New Feature:

1. Apply shared memory manager and namespace to control the class attributes. (Not Completed)



# Version 3.2 (Future Plan)

New Feature:

1. Search five portfolios in the same time.
2. Save the model and update it by feeding new data.



# Version 4.0 (Future Plan)

New Feature:

1. Applied Deep Learning and professional investors’ insight into the evaluation function.

