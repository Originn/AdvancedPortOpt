# Stock Portfolio Manager
#### Website:  https://limitless-meadow-36639.herokuapp.com/
#### Description:
This app was build using Yahoo finance free stock quotes [yfinance](https://pypi.org/project/yfinance/) and
a library developed by Robert Martin [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/).

PyPortfolioOpt library features several useful functions derived from modern portfolio optimization methods such as Risk Models,
Mean-Variance Optimization, Black-Litterman Allocation and more.

Yfinance project helps with pulling historic data of stocks that are listed all around the world, including cryptocurrencies. This helps
with portfolio diversification.

The build webpage is where the list of stocks should be input into. The optimal number is between 30 - 50 stocks. UK listed stocks are converted from pounds to dollars using an [API](https://api.exchangerate-api.com/v4/latest/USD).
Input the date range of which to extract historical price data from. The To field is already populated
with the current date for ease of use.

This project also helped to optimize yfinance speed by downloading the stock data using json, which cut the time by 80% to view, build
,buy and sell the portfolio. There is no API need for downloading the data which is extremally helpful when testing a large amount of stocks,
with different time frames. This was a crucial decision in the design as paid API was not feasible at the moment.

Passwords are encrypted in a secure database.

I hope you will enjoy the app and find it useful!
