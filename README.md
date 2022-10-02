# Stock Portfolio Manager
#### Website:   https://advportopt.herokuapp.com/
#### Description:
This app was build using Yahoo finance free stock quotes [yfinance](https://pypi.org/project/yfinance/) and
a library developed by Robert Martin [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/).

PyPortfolioOpt library features several useful functions derived from modern portfolio optimization methods such as Risk Models,
Mean-Variance Optimization, Black-Litterman Allocation and more.

Yfinance project helps with pulling historic data of stocks that are listed all around the world, including cryptocurrencies. This helps
with portfolio diversification.

The webapp combine the two powerful libraries with a demo account for testing purposes. The build webpage is where the list of stocks should be input into. The optimal number is between 30 - 50 stocks. UK listed stocks are converted from pounds to dollars using an [API](https://api.exchangerate-api.com/v4/latest/USD).

Passwords are encrypted in a secure database.

I hope you will enjoy the app and find it useful!

known issues: Testing of past performances feature should be improved.
