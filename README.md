# Stock Portfolio Manager
#### Video Demo:  https://youtu.be/fWRtpqz3OxI
#### Description:
This app was build using Yahoo finance free stock quotes [yfinance](https://pypi.org/project/yfinance/) and 
a liberary devolped by Robert Martin [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/).

PyPortfolioOpt library features several usefull functions derived from modern portfolio optimization methods such as Risk Models,
Mean-Variance Optimization, Black-Litterman Allocation and more.

Yfinance project helps with pulling historic data of stocks that are listed all around the world, including cryptocurrencies. This helps
with portfolio diversification. 

The build.html webpage is where the user should input the list of stocks (due to IDE performance ~50 is sufficient). I have also included
the option to add UK listed stocks by converting pounds to dollars using an [API]('https://api.exchangerate-api.com/v4/latest/USD').
Next, the user should input the date range of which he wants to extract historical price data from. The To field is already populated 
with the current date for ease of use.

The amount to be invested input will automatically be filled with the available cash from the demo account, again for ease of use. 
The input field output an alert if number or negative numbers are entered. The volatilty input asks the user for his upperbounds constraints,
again inputing an alert if an unvalid entry is made.

The gamma input is a number between 0.1 and 1 which tell the optimizer how to allocate the weights as larger gamma will output 
an equal allocation. Finally the annual rate of return is requested for the last portfolio suggestion in case the user is interested 
with a specific yield with minimum volatility.

After the Build botton is pressed, the return us built.html. I have add the price graphs, bar and pie charts using images, which is not
optimal in case of multiple users, but using a native graph plotter in HTML was difficult. In the built.html the 3 portfolios are suggested
to the user, where he can choose which one he is more content with. The 3 options are displayed with the names and numbers of stocks to buy,
and the left over amount after buying it. Initially I have thought to leave at as is and let the buyer buy the stocks one by one from the 
buy.html page, but that will be very bad UX. I therefore created a button that calls a function (/allocation) which will buy all portfolio
automatically and redirect the user to the index where he can follow the portfolio performance. 

I have added to the index columns that will show the percent change from the date if was bought, using flask language and CSS to add an 
green or red arrow depends on the change. I have added it to each stock in the index and also for the total amount. The prices on sell are
updated to the latest amount.

I have also add another SQL table in order to capture the stocks input the user had provided in the Build function. I have also added a
Sell All button for ease of use, which of course take into account the stock latest prices. Additionally I have created a form which the user
can choose the date range of his buy/sell history. This was for the user smooth UX. 

This project aslo helped to optimize yfinance speed by downloading the stock data using json, which cut the time by 80% to view, build
,buy and sell the portfolio. There is no API need for downloading the data which is extreamlly helpful when testing a large amount of stocks,
with different time frames. This was a crucial decision in the design as paid API was not feasible at the moment.

The registration page will also check if the user name is already taken, it will hash the user password and store it in the database, and 
check the hash on user loggin. I have added few rules for password creation. 

I have added few helpers functions. One was for the lookup function which was needed to optimized for speed. A converter from GBP to USD and
few other functions.

Few conssesion were made due to the limitation of CS50:
It was possible to add user input for market constraints e.g. 5% Technology stocks, 10% consumer stapels, but app speed was greatly affected.

I hope you will enjoy the app and find it useful!


