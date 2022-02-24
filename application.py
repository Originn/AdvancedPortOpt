import os

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import plotly.graph_objects as go
import plotly
import plotly.express as px
import requests
from pandas_datareader import data as pdr
import math
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
import datetime
import re
import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import risk_models, DiscreteAllocation, objective_functions, EfficientSemivariance, efficient_frontier, EfficientFrontier
from pypfopt import EfficientFrontier
import pypfopt
import io
from helpers import login_required, lookup, usd, gbp, GBPtoUSD, contains_multiple_words
import json
#from waitress import serve

dicts={}
pd.set_option('display.precision', 7)
# Configure application
app = Flask(__name__)
DATABASE_URI = os.environ['DATABASE_URL']
print(DATABASE_URI)
DATABASE_URI= DATABASE_URI[:8]+'ql' + DATABASE_URI[8:]
print(DATABASE_URI)
engine = create_engine(os.getenv('DATABASE_URI'))

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


#db = SQLAlchemy(app)
db=scoped_session(sessionmaker(bind=engine))


@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    #quering for the symbol and the corresponding sum of the same stock and the average price paid for the stock
    stocks = db.execute("SELECT symbol, SUM(number_of_shares) as sumshares, AVG(price) as avgprice, AVG(purchase_p) as purchase_p FROM records WHERE user_id = :i_d GROUP BY symbol", {"i_d": session["user_id"]}).all()
    if stocks is None:
        stocks = []
    else:
        stocks = [{'symbol': a, 'sumshares': b, 'avgprice': c, 'purchase_p': d} for a, b, c, d in stocks]
    cash = db.execute("SELECT cash FROM users WHERE id = :id", {"id": session["user_id"]}).all()
    cash = ([i[0] for i in cash][0])

    totalPortValue = 0
    totalprolos = 0

    #building the index
    for stock in stocks:
        symbol=(stock["symbol"])
        data=lookup(symbol)
        name=data["shortName"]
        price = data["regularMarketPrice"]
        stock['name'] = name
        #check if the stock is listed in the UK
        if ".L" in data["symbol"]:
            #if it is - convert the price from GBP to USD
            price=GBPtoUSD()*price
        stock['ap'] = (stock['sumshares'] * price)/stock['sumshares']
        stock['total'] = stock['sumshares'] * price
        stock['perc_change'] = round(((stock['ap'] - stock['purchase_p'])/stock['purchase_p'])*100, 3)
        stock['prolos'] = (stock['perc_change']/100)*stock['total']
        totalprolos += stock['prolos']
        totalPortValue += stock['sumshares'] * price

    availableCash = cash
    grandTotal = availableCash + totalPortValue

    return render_template("/index.html",availableCash=round(availableCash, 4), stocks=stocks, totalPortValue=totalPortValue, grandTotal=grandTotal, totalprolos=totalprolos)

@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":

        try:
            shares = int(request.form.get("shares"))
        except ValueError:
            flash("Must provide a valid whole number", 400)
            return redirect ("/buy")

        share = int(request.form.get("shares"))

        if request.form.get("shares") == "" or int(request.form.get("shares")) <= 0:
            flash("Must provide positive number of shares")
            return redirect ("/buy")


        else:
            data = lookup(request.form.get("symbol"))
            symbol=data['symbol']
            availableCash=db.execute("SELECT cash FROM users WHERE id = :user_id", {'user_id':session["user_id"]})
            availableCash=[item[0] for item in availableCash][0]

            price=data["regularMarketPrice"]
            if ".L" in symbol:
                price=GBPtoUSD()*price
            sharesPrice = price * int(request.form.get("shares"))
            if sharesPrice > availableCash:
                flash("Not enough money")
                return redirect ("/buy")
            else:
                now = datetime.datetime.now()
                formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
                #insert new row in the database to record the purchase
                db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",{
                            'user_id':session["user_id"], 'symbol':symbol, 'price':price, 'number_of_shares':request.form.get("shares"), 'time':formatted_date, 'status':'purchase'})
                db.execute("INSERT INTO records (user_id, symbol, price, number_of_shares, execution_time, transaction_type, purchase_p) VALUES(:user_id, :symbol, :price, :number_of_shares, :execution_time, :transaction_type, :purchase_p)",{'user_id':session["user_id"], 'symbol':symbol, 'price':price, 'number_of_shares':request.form.get("shares"), 'execution_time':formatted_date, 'transaction_type':"purchase", 'purchase_p':price})
                db.execute("UPDATE users SET cash = :cash WHERE id = :user_id",{'cash':availableCash-sharesPrice, 'user_id':session["user_id"]})
                db.commit()
                return redirect("/")
    else:
        return render_template("buy.html")


@app.route("/history", methods=["GET", "POST"])
@login_required
def history():
    """Show history of transactions"""
    if request.method == "POST":
        hist = db.execute("SELECT * FROM history WHERE DATE(time) BETWEEN :start AND :end AND user_id = :i_d ORDER BY time DESC", {'start':str(request.form.get("start")), 'end':str(request.form.get("end")), 'i_d':session["user_id"]}).all()
        if hist == []:
           return render_template("history.html")
        else:
            hist=[{'status': a, 'symbol': b, 'price': c, 'number_of_shares': d, 'time': e, 'user_id': f} for a, b, c, d, e, f in hist]
        status = hist[0]["status"]
        symbol = hist[0]["symbol"]
        price = hist[0]["price"]
        number_of_shares = hist[0]["number_of_shares"]
        time = hist[0]["time"]
        return render_template("history1.html", hist=hist)
    else:
        #get the earlist date of the history records
        edate=db.execute("SELECT DATE(time) FROM history WHERE user_id = :i_d ORDER BY time ASC", {'i_d':session["user_id"]})
        edate=[item[0] for item in edate][0]
        #get the last date of the history records
        ldate=db.execute("SELECT DATE(time) FROM history WHERE user_id = :i_d ORDER BY time DESC", {'i_d':session["user_id"]})
        ldate=[item[0] for item in ldate][0]
        return render_template("history.html", edate=edate, ldate=ldate)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Forget any user_id
        session.clear()
        # Ensure username was submitted
        if not request.form.get("username"):
            flash("must provide username")
            return redirect("/login")

        # Ensure password was submitted
        elif not request.form.get("password"):
            flash("must provide username")
            return redirect("/login")

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = :username", {"username": request.form.get("username")})
        try:
            rows = [i for i in rows][0]
        except IndexError:
            flash("invalid username and/or password")
            return redirect("/login")
        # Ensure username exists and password is correct
        if not check_password_hash(rows[1], request.form.get("password")):
            flash("invalid username and/or password")
            return redirect("/login")

        # Remember which user has logged in
        session["user_id"] = rows[3]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":
        if request.form.get("symbol") == "":
            flash("Must provide a symbol")
            return redirect("/quote")
        quote = lookup(request.form.get("symbol"))
        #if ticker does not exist
        try:
            if quote["symbol"] is None:
                flash("No such symbol")
                return redirect("/quote")
        except TypeError:
            flash("No symbol entered")
            return redirect("/quote")
        else:
            price=quote["regularMarketPrice"]
            symbol=quote["symbol"]
            #if the stock is listed in the United Kingdon add £ sign
            if ".L" in symbol:
                price=gbp(price)
            if "." not in symbol:
                price=usd(price)
            return render_template("quoted.html", name=quote["shortName"], price=price, symbol=symbol)
    else:
        return render_template("quote.html")

@app.route("/quoted")
@login_required
def quoted():
    return render_template("quoted.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    if request.method == "POST":

        name = request.form.get("username")
        if not name:
            flash("Missing name")
            return redirect("/register")
        findName=db.execute("SELECT username FROM users WHERE username = :name", {'name' : request.form.get("username")}).all()
        try:
            findName=[i[0] for i in findName][0]
            if name == findName:
                flash("User name already taken")
                return redirect("/register")
        except:
            pass


        password = request.form.get("password")
        if not password:
            flash("Missing password")
            return redirect("/register")
        if len(password) < 8:
            flash("Password is less than 8 characters")
            return redirect("/register")

        if not (re.search(r"[\d]+", password) and re.search(r"[A-Z]+", password)):
            flash("This password must contain at least 1 digit and at least 1 uppercase character")
            return redirect("/register")



        confirmation = request.form.get("confirmation")
        if password != confirmation:
            flash("Confirmation was not successful")
            return redirect("/register")

        if password == confirmation:
            password = generate_password_hash(password)
            db.execute("INSERT INTO users (username, hash) VALUES (:n, :u)", {'n': str(name), 'u': str(password)})
            db.commit()
            #Store user id in db
            rows = db.execute("SELECT * FROM users WHERE username = :username", {"username": request.form.get("username")}).all()
            session["user_id"] = [i[3] for i in rows][0]
            engine.table_names()
            db.execute("INSERT INTO records (user_id) VALUES (:user_id)", {'user_id': int(session["user_id" ])})
            db.commit()
            return redirect("/")

    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    if request.method == "POST":
        symbolToSell=request.form.get("symbol")
        NumOfShares = db.execute("SELECT symbol, SUM(number_of_shares) as sumshare FROM records WHERE symbol = :symbolToSell AND user_id = :user_id AND transaction_type = :ty GROUP BY symbol", {'symbolToSell':str(request.form.get("symbol")), 'user_id':session["user_id"], 'ty':'purchase'})
        NumOfShares=[{'symbol': a, 'sumshare': b} for a, b in NumOfShares]
        NumOfshareToSell = int(request.form.get("shares"))
        if NumOfshareToSell > int(NumOfShares[0]["sumshare"]):
            flash("You don't have enough stocks")
            return redirect("/sell")
        else:
            cash = db.execute("SELECT cash FROM users WHERE id = :id", {'id':session["user_id"]})
            cash=[item[0] for item in cash][0]
            price = float(lookup(request.form.get("symbol"))['regularMarketPrice'])
            if ".L" in symbolToSell:
                price=GBPtoUSD()*price
            moneyBack = NumOfshareToSell*price
            #setting up the time stamp to enter the sell into transaction table
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            #entering the sale into transaction table
            negative= (int(request.form.get("shares"))) * (-1)
            db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",{'user_id':session["user_id"], 'symbol':request.form.get("symbol"), 'price':price, 'number_of_shares':request.form.get("shares"), 'time':formatted_date, 'status':'sold'})
            #db.execute("INSERT INTO records (user_id, symbol, price, number_of_shares, execution_time, transaction_type) VALUES(:user_id, :symbol, :price, :number_of_shares, :execution_time, :transaction_type)",{'user_id':session["user_id"], 'symbol':request.form.get("symbol"), 'price':price, 'number_of_shares':negative, 'execution_time':formatted_date, 'transaction_type':'sold'})
            db.execute("UPDATE users SET cash = :cash WHERE id = :user_id", {'cash':cash+moneyBack, 'user_id':session["user_id"]})
            if NumOfshareToSell == int(NumOfShares[0]["sumshare"]):
                db.execute("DELETE FROM records WHERE user_id =:ui AND symbol=:symbol", {'ui':session["user_id"], 'symbol':request.form.get("symbol")})
            else:
                db.execute("UPDATE records SET number_of_shares =:n WHERE user_id =:ui AND symbol= :symbol", {'ui':session["user_id"], 'symbol':request.form.get("symbol"), 'n':int(NumOfShares[0]["sumshare"])-int(request.form.get("shares"))})
            db.commit()
            return redirect("/")
    else:
        stocks = db.execute(
            "SELECT symbol, SUM(number_of_shares) as sumshare FROM records WHERE user_id = :user_id GROUP BY symbol", {'user_id':session["user_id"]}).all()
        stocks=[{'symbol': a, 'sumshare': b} for a, b in stocks]
        return render_template("/sell.html", stocks=stocks)

@app.route("/about")
def about():
    return render_template("/about.html")

@app.route("/risk models")
def risk():
    return render_template("/risk models.html")

@app.route("/expected returns")
def expected_returns():
    return render_template("/expected returns.html")

@app.route("/build",methods=["GET", "POST"])
@login_required
def build():
    if request.method == "POST":
        #yf.pdr_override()
        symbols = request.form.get("symbols")
        if contains_multiple_words(symbols) == False:
            flash("The app purpose is to optimize a portfolio given a list of stocks. Please enter a list of stocks seperated by a new row.")
            return redirect("/build")
        db.execute("INSERT INTO build (user_id, stocks, date_start, date_end, amount, shortperc, volatility, gamma, target_return) VALUES(:user_id, :stocks, :date_start, :date_end, :amount, :shortperc, :volatility, :gamma, :target_return)",{'user_id':session["user_id"], 'stocks':symbols.upper(), 'date_start':request.form.get("start"), 'date_end':request.form.get("end"), 'amount':request.form.get("funds"), 'shortperc':request.form.get("short"), 'volatility':request.form.get("volatility"), 'gamma':request.form.get("gamma"), 'target_return':request.form.get("return")})
        db.commit()
        try:
            df = yf.download(symbols, start=request.form.get("start"), end=request.form.get("end"), threads=False)["Adj Close"].dropna(axis=1, how='all')
            df = df.replace(0, np.nan)
            try:
                listofna=df.columns[df.isna().iloc[-2]].tolist()
            except IndexError:
                flash("Please enter valid stocks from Yahoo Finance.")
                return redirect("/build")
            df = df.loc[:,df.iloc[-2,:].notna()]


        except ValueError:
            flash("Please enter a valid symbols (taken from Yahoo Finance)")
            return redirect("/build")
        prices = df.copy()
        fig = px.line(prices, x=prices.index, y=prices.columns, title='Price Graph')
        fig = fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(width=900, height=600)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        sample_cov = risk_models.sample_cov(prices, frequency=252)

        #plotting the covariance matrix
        heat = go.Heatmap(
            z = risk_models.cov_to_corr(sample_cov),
            x = sample_cov.columns.values,
            y = sample_cov.columns.values,
            zmin = 0, # Sets the lower bound of the color domain
            zmax = 1,
            xgap = 1, # Sets the horizontal gap (in pixels) between bricks
            ygap = 1,
            colorscale = 'RdBu'
        )

        title = 'Covariance matrix'

        layout = go.Layout(
            title_text=title,
            title_x=0.5,
            width=600,
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        fig1=go.Figure(data=[heat], layout=layout)
        fig1.update_layout(width=500, height=500)
        plot_json1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)



        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        heat = go.Heatmap(
            z = risk_models.cov_to_corr(S),
            x = S.columns.values,
            y = S.columns.values,
            zmin = 0, # Sets the lower bound of the color domain
            zmax = 1,
            xgap = 1, # Sets the horizontal gap (in pixels) between bricks
            ygap = 1,
            colorscale = 'RdBu'
        )

        title = 'Ledoit-Wolf shrinkage'

        layout = go.Layout(
            title_text=title,
            title_x=0.5,
            width=600,
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        fig2=go.Figure(data=[heat], layout=layout)
        fig2.update_layout(width=500, height=500)
        plot_json2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        #Section 2 -Return estimation
        #it is often a bad idea to provide returns using a simple estimate like the mean of past returns. Research suggests that better off not providing expected returns – you can then just find the min_volatility() portfolio or use HRP.
        mu = pypfopt.expected_returns.capm_return(prices)
        fig3 = px.bar(mu, orientation='h')
        fig3.update_layout(width=700, height=500)
        plot_json3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


        #using risk models optimized for the Efficient frontier to reduce to min volitility
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(None, S)
        try:
            ef.min_volatility()
            weights = ef.clean_weights()
            nu = pd.Series(weights)
            fig4 = px.bar(nu, orientation='h')
            fig4.update_layout(width=700, height=500)
            plot_json4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
            av=ef.portfolio_performance()[1]
            av=round(av, 3)*1

        #if we want to buy the portfolio mentioned above
            df = df.iloc[[-1]]
            for col in df.columns:
                if col.endswith(".L"):
                    df.loc[:,col] = df.loc[:,col]*GBPtoUSD()
            try:
                latest_prices = df.iloc[-1]
            except IndexError:
                flash("There is an issue with Yahoo API please try again later")
                return redirect("/")
            # prices as of the day you are allocating
            if float(request.form.get("funds")) <= 0 or float(request.form.get("funds")) == " ":
                flash("Amount need to be a positive number")
                return redirect("/build")
            if float(request.form.get("funds")) < float(latest_prices.min()):
                flash("Amount is not high enough to cover the lowest priced stock")
                return redirect("/build")
            try:
                da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
            except TypeError:
                delisted=df.columns[df.isna().any()].tolist()
                delisted= ", ".join(delisted)
                flash("Can't get latest prices for the following stock/s, please remove to contiue : %s" % delisted)
                return redirect("/build")
            alloc, leftover = da.lp_portfolio()
            session['alloc']=alloc
            session['latest_prices']=latest_prices
        except ValueError:
            pass

        #Maximise return for a given risk, with L2 regularisation
        try:
            ef = EfficientFrontier(mu, S)
            #ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
            ef.add_objective(objective_functions.L2_reg, gamma=(float(request.form.get("gamma"))))  # gamme is the tuning parameter
            ef.efficient_risk(int(request.form.get("volatility"))/100)
            weights = ef.clean_weights()
            su = pd.DataFrame([weights])
            fig5 = px.pie(su, values=weights.values(), names=su.columns)
            fig5.update_traces(textposition='inside')
            fig5.update_layout(width=500, height=500, uniformtext_minsize=12, uniformtext_mode='hide')
            plot_json5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
            perf =ef.portfolio_performance()
        except Exception as e:
            flash(str(e))
            return redirect("/build")


        #if we want to buy the portfolio mentioned above
        for col in df.columns:
            if col.endswith(".L"):
                df.loc[:,col] = df.loc[:,col]*GBPtoUSD()
        latest_prices1 = df.iloc[0]  # prices as of the day you are allocating
        if float(request.form.get("funds")) <= 0 or float(request.form.get("funds")) == " ":
            flash("Amount need to be a positive number")
            return redirect("/build")
        if float(request.form.get("funds")) < float(latest_prices.min()):
            flash("Amount is not high enough to cover the lowest priced stock")
            return redirect("/build")
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
        alloc1, leftover1 = da.lp_portfolio()
        session['alloc1']=alloc1
        session['latest_prices1']=latest_prices1

        #Efficient semi-variance optimization

        returns = pypfopt.expected_returns.returns_from_prices(prices)
        returns = returns.dropna()
        es = EfficientSemivariance(mu, returns)
        es.efficient_return(float(request.form.get("return"))/100)
        perf2=es.portfolio_performance()
        weights = es.clean_weights()

        #if we want to buy the portfolio mentioned above
        for col in df.columns:
            if col.endswith(".L"):
                df.loc[:,col] = df.loc[:,col]*GBPtoUSD()
        latest_prices2 = df.iloc[0]  # prices as of the day you are allocating
        if float(request.form.get("funds")) <= 0 or float(request.form.get("funds")) == " ":
            flash("Amount need to be a positive number")
            return redirect("/build")
        if float(request.form.get("funds")) < float(latest_prices.min()):
            flash("Amount is not high enough to cover the lowest priced stock")
            return redirect("/build")
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
        alloc2, leftover2 = da.lp_portfolio()
        session['alloc2']=alloc2
        session['latest_prices2']=latest_prices2
        return render_template ("built.html",av=av, leftover=leftover, alloc=alloc, ret=float(request.form.get("return")),gamma=request.form.get("gamma"),volatility=request.form.get("volatility"),perf=perf, perf2=perf2, alloc1=alloc1, alloc2=alloc2, plot_json=plot_json, plot_json1=plot_json1, plot_json2=plot_json2, plot_json3=plot_json3, plot_json4=plot_json4, plot_json5=plot_json5, leftover1=leftover1, leftover2=leftover2,listofna=(', '.join(listofna)))
    else:
        availableCash=db.execute("SELECT cash FROM users WHERE id = :id", {"id": session["user_id"]}).all()
        availableCash=[i[0] for i in availableCash][0]

        return render_template("build.html", availableCash=round(availableCash, 4), GBP=GBPtoUSD())

@app.route("/allocation", methods=["GET", "POST"])
@login_required
def allocation():
    alloc=session['alloc']
    latest_prices=session['latest_prices']
    values = alloc.values()
    total = sum(values)
    for key, value in alloc.items():
        price=latest_prices[key]
        amount=value
        availableCash=db.execute("SELECT cash FROM users WHERE id = :user_id", {'user_id': session["user_id"]})
        availableCash=[item[0] for item in availableCash][0]
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",
            {'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'time':formatted_date, 'status':'purchase'})
            db.execute("INSERT INTO records (user_id, symbol, price, number_of_shares, execution_time, transaction_type, purchase_p) VALUES(:user_id, :symbol, :price, :number_of_shares, :execution_time, :transaction_type, :purchase_p)",{'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'execution_time':formatted_date, 'transaction_type':"purchase", 'purchase_p':price})
            db.execute("UPDATE users SET cash = :cash WHERE id = :user_id", {'cash': availableCash-(amount*price), 'user_id':session["user_id"]})
            db.commit()

    return redirect("/")

@app.route("/allocation1", methods=["GET", "POST"])
@login_required
def allocation1():
    alloc1=session['alloc1']
    latest_prices1=session['latest_prices1']
    values = alloc1.values()
    total = sum(values)

    for key, value in alloc1.items():
        price=latest_prices1[key]
        amount=value
        availableCash=db.execute("SELECT cash FROM users WHERE id = :user_id", {'user_id':session["user_id"]})
        availableCash=[item[0] for item in availableCash][0]
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",
            {'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'time':formatted_date, 'status':'purchase'})
            db.execute("INSERT INTO records (user_id, symbol, price, number_of_shares, execution_time, transaction_type, purchase_p) VALUES(:user_id, :symbol, :price, :number_of_shares, :execution_time, :transaction_type, :purchase_p)",{'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'execution_time':formatted_date, 'transaction_type':"purchase", 'purchase_p':price})
            db.execute("UPDATE users SET cash = :cash WHERE id = :user_id", {'cash': availableCash-(amount*price), 'user_id':session["user_id"]})
            db.commit()

    return redirect("/")

@app.route("/allocation2", methods=["GET", "POST"])
@login_required
def allocation2():
    alloc2=session['alloc2']
    latest_prices2=session['latest_prices2']
    values = alloc2.values()
    total = sum(values)
    for key, value in alloc2.items():
        price=latest_prices2[key]
        amount=value
        availableCash=db.execute("SELECT cash FROM users WHERE id = :user_id", {'user_id':session["user_id"]})
        availableCash=[item[0] for item in availableCash][0]
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",
            {'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'time':formatted_date, 'status':'purchase'})
            db.execute("INSERT INTO records (user_id, symbol, price, number_of_shares, execution_time, transaction_type, purchase_p) VALUES(:user_id, :symbol, :price, :number_of_shares, :execution_time, :transaction_type, :purchase_p)",{'user_id':session["user_id"], 'symbol':key, 'price':price, 'number_of_shares':int(amount), 'execution_time':formatted_date, 'transaction_type':"purchase", 'purchase_p':price})
            db.execute("UPDATE users SET cash = :cash WHERE id = :user_id", {'cash': availableCash-(amount*price), 'user_id':session["user_id"]})
            db.commit()

    return redirect("/")

@app.route("/sell_all", methods=["GET", "POST"])
@login_required
def sell_all():
    numberofShares=db.execute("SELECT symbol, number_of_shares FROM records WHERE user_id= :uid", {'uid':session["user_id"]})
    numberofShares=[{'symbol': a, 'number_of_shares': b} for a, b in numberofShares]
    for stock in numberofShares:
        symbol=stock["symbol"]
        data=lookup(symbol)
        NumOfshareToSell=stock["number_of_shares"]
        cash = db.execute("SELECT cash FROM users WHERE id = :id", {'id':session["user_id"]})
        cash=[item[0] for item in cash][0]
        price = data["regularMarketPrice"]
        if ".L" in symbol:
            price=GBPtoUSD()*price
        moneyBack = (int(NumOfshareToSell))*price
        #setting up the time stamp to enter the sell into transaction table
        now = datetime.datetime.now()
        formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
        #entering the sale into transaction table
        negative= (int(NumOfshareToSell)) * (-1)
        db.execute("INSERT INTO history (user_id, symbol, price, number_of_shares, time, status) VALUES(:user_id, :symbol, :price, :number_of_shares, :time, :status)",{'user_id':session["user_id"], 'symbol':symbol, 'price':price, 'number_of_shares':NumOfshareToSell, 'time':formatted_date, 'status':'sold'})
        db.execute("UPDATE users SET cash = :cash WHERE id = :user_id", {'cash':cash+moneyBack, 'user_id':session["user_id"]})
    db.execute("DELETE FROM records WHERE user_id =:ui", {'ui':session["user_id"]})
    db.commit()
    return redirect("/")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    flash(e.name)
    return redirect("/")


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
