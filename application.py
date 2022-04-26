import os, plotly, requests, math, datetime, re, json, psycopg2, bmemcached, redis
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np
import pypfopt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy import desc
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from pypfopt import risk_models, DiscreteAllocation, objective_functions, EfficientSemivariance, efficient_frontier, EfficientFrontier
from pypfopt import EfficientFrontier
from helpers import login_required, lookup, usd, gbp, GBPtoUSD, contains_multiple_words, symbol_search, price_lookup
from urllib.parse import urlparse
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from pytrading212 import *
import random
import yfinance.shared as shared
import numpy as np


# Configure application
app = Flask(__name__)

#setting options for webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("no-sandbox")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--headless")

#initiating chrome driver
driver = webdriver.Chrome('/home/originn/AdvancedPortOpt/chromedriver/stable/chromedriver', options=chrome_options)
#setting 212 account parameters
email = os.environ.get("EMAIL")
password = os.environ.get("PASS")

#initiate trading 212
#trading212 = Trading212(email, password, driver, mode=Mode.DEMO)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
DATABASE_URI = os.environ['DATABASE_URL']
DATABASE_URI= DATABASE_URI[:8]+'ql' + DATABASE_URI[8:]
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "redis"
if 'HEROKU' in os.environ:
    url = urlparse(os.environ.get("REDIS_TLS_URL"))
else:
    url = urlparse(os.environ.get("REDIS_URL"))
app.config["SESSION_REDIS"]=redis.Redis(host=url.hostname, port=url.port, username=url.username, password=url.password, ssl=True, ssl_cert_reqs=None)

db = SQLAlchemy(app)

#set memcache in Heroku
servers = os.environ.get('MEMCACHIER_SERVERS', '').split(',')
user = os.environ.get('MEMCACHIER_USERNAME', '')
passw = os.environ.get('MEMCACHIER_PASSWORD', '')

mc = bmemcached.Client(servers, username=user, password=passw)

mc.enable_retry_delay(True)

class Users(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String)
    hash=db.Column(db.Text)
    cash=db.Column(db.Float)

    def __init__(self, username, hash, cash):
        self.username = username
        self.hash = hash
        self.cash = cash

class Records(db.Model):
    symbol=db.Column(db.Text)
    number_of_shares=db.Column(db.Integer)
    transaction_type=db.Column(db.Text)
    execution_time=db.Column(db.DateTime)
    purchase_p=db.Column(db.Float)
    user_id=db.Column(db.Integer, primary_key=True)
    price=db.Column(db.Float)

    def __init__(self, user_id, symbol, number_of_shares, transaction_type, execution_time, purchase_p, price):
        self.user_id = user_id
        self.symbol = symbol
        self.number_of_shares = number_of_shares
        self.transaction_type = transaction_type
        self.execution_time = execution_time
        self.purchase_p = purchase_p
        self.price = price

class History(db.Model):
    status=db.Column(db.Text)
    symbol=db.Column(db.String)
    price=db.Column(db.Integer)
    number_of_shares=db.Column(db.Integer)
    time=db.Column(db.TIMESTAMP)
    user_id=db.Column(db.Integer, primary_key=True)

    def __init__(self, user_id, symbol, price, number_of_shares, time, status):
        self.user_id = user_id
        self.symbol = symbol
        self.price = price
        self.number_of_shares = number_of_shares
        self.time = time
        self.status = status

class Build(db.Model):
    user_id=db.Column(db.Integer, primary_key=True)
    stocks=db.Column(db.String)
    date_start=db.Column(db.Date)
    amount=db.Column(db.Integer)
    shortperc=db.Column(db.Integer)
    volatility=db.Column(db.Integer)
    target_return=db.Column(db.Integer)
    date_end=db.Column(db.Date)
    gamma=db.Column(db.Float)

    def __init__(self, user_id, stocks, date_start, amount, shortperc, volatility, target_return, date_end, gamma):
        self.user_id = user_id
        self.stocks = stocks
        self.date_start = date_start
        self.amount = amount
        self.shortperc = shortperc
        self.volatility = volatility
        self.target_return = target_return
        self.date_end = date_end
        self.gamma = gamma

class Test(db.Model):
    start_date=db.Column(db.Date)
    end_date=db.Column(db.Date)
    symbols=db.Column(db.String)
    profit_loss=db.Column(db.Integer)
    user_id=db.Column(db.Integer, primary_key=True)

    def __init__(self, start_date, end_date, symbols, profit_loss, user_id):
        self.user_id = user_id
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.profit_loss = profit_loss


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter
app.jinja_env.filters["usd"] = usd
pd.set_option('display.precision', 7)
nasdaq_exchange_info_dict=mc.get("nasdaq_exchange_info_dict")
nasdaq_exchange_info = mc.get("nasdaq_exchange_info")
Session(app)

@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    #quering for the symbol and the corresponding sum of the same stock and the average price paid for the stock
    stocks = db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares'), func.avg(Records.purchase_p).label('purchase_p')).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()

    if not stocks:
        stocks = []
        availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        grandTotal = availableCash
        totalPortValue = 0
        totalprolos = 0

    else:
        start= time.time()
        stocks = [r._asdict() for r in stocks]
        cash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        totalPortValue = 0
        totalprolos = 0
        #print(nasdaq_exchange_info_dict)

        #building the index
        for stock in stocks:
            price = price_lookup(stock['symbol'])
            stock["name"] = nasdaq_exchange_info_dict.get(stock['symbol'], 'no')
            #check if the stock is listed in the UK
            if ".L" in stock["symbol"]:
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
        end = time.time()
        print(end - start)

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
            price = price_lookup(request.form.get("symbol"))
            symbol = request.form.get("symbol")
            availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
            if ".L" in symbol:
                price=GBPtoUSD()*price
            sharesPrice = price * int(request.form.get("shares"))
            if sharesPrice > availableCash:
                flash("Not enough money")
                return redirect ("/buy")
            else:
                formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #insert new row in the database to record the purchase
                new_history=History(session["user_id"], symbol, price, request.form.get("shares"), formatted_date, 'purchase')
                db.session.add(new_history)
                new_record=Records(session["user_id"], symbol, request.form.get("shares"), 'purchase', formatted_date, price, price*int(request.form.get("shares")))
                db.session.add(new_record)
                Users.query.filter_by(id=session["user_id"]).update({'cash': availableCash-sharesPrice})
                db.session.commit()
                return redirect("/")
    else:
        nasdaq_exchange_info = mc.get("nasdaq_exchange_info")
        return render_template("buy.html", nasdaq_exchange_info=nasdaq_exchange_info)


@app.route("/history", methods=["GET", "POST"])
@login_required
def history():
    """Show history of transactions"""
    if request.method == "POST":
        if request.form.get("start")==request.form.get("end"):
            history = db.session.query(History.status, History.symbol, History.price, History.number_of_shares, History.time).filter_by(user_id=session["user_id"]).filter(History.time >= request.form.get("start")).all()
        else:
            history = db.session.query(History.status, History.symbol, History.price, History.number_of_shares, History.time).filter_by(user_id=session["user_id"]).filter(History.time >= request.form.get("start"), History.time <= request.form.get("end")).all()
        if not history:
           return render_template("history.html")
        else:
            hist=[r._asdict() for r in history]
        return render_template("history1.html", hist=hist)
    else:
        try:
            #get the earlist date of the history records
            edate=db.session.query(History.time).filter_by(user_id=session["user_id"]).order_by(History.time).first().time.strftime('%Y-%m-%d')
            print(type(edate))
            #get the last date of the history records
            ldate=db.session.query(History.time).filter_by(user_id=session["user_id"]).order_by(desc(History.time)).first().time.strftime('%Y-%m-%d')
            return render_template("history.html", edate=edate, ldate=ldate)
        except AttributeError:
            flash('No history yet')
            return redirect ("/")

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
        try:
            user = Users.query.filter_by(username=request.form.get("username")).first()
            user.username == request.form.get("username")
        except AttributeError:
            flash("invalid username and/or password")
            return redirect("/login")
        # Ensure username exists and password is correct
        if not check_password_hash(user.hash, request.form.get("password")):
            flash("invalid username and/or password")
            return redirect("/login")
        # Remember which user has logged in
        session["user_id"] = user.id
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
        print(quote)
        #if ticker does not exist
        try:
            if quote["symbol"] is None:
                flash("No such symbol")
                return redirect("/quote")
        except TypeError:
            flash("No symbol entered")
            return redirect("/quote")
        else:
            symbol=quote["symbol"]
            price=quote['regularMarketPrice']
            #if the stock is listed in the United Kingdon add £ sign
            if ".L" in symbol:
                price=gbp(price)
            if "." not in symbol:
                price=usd(price)
            return render_template("quoted.html", name=quote["shortName"], price=price, symbol=symbol)
    else:

        return render_template("quote.html", nasdaq_exchange_info=nasdaq_exchange_info)

@app.route("/quoted")
@login_required
def quoted():
    return render_template("quoted.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    if request.method == "POST":

        username = str(request.form.get("username"))
        if not username:
            flash("Missing name")
            return redirect("/register")
        try:
            findName=Users.query.filter_by(username=request.form.get("username")).first()
            if username == findName:
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
            hash = generate_password_hash(password)
            new_user=Users(username, hash, cash=10000)
            db.session.add(new_user)
            db.session.commit()
            #Store user id in db
            session["user_id"] = Users.query.filter_by(username=request.form.get("username")).first().id
            return redirect("/")

    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    if request.method == "POST":
        symbolToSell=request.form.get("symbol")
        NumOfShares =db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares')).filter_by(user_id=session["user_id"], symbol=str(request.form.get("symbol")), transaction_type='purchase').group_by(Records.symbol).all()
        NumOfShares=[{'symbol': a, 'sumshare': b} for a, b in NumOfShares]
        NumOfshareToSell = int(request.form.get("shares"))
        if NumOfshareToSell > int(NumOfShares[0]["sumshare"]):
            flash("You don't have enough stocks")
            return redirect("/sell")
        else:
            cash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
            price = float(lookup(request.form.get("symbol"))['regularMarketPrice'])
            if ".L" in symbolToSell:
                price=GBPtoUSD()*price
            moneyBack = NumOfshareToSell*price
            #setting up the time stamp to enter the sell into transaction table
            formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #entering the sale into history table
            new_history=History(session["user_id"], request.form.get("symbol"), price, int(request.form.get("shares")), formatted_date, 'sell')
            db.session.add(new_history)
            Users.query.filter_by(id=session["user_id"]).update({'cash':cash+moneyBack})
            db.session.commit()
            if NumOfshareToSell == int(NumOfShares[0]["sumshare"]):
                Records.query.filter_by(user_id=session["user_id"], symbol=request.form.get("symbol")).delete()
                db.session.commit()
            else:
                Records.query.filter_by(user_id=session["user_id"], symbol=request.form.get("symbol")).update({'number_of_shares' : int(NumOfShares[0]["sumshare"])-int(request.form.get("shares"))})
            db.session.commit()
            return redirect("/")
    else:
        stocks = db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares')).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()
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
        symbols = request.form.get("symbols")
        mc.set("symbols", symbols)
        if contains_multiple_words(symbols) == False:
            flash("The app purpose is to optimize a portfolio given a list of stocks. Please enter a list of stocks seperated by a new row.")
            return redirect("/build")
        Build(session["user_id"], symbols.upper(), request.form.get("start"), request.form.get("end"), request.form.get("funds"), request.form.get("short"), request.form.get("volatility"), request.form.get("gamma"), request.form.get("return"))
        db.session.commit()
        try:
            df = yf.download(symbols, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all')
            failed=(list(shared._ERRORS.keys()))
            df = df.replace(0, np.nan)
            try:
                listofna=df.columns[df.isna().iloc[-2]].tolist()+failed
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
        fig.update_layout(width=1350, height=900)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        exp_cov = risk_models.exp_cov(prices, frequency=252)

        #plotting the covariance matrix
        heat = go.Heatmap(
            z = risk_models.cov_to_corr(exp_cov),
            x = exp_cov.columns.values,
            y = exp_cov.columns.values,
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
            width=800,
            height=800,
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
            width=800,
            height=800,
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
        latest_prices1 = df.iloc[-1]  # prices as of the day you are allocating
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
        try:
            es.efficient_return(float(request.form.get("return"))/100)
        except ValueError as e:
            flash(str(e))
            return redirect("/build")
        perf2=es.portfolio_performance()
        weights = es.clean_weights()

        #if we want to buy the portfolio mentioned above
        for col in df.columns:
            if col.endswith(".L"):
                df.loc[:,col] = df.loc[:,col]*GBPtoUSD()
        latest_prices2 = df.iloc[-1]  # prices as of the day you are allocating
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
        mc.delete("symbols")
        return render_template ("built.html",av=av, leftover=leftover, alloc=alloc, ret=float(request.form.get("return")),gamma=request.form.get("gamma"),volatility=request.form.get("volatility"),perf=perf, perf2=perf2, alloc1=alloc1, alloc2=alloc2, plot_json=plot_json, plot_json1=plot_json1, plot_json2=plot_json2, plot_json3=plot_json3, plot_json4=plot_json4, plot_json5=plot_json5, leftover1=leftover1, leftover2=leftover2,listofna=(', '.join(listofna)))
    else:
        if mc.get("symbols"):
            cached_symbols=mc.get("symbols")
        else:
            cached_symbols=''
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        nasdaq_exchange_info=mc.get("nasdaq_exchange_info")
        return render_template("build.html", availableCash=round(availableCash, 4), GBP=GBPtoUSD(), nasdaq_exchange_info=nasdaq_exchange_info, cached_symbols=cached_symbols)

@app.route("/allocation", methods=["POST"])
@login_required
def allocation():
    if request.form.get('demo') == 'buy':
        alloc=session['alloc']
        latest_prices=session['latest_prices']
        values = alloc.values()
        total = sum(values)
        for key, value in alloc.items():
            price=latest_prices[key]
            amount=value
            availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
            sharesPrice = price * amount
            if sharesPrice > availableCash:
                flash("Not enough money to buy:", str(key))
                return redirect ("/built")

            else:
                if ".L" in key:
                    price=GBPtoUSD()*price
                formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #insert new row in the database to record the purchase
                new_history=History(session["user_id"], key, price, int(amount), formatted_date, 'purchase')
                db.session.add(new_history)
                new_record=Records(session["user_id"], key, int(amount), 'purchase', formatted_date, price, price*int(amount))
                db.session.add(new_record)
                Users.query.filter_by(id=session["user_id"]).update({'cash':availableCash-(amount*price)})
                db.session.commit()
    else:
        #alloc=session['alloc']
        #for key, value in alloc.items():
        portfolio = trading212.get_portfolio_composition()
        print(portfolio)
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
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            new_history=History(session["user_id"], key, price, int(amount), formatted_date, 'purchase')
            db.session.add(new_history)
            new_record=Records(session["user_id"], key, int(amount), 'purchase', formatted_date, price, price*int(amount))
            db.session.add(new_record)
            Users.query.filter_by(id=session["user_id"]).update({'cash':availableCash-(amount*price)})
            db.session.commit()

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
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            new_history=History(session["user_id"], key, price, int(amount), formatted_date, 'purchase')
            db.session.add(new_history)
            new_record=Records(session["user_id"], key, int(amount), 'purchase', formatted_date, price, price*int(amount))
            db.session.add(new_record)
            Users.query.filter_by(id=session["user_id"]).update({'cash':availableCash-(amount*price)})
            db.session.commit()

    return redirect("/")

@app.route("/sell_all", methods=["GET", "POST"])
@login_required
def sell_all():
    numberofShares=db.session.query(Records.symbol, Records.number_of_shares).filter_by(user_id=session["user_id"]).all()
    #numberofShares=[{'symbol': a, 'number_of_shares': b} for a, b in numberofShares]
    for stock in numberofShares:
        price=price_lookup(stock.symbol)
        cash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        if ".L" in stock.symbol:
            price=GBPtoUSD()*price
        moneyBack = (int(stock.number_of_shares))*price
        #setting up the time stamp to enter the sell into transaction table
        formatted_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #entering the sale into history table
        new_history=History(session["user_id"], stock.symbol, price, stock.number_of_shares, formatted_date, 'sell')
        db.session.add(new_history)
        Users.query.filter_by(id=session["user_id"]).update({'cash':cash+moneyBack})
        db.session.commit()
    Records.query.filter_by(user_id=session["user_id"]).delete()
    db.session.commit()
    return redirect("/")

@app.route("/test", methods=["GET", "POST"])
@login_required
def test():
    if request.method == "POST":
        if request.form.get("data") == 'sp500':
            nasdaq_exchange_info = pd.read_csv('s&p500.csv')
            nasdaq_exchange_info = list(nasdaq_exchange_info['Symbol'])
        elif request.form.get("data") == 'nasdaq':
            nasdaq_exchange_info = pd.read_csv('nasdaqlisted.txt', sep = '|')
            nasdaq_exchange_info = list(nasdaq_exchange_info['Symbol'])
        else:
            nasdaq_exchange_info = pd.read_csv('otherlisted.txt', sep = '|')
            nasdaq_exchange_info = list(nasdaq_exchange_info['NASDAQ Symbol'])
        #nasdaq_exchange_info=nasdaq_exchange_info['Symbol'].tolist()
        #selected="AAPL MSFT A V TSLA GOOGL BA TEVA OXY AIG BABA TWTR"
        i=0
        for i in range(10):
            selected=random.sample(nasdaq_exchange_info, int(request.form.get("random")))
            #print('selected:', selected)
            try:
                df = yf.download(selected, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all').sort_values('Date')
                #df.to_csv('dfcsv.csv')
                #tobefixed = []
                #for column in df:
                    #print(df[column][0])
                    #print(df[column].isnull().sum() * 100 / len(df[column]))
                    #print('first value is:',df[column][0])
                    #if df[column].isnull().sum() * 100 / len(df[column]) > 0.4 and np.isfinite(df[column][0]):
                        #print(column)
                        #tobefixed.append(column)
                        #fixed = yf.download(column, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"]
                        #print('the new null ratio is:', fixed.isnull().sum() * 100 / len(fixed))
                        #fixed.to_csv(column+'.csv')
                #print('tobefixed:', tobefixed)
                #fixed = yf.download(tobefixed, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all')
                #for column in fixed:
                    #print(fixed[column].isnull().sum() * 100 / len(fixed[column]))
                failed=(list(shared._ERRORS.keys()))
                df = df.replace(0, np.nan)
                try:
                    listofna=df.columns[df.isna().iloc[-2]].tolist()+failed
                except IndexError:
                    flash("Please enter valid stocks from Yahoo Finance.")
                    return redirect("/test")
                df = df.loc[:,df.iloc[-2,:].notna()]
                #print(df)
            except ValueError:
                flash("Please enter a valid symbols (taken from Yahoo Finance)")
                return redirect("/test")
            prices = df.copy()
            #fig = px.line(prices, x=prices.index, y=prices.columns, title='Price Graph')
            #fig = fig.update_xaxes(rangeslider_visible=True)
            #fig.update_layout(width=1350, height=900)
            #plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            mu = pypfopt.expected_returns.capm_return(prices)
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            try:
                ef = EfficientFrontier(mu, S)
                ef.add_objective(objective_functions.L2_reg, gamma=(request.form.get("gamma")))  # gamme is the tuning parameter
                ef.efficient_risk(25)
                weights = ef.clean_weights()
                su = pd.DataFrame([weights])
                perf =ef.portfolio_performance()
            except Exception as e:
                flash(str(e))
                return redirect("/test")
            #for col in df.columns:
                #if col.endswith(".L"):
                    #df.loc[:,col] = df.loc[:,col]*GBPtoUSD()
            latest_prices1 = df.iloc[-1]  # prices as of the day you are allocating
            da = DiscreteAllocation(weights, latest_prices1, total_portfolio_value=10000)
            alloc1, leftover1 = da.lp_portfolio()
            #print('latest_prices1:', latest_prices1)

            totalprice=0
            totalnewprice=0
            for key, value in alloc1.items():
                price=latest_prices1[key]
                sharesPrice = price * int(value)
                todayprice = price_lookup(key)
                totalnewprice += todayprice*int(value)
                #if ".L" in key:
                    #price=GBPtoUSD()*price
                totalprice += sharesPrice
            profitloss=totalnewprice-totalprice+leftover1
            new_test=Test(request.form.get("start"), request.form.get("end"), list(alloc1.keys()), profitloss, session["user_id"])
            db.session.add(new_test)
            db.session.commit()
            i += 1
        return render_template("test1.html", plot_json=plot_json, listofna=listofna, profitloss=profitloss, alloc1=alloc1)
    else:
        return render_template("test.html")



def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    flash(e.name)
    return redirect("/")


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
