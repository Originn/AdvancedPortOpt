from flask import current_app as app
from flask import render_template
from helpers import login_required
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from flask import Flask, flash, redirect, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
import re, datetime, time, bmemcached, os, json, plotly
from werkzeug.security import check_password_hash, generate_password_hash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import func, cast, Date, desc
from helpers import price_lookup, clean_header, usd, GBPtoUSD, contains_multiple_words, lookup, gbp
import yfinance as yf
from datetime import datetime
from flask_session import Session
from models import db, Records, Users, History, Build, Test, Stocks
import random
import yfinance.shared as shared
import numpy as np
import pypfopt
from pypfopt import risk_models, DiscreteAllocation, objective_functions, EfficientSemivariance, efficient_frontier, EfficientFrontier, HRPOpt
from pypfopt import EfficientFrontier
from multiprocessing import Process
from pandas_datareader import data


yf.pdr_override()

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response



pd.set_option('display.precision', 7)
app.jinja_env.filters["usd"] = usd
servers = os.environ.get('MEMCACHIER_SERVERS', '').split(',')
user = os.environ.get('MEMCACHIER_USERNAME', '')
passw = os.environ.get('MEMCACHIER_PASSWORD', '')

mc = bmemcached.Client(servers, username=user, password=passw)
mc.enable_retry_delay(True)

nasdaq_exchange_info_dict=mc.get("nasdaq_exchange_info_dict")
nasdaq_exchange_info = mc.get("nasdaq_exchange_info")
top_50_crypto=mc.get("top_50_crypto")
top_world_stocks = mc.get("top_world")
top_US_stocks= mc.get("top_US")
top_div = mc.get("top_div")
users_stocks = [[sn, s] for sn, s in db.session.query(Stocks.shortname, Stocks.symbol)]
nasdaq_exchange_info.extend(users_stocks)

Session(app)
#app = init_dashboard(app)

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
            symbol = (request.form.get("symbol")).upper()
            availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
            if ".L" in symbol:
                price=GBPtoUSD()*price
            sharesPrice = price * int(request.form.get("shares"))
            if sharesPrice > availableCash:
                flash("Not enough money")
                return redirect ("/buy")
            else:
                history = db.session.query(History.symbol, History.number_of_shares, History.cml_cost, History.cml_units).filter_by(user_id=session["user_id"]).all()
                history = pd.DataFrame(history)
                try:
                    cml_units = history.query('symbol==@symbol').iloc[-1,-1] + int(request.form.get("shares"))
                except:
                    cml_units = int(request.form.get("shares"))
                try:
                    cml_cost = history[history['symbol']==symbol].tail(1).reset_index().loc[0, 'cml_cost'] + sharesPrice
                except:
                    cml_cost = sharesPrice
                cost_unit = 0
                cost_transact = 0
                avg_price = cml_cost/cml_units
                #####I need to figure out why the valus are not as intended on the sql table
                formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #insert new row in the database to record the purchase
                new_history=History(session["user_id"], symbol, price, request.form.get("shares"), formatted_date, 'purchase', 0, round(cml_cost, 2), round(-(sharesPrice), 2), round(avg_price, 2), cost_unit, round(cost_transact, 2), int(cml_units))
                db.session.add(new_history)
                new_record=Records(session["user_id"], symbol, request.form.get("shares"), 'purchase', price, round(price*int(request.form.get("shares")), 2), formatted_date, None)
                db.session.add(new_record)
                Users.query.filter_by(id=session["user_id"]).update({'cash': availableCash-sharesPrice})
                db.session.commit()
                return redirect("/")
    else:
        global nasdaq_exchange_info
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
            flash("invalid username please register")
            return redirect("/login")
        # Ensure username exists and password is correct
        if not check_password_hash(user.hash, request.form.get("password")):
            flash("invalid username and/or password")
            return redirect("/login")
        # Remember which user has logged in
        session["user_id"] = user.id
        session["user_name"] = user.username
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
            if findName:
                flash("User name already taken")
                return redirect("/register")
        except:
            pass


        password = request.form.get("password")
        if not password:
            flash("Missing password")
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
        NumOfshareToSell = int(request.form.get("shares"))
        if NumOfshareToSell > NumOfShares[0][1]:
            flash("You don't have enough stocks")
            return redirect("/sell")
        else:
            share_data=db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares'), Records.execution_time, Records.purchase_p).filter_by(user_id=session["user_id"], symbol=str(request.form.get("symbol")), transaction_type='purchase').group_by(Records.symbol,  Records.execution_time, Records.purchase_p).order_by(Records.execution_time).all()
            number_of_items=len(share_data)
            #selling FIFO method first in first out
            for i in range(number_of_items):
                if NumOfshareToSell < share_data[0][1]:
                    history = db.session.query(History.symbol, History.status, History.number_of_shares, History.cml_cost, History.cml_units, History.avg_price).filter_by(user_id=session["user_id"]).all()
                    history = pd.DataFrame(history)
                    price = float(price_lookup(request.form.get("symbol")))
                    if ".L" in symbolToSell:
                        price = GBPtoUSD()*price
                    cml_units = history.query('symbol==@symbolToSell').iloc[-1,4] - NumOfshareToSell
                    cost_unit = history.query('symbol==@symbolToSell').iloc[-1,-1]
                    cost_transact = cost_unit*NumOfshareToSell
                    cml_cost = history.query('symbol == @symbolToSell').iloc[-1,3] - cost_transact
                    avg_price = cml_cost/cml_units
                    moneyBack = price*NumOfshareToSell
                    gain_loss = price*NumOfshareToSell - share_data[0][3]*NumOfshareToSell
                    formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    Records.query.filter_by(user_id=session["user_id"], symbol=request.form.get("symbol"), execution_time=share_data[0][2]).update({'number_of_shares' :  share_data[0][1] - NumOfshareToSell})
                    new_history=History(session["user_id"], request.form.get("symbol"), price, -(NumOfshareToSell), formatted_date, 'sell', round(gain_loss, 2), round(cml_cost, 2), round(price*NumOfshareToSell, 2), round(avg_price, 2), cost_unit, cost_transact, int(cml_units))
                    db.session.add(new_history)
                    Users.query.filter_by(id=session["user_id"]).update({'cash':Users.cash+moneyBack+gain_loss})
                    db.session.commit()
                    break
                else:
                    history = db.session.query(History.symbol, History.status, History.number_of_shares, History.cml_cost, History.cml_units, History.avg_price).filter_by(user_id=session["user_id"]).all()
                    history = pd.DataFrame(history)
                    price = float(price_lookup(request.form.get("symbol")))
                    if ".L" in symbolToSell:
                        price=GBPtoUSD()*price
                    cml_units=history.query('symbol==@symbolToSell').iloc[-1,4] - share_data[0][1]
                    cost_unit=history.query('symbol==@symbolToSell').iloc[-1,-1]
                    cost_transact = cost_unit*share_data[0][1]
                    cml_cost=history.query('symbol == @symbolToSell').iloc[-1,3] - cost_transact
                    if cml_units == 0:
                        avg_price = 0
                    else:
                        avg_price = cml_cost/cml_units
                    moneyBack = price*share_data[0][1]
                    gain_loss=price*share_data[0][1] - share_data[0][3]*share_data[0][1]
                    formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    Records.query.filter_by(user_id=session["user_id"], symbol=request.form.get("symbol"), execution_time=share_data[0][2]).delete()
                    Users.query.filter_by(id=session["user_id"]).update({'cash': Users.cash + moneyBack + gain_loss})
                    new_history=History(session["user_id"], request.form.get("symbol"), price, -(share_data[0][1]), formatted_date, 'sell', round(gain_loss, 2), round(cml_cost, 2), round(price*share_data[0][1], 2), round(avg_price, 2), round(cost_unit, 2), round(cost_transact, 2), int(cml_units))
                    db.session.add(new_history)
                    try:
                        pop=list(share_data.pop(0))
                        NumOfshareToSell=NumOfshareToSell-pop[1]
                    except:
                        break
                    db.session.commit()

            return redirect("/")
    else:
        stocks = db.session.query(Records.symbol).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()
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

        def enter_sql_data(app, df, nasdaq_exchange_info, Stocks):
            for ticker in df.columns:
                ticker=ticker.upper()
                if any(sublist[1]==ticker in sublist for sublist in nasdaq_exchange_info) is False:
                    ticker_ln = yf.Ticker(ticker).stats()["price"].get('longName')
                    if not ticker_ln:
                        ticker_ln = ticker
                    ticker_list=[ticker_ln, ticker]
                    with app.app_context():
                        new_stock=Stocks(ticker, ticker_ln)
                        db.session.add(new_stock)
                        db.session.commit()
                    nasdaq_exchange_info.extend([ticker_list])
        global nasdaq_exchange_info
        app1 = app._get_current_object()
        p1 = Process(target=enter_sql_data, args=[app1, df, nasdaq_exchange_info, Stocks])
        p1.start()

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


        #using risk models optimized for the Efficient frontier to reduce to min volitility, good for crypto currencies - not implemented in the website now.
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
        print('top_US_stocks', top_US_stocks)
        return render_template("build.html", availableCash=round(availableCash, 4), GBP=GBPtoUSD(), nasdaq_exchange_info=nasdaq_exchange_info, cached_symbols=cached_symbols, top_50_crypto=top_50_crypto, top_world_stocks=top_world_stocks, top_US_stocks=top_US_stocks, top_div=top_div)

@app.route("/allocation1", methods=["GET", "POST"])
@login_required
def allocation1():
    alloc1=session['alloc1']
    for key, value in alloc1.items():
        price=price_lookup(key)
        amount=value
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            history = db.session.query(History.symbol, History.number_of_shares, History.cml_cost, History.cml_units).filter_by(user_id=session["user_id"]).all()
            history = pd.DataFrame(history)
            try:
                cml_units = history.query('symbol==@key').iloc[-1,-1] + amount
            except:
                cml_units = amount
            try:
                cml_cost = history[history['symbol']==key].tail(1).reset_index().loc[0, 'cml_cost'] + sharesPrice
            except:
                cml_cost = sharesPrice
            cost_unit = 0
            cost_transact = 0
            avg_price = cml_cost/cml_units

            new_history=History(session["user_id"], key, price, int(amount), formatted_date, 'purchase', 0, round(cml_cost, 2), round(-(sharesPrice), 2), round(avg_price, 2), cost_unit, round(cost_transact, 2), int(cml_units))
            db.session.add(new_history)
            new_record=Records(session["user_id"], key, int(amount), 'purchase', price, price*int(amount), formatted_date, None)
            db.session.add(new_record)
            Users.query.filter_by(id=session["user_id"]).update({'cash':availableCash-(amount*price)})

    db.session.commit()

    return redirect("/")

@app.route("/allocation2", methods=["GET", "POST"])
@login_required
def allocation2():
    alloc2=session['alloc2']
    for key, value in alloc2.items():
        price=price_lookup(key)
        amount=value
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        sharesPrice = price * amount
        if sharesPrice > availableCash:
            flash("Not enough money to buy:", str(key))
            return redirect ("/built")

        else:
            if ".L" in key:
                price=GBPtoUSD()*price
            formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #insert new row in the database to record the purchase
            history = db.session.query(History.symbol, History.number_of_shares, History.cml_cost, History.cml_units).filter_by(user_id=session["user_id"]).all()
            history = pd.DataFrame(history)
            try:
                cml_units = history.query('symbol==@key').iloc[-1,-1] + amount
            except:
                cml_units = amount
            try:
                cml_cost = history[history['symbol']==key].tail(1).reset_index().loc[0, 'cml_cost'] + sharesPrice
            except:
                cml_cost = sharesPrice
            cost_unit = 0
            cost_transact = 0
            avg_price = cml_cost/cml_units

            new_history=History(session["user_id"], key, price, int(amount), formatted_date, 'purchase', 0, round(cml_cost, 2), round(-(sharesPrice), 2), round(avg_price, 2), cost_unit, round(cost_transact, 2), int(cml_units))
            db.session.add(new_history)
            new_record=Records(session["user_id"], key, int(amount), 'purchase', price, price*int(amount), formatted_date, None)
            db.session.add(new_record)
            Users.query.filter_by(id=session["user_id"]).update({'cash':availableCash-(amount*price)})

    db.session.commit()

    return redirect("/")

@app.route("/sell_all", methods=["GET", "POST"])
@login_required
def sell_all():
    share_data=db.session.query(Records.symbol, Records.number_of_shares, Records.purchase_p).filter_by(user_id=session["user_id"]).all()
    history = db.session.query(History.symbol, History.avg_price).filter_by(user_id=session["user_id"]).all()
    history = pd.DataFrame(history)
    for stock in share_data:
        price=price_lookup(stock.symbol)
        if ".L" in stock.symbol:
            price=GBPtoUSD()*price
        moneyBack = stock[1]*price
        profit=price*stock[1]-stock[1]*stock[2]
        cml_units = 0
        avg_price = 0
        cml_cost = 0
        cost_unit = history.query('symbol==@stock.symbol').iloc[-1,-1]
        cost_transact = cost_unit*stock.number_of_shares
        #setting up the time stamp to enter the sell into transaction table
        formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #entering the sale into history table
        new_history=History(session["user_id"], stock.symbol, price, stock.number_of_shares, formatted_date, 'sell', profit, cml_cost, round(price*stock.number_of_shares, 2), avg_price, cost_unit, round(cost_transact, 2), cml_units)
        db.session.add(new_history)
        Users.query.filter_by(id=session["user_id"]).update({'cash': Users.cash + moneyBack + profit})
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
        i=0
        for i in range(4):
            method='semi-variance'
            selected=random.sample(nasdaq_exchange_info, int(request.form.get("random")))
            try:
                df = yf.download(selected, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all').sort_values('Date')
                #another download to check the time from last download to today in order to check if the target was reached in any of the days
                dftest=yf.download(selected, start=request.form.get("end"), end=datetime.today().strftime('%Y-%m-%d'), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all').sort_values('Date')
                failed=(list(shared._ERRORS.keys()))
                df = df.replace(0, np.nan)
                try:
                    listofna=df.columns[df.isna().iloc[-2]].tolist()+failed
                except IndexError:
                    flash("Please enter valid stocks from Yahoo Finance.")
                    return redirect("/test")
                df = df.loc[:,df.iloc[-2,:].notna()]
            except ValueError:
                flash("Please enter a valid symbols (taken from Yahoo Finance)")
                return redirect("/test")
            prices = df.copy()
            mu = pypfopt.expected_returns.capm_return(prices)
            returns = pypfopt.expected_returns.returns_from_prices(prices)
            returns = returns.dropna()
            es = EfficientSemivariance(mu, returns)
            try:
                es.efficient_return(0.15)
            except ValueError as e:
                flash(str(e))
                return redirect("/test")
            perf2=es.portfolio_performance()
            weights = es.clean_weights()
            gamma=None
            latest_prices2 = df.iloc[-1]  # prices as of the day you are allocating
            da = DiscreteAllocation(weights, latest_prices2, total_portfolio_value=10000)
            alloc2, leftover2 = da.lp_portfolio()
            #create a df where we can test if the portfolio reached the desired yield.
            dftest=dftest.loc[:, dftest.columns.isin(list(alloc2.keys()))]
            #create a total value df where we can see the value of the portfolio on each day
            df1 = dftest.dot(pd.Series(alloc2))+leftover2
            print(df1)
            max_profit_value=df1.max()-10000
            totalprice=0
            totalnewprice=0
            #check what is the value of the stocks today
            for key, value in alloc2.items():
                price=latest_prices2[key]
                sharesPrice = price * int(value)
                todayprice = price_lookup(key)
                totalnewprice += todayprice*int(value)
                if ".L" in key:
                    price=GBPtoUSD()*price
                totalprice += sharesPrice
            #what is thevalue of the portfolio today
            profitloss=totalnewprice-totalprice+leftover2
            print(profitloss)
            input()
            new_test=Test(request.form.get("start"), request.form.get("end"), list(alloc2.keys()), profitloss, session["user_id"], profit_date, method, max_profit_value, target_profit, gamma)
            db.session.add(new_test)
            db.session.commit()
            i += 1
        return render_template("test1.html", listofna=listofna, profitloss=profitloss, alloc2=alloc2)
    else:
        return render_template("test.html")


@app.route("/dash")
@login_required
def redirect_to_dashapp():
    ### I will  need to use the History SQL table in order to find gain_loss
    return redirect('/dashapp/')



def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    flash(e.name)
    return redirect("/")


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
