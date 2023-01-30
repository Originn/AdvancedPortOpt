from flask import current_app as app
from flask import render_template, g
from application.helpers import login_required, price_lookup, clean_header, usd, GBPtoUSD, contains_multiple_words, lookup, gbp, mc
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from flask import Flask, flash, redirect, render_template, request, session, url_for, copy_current_request_context, jsonify
from flask_sqlalchemy import SQLAlchemy
import re, datetime, time, bmemcached, os, json, plotly
from werkzeug.security import check_password_hash, generate_password_hash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import func, cast, Date, desc, null
import yfinance as yf
from datetime import datetime
from application.db import db, Records, Users, History, Build, Test, Stocks
import random
import yfinance.shared as shared
import numpy as np
import pypfopt
from pypfopt import risk_models, DiscreteAllocation, objective_functions, EfficientSemivariance, efficient_frontier, EfficientFrontier, HRPOpt, EfficientCVaR
from pypfopt import EfficientFrontier
from pandas_datareader import data
import kthread
from datetime import datetime, timedelta
from pandas.tseries.offsets import Day
import plotly.subplots as subplots
import pandas_datareader.data as web

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

nasdaq_exchange_info_dict=mc.get("nasdaq_exchange_info_dict")
nasdaq_exchange_info = mc.get("nasdaq_exchange_info")
top_50_crypto=mc.get("top_50_crypto")
top_world_stocks = mc.get("top_world")
top_US_stocks= mc.get("top_US")
top_div = mc.get("top_div")
win_loss_trend = mc.get("win_loss_trend")
win_loss_signal= mc.get("win_loss_signal")

users_stocks = [[sn, s] for sn, s in db.session.query(Stocks.shortname, Stocks.symbol)]
nasdaq_exchange_info.extend(users_stocks)

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
        stocks = [r._asdict() for r in stocks]
        cash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        totalPortValue = 0
        totalprolos = 0

        #building the index
        for stock in stocks:
            try:
                price = float(web.get_quote_yahoo(stock['symbol'])['preMarketPrice'])
            except:
                price = price_lookup(stock['symbol'])
            stock["name"] = nasdaq_exchange_info_dict.get(stock['symbol'], stock['symbol'])
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
#@profile(stream=fp)
def build():
    if request.method == "POST":
        global_dict = {}
        userId = session['user_id']
        global_dict[int(userId)] = {}
        global_dict[int(userId)]['finished'] = 'False'
        tickers = request.form.get("symbols")
        tickers = list(set(tickers.split()))
        @copy_current_request_context
        def operation(global_dict, session):
            symbols = request.form.get("symbols")
            mc.set(str(userId) + "_symbols", symbols)
            symbols = list(set(symbols.split()))
            if contains_multiple_words(symbols) == False:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = "The app purpose is to optimize a portfolio given a list of stocks. Please enter a list of stocks seperated by a new row."
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return
            if float(request.form.get("funds")) <= 0 or float(request.form.get("funds")) == " ":
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = "Amount need to be a positive number"
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return
            Build(session["user_id"], request.form.get("symbols").upper(), request.form.get("start"), request.form.get("end"), request.form.get("funds"), request.form.get("short"), request.form.get("volatility"), request.form.get("gamma"), request.form.get("return"))
            db.session.commit()
            try:
                mc.set(str(userId)+'start_date', request.form.get("start"))
                df = yf.download(symbols, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all')
                failed=(list(shared._ERRORS.keys()))
                df = df.replace(0, np.nan)
                try:
                    global_dict[int(userId)]['listofna']=df.columns[df.isna().iloc[-2]].tolist()+failed
                except IndexError:
                    global_dict[int(userId)]['finished'] = 'True'
                    global_dict[int(userId)]['error'] = "Please enter valid stocks from Yahoo Finance."
                    mc.set("user_dict", global_dict[int(userId)])
                    del global_dict[int(userId)]
                    return
                df = df.loc[:,df.iloc[-2,:].notna()]
            except ValueError:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = "Please enter a valid symbols (taken from Yahoo Finance)"
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return

            prices = df.copy()
            fig = px.line(prices, x=prices.index, y=prices.columns)
            fig = fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(width=1350, height=900, title_text = 'Price Graph', title_x = 0.5)
            global_dict[int(userId)]['plot_json_graph'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


            try:
                S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            except:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = "Could not fix ledoit_wolf matrix. Please try a different risk model."
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return

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
                width=500,
                height=500,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange='reversed'
            )

            fig=go.Figure(data=[heat], layout=layout)
            global_dict[int(userId)]['plot_json_Ledoit_Wolf'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


            #Section 2 -Return estimation
            #it is often a bad idea to provide returns using a simple estimate like the mean of past returns. Research suggests that better off not providing expected returns – you can then just find the min_volatility() portfolio or use HRP.
            mu = pypfopt.expected_returns.capm_return(prices)

            #using risk models optimized for the Efficient frontier to reduce to min volitility, good for crypto currencies ('long only')
            ef = EfficientFrontier(None, S)
            try:
                ef.min_volatility()
                weights = ef.clean_weights()
                nu = pd.Series(weights)
                fig = px.bar(nu, orientation='h')
                fig.update_layout(width=700, height=500, title_text = "Weights for minimum volatility (long only)", title_x = 0.5, showlegend=False, yaxis_title=None, xaxis_title=None)
                global_dict[int(userId)]['plot_json_weights_min_vol_long'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                av=ef.portfolio_performance()[1]
                global_dict[int(userId)]['av_min_vol_long']=round(av*100, 2)


            #if we want to buy the portfolio mentioned above
                for col in prices.columns:
                    if col.endswith(".L"):
                        prices.loc[:,col] = prices.loc[:,col]*GBPtoUSD()
                try:
                    latest_prices = prices.iloc[-1]
                except IndexError:
                    global_dict[int(userId)]['finished'] = 'True'
                    global_dict[int(userId)]['error'] = "There is an issue with Yahoo API please try again later"
                    mc.set("user_dict", global_dict[int(userId)])
                    del global_dict[int(userId)]
                    return
                # prices as of the day you are allocating
                if float(request.form.get("funds")) < float(latest_prices.min()):
                    global_dict[int(userId)]['finished'] = 'True'
                    global_dict[int(userId)]['error'] = "Amount is not high enough to cover the lowest priced stock"
                    mc.set("user_dict", global_dict[int(userId)])
                    del global_dict[int(userId)]
                    return
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
                    mc.set(str(userId) + '_funds', float(request.form.get("funds")))
                except TypeError:
                    delisted=prices.columns[df.isna().any()].tolist()
                    delisted= ", ".join(delisted)
                    global_dict[int(userId)]['finished'] = 'True'
                    global_dict[int(userId)]['error'] = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                    mc.set("user_dict", global_dict[int(userId)])
                    del global_dict[int(userId)]
                    return
                alloc, global_dict[int(userId)]['leftover_min_vol_long'] = da.lp_portfolio()
                global_dict[int(userId)]['alloc_min_vol_long']=alloc
                fig = px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
                fig.update_traces(textposition='inside')
                fig.update_layout(width=500, height=500, uniformtext_minsize=12, uniformtext_mode='hide', title_text='Suggested Portfolio Distribution for min volatility (long)', title_x=0.5)
                global_dict[int(userId)]['plot_json_dist_min_vol_long'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            except ValueError as e:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = str(e)
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return

            #using risk models optimized for the Efficient frontier to reduce to min volitility, good for crypto currencies ('long and short')
            ef = EfficientFrontier(None, S, weight_bounds=(None, None))
            try:
                ef.min_volatility()
                weights = ef.clean_weights()
                nu = pd.Series(weights)
                fig = px.bar(nu, orientation='h')
                fig.update_layout(width=700, height=500, title_text = "Weights for minimum volatility (long/short)", title_x = 0.5, showlegend=False, yaxis_title=None, xaxis_title=None)
                global_dict[int(userId)]['plot_json_weight_min_vol_long_short'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                av=ef.portfolio_performance()[1]
                global_dict[int(userId)]['av']=round(av*100, 2)

            #if we want to buy the portfolio mentioned above
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
                except TypeError:
                    delisted=prices.columns[df.isna().any()].tolist()
                    delisted= ", ".join(delisted)
                    global_dict[int(userId)]['finished'] = 'True'
                    global_dict[int(userId)]['error'] = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                    mc.set("user_dict", global_dict[int(userId)])
                    del global_dict[int(userId)]
                    return
                global_dict[int(userId)]['alloc_min_vol_long_short'], global_dict[int(userId)]['leftover_min_vol_long_short'] = da.lp_portfolio()
            except ValueError as e:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = str(e)
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return

            #Maximise return for a given risk, with L2 regularisation
            mc.set(str(userId)+'_volatility', float(request.form.get("volatility")))
            mc.set(str(userId)+'_gamma', float(request.form.get("gamma")))
            mc.set(str(userId)+'_cvar', request.form.get("cvar"))
            mc.set(str(userId)+'_return', request.form.get("return"))
            try:
                ef = EfficientFrontier(mu, S)
                ef.add_objective(objective_functions.L2_reg, gamma=(float(request.form.get("gamma"))))  # gamme is the tuning parameter
                ef.efficient_risk(float(request.form.get("volatility"))/100)
                weights = ef.clean_weights()
                su = pd.DataFrame([weights])
                #finding zero weights
                num_small = len([k for k in weights if weights[k] <= 1e-4])
                global_dict[int(userId)]['num_small'] = str(f"{num_small}/{len(ef.tickers)} tickers have zero weight")
                fig = px.pie(su, values=weights.values(), names=su.columns)
                fig.update_traces(textposition='inside')
                fig.update_layout(width=500, height=500, uniformtext_minsize=12, uniformtext_mode='hide', title_text='Weights Distribution using Capital Asset Pricing Model', title_x=0.5)
                global_dict[int(userId)]['plot_json_L2_weights'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                global_dict[int(userId)]['perf_L2'] = ef.portfolio_performance()
            except Exception as e:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = str(e)
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return


            #if we want to buy the portfolio mentioned above
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
            alloc, global_dict[int(userId)]['leftover_L2'] = da.lp_portfolio()
            global_dict[int(userId)]['alloc_L2']=alloc
            fig= px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
            fig.update_traces(textposition='inside')
            fig.update_layout(width=500, height=500, uniformtext_minsize=12, font=dict(size=10), uniformtext_mode='hide', title_text='Suggested Portfolio Distribution using Capital Asset Pricing Model', title_x=0.5)
            global_dict[int(userId)]['plot_json_L2_port'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            #Efficient semi-variance optimization
            returns = pypfopt.expected_returns.returns_from_prices(prices)
            returns = returns.dropna()
            es = EfficientSemivariance(mu, returns)
            try:
                es.efficient_return(float(request.form.get("return"))/100)
            except ValueError as e:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = str(e)
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return
            global_dict[int(userId)]['perf_semi_v']=es.portfolio_performance()
            weights = es.clean_weights()

            #if we want to buy the portfolio mentioned above
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
            alloc, global_dict[int(userId)]['leftover_semi_v'] = da.lp_portfolio()
            global_dict[int(userId)]['alloc_semi_v'] = alloc
            fig = px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
            fig.update_traces(textposition='inside')
            fig.update_layout(width=500, height=500, uniformtext_minsize=12, font=dict(size=10), uniformtext_mode='hide', title_text='Suggested Portfolio Distribution using Capital Asset Pricing Model', title_x=0.5)
            global_dict[int(userId)]['plot_json_semi_v'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            global_dict[int(userId)]['ret']=float(request.form.get("return"))
            global_dict[int(userId)]['gamma']=request.form.get("gamma")
            global_dict[int(userId)]['volatility']=request.form.get("volatility")

            #construct the portfolio with the minimum CVaR
            returns =pypfopt.expected_returns.returns_from_prices(prices).dropna()
            global_dict[int(userId)]['cvar_value']=request.form.get("cvar")
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            weight_arr = ef.weights
            portfolio_rets = (returns * weight_arr).sum(axis=1)
            fig = px.histogram(portfolio_rets, nbins = 50)
            fig.update_layout(width=500, height=500, yaxis_title=None, xaxis_title=None, showlegend=False)
            global_dict[int(userId)]['plot_json_cvar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            global_dict[int(userId)]['var'] = portfolio_rets.quantile(0.05)
            global_dict[int(userId)]['cvar'] = portfolio_rets[portfolio_rets <= global_dict[int(userId)]['var']].mean()
            ec = EfficientCVaR(mu, returns)
            try:
                ec.efficient_risk(target_cvar=float(request.form.get("cvar"))/100)
            except:
                global_dict[int(userId)]['finished'] = 'True'
                global_dict[int(userId)]['error'] = f"Please enter CVaR higher than {round(global_dict[int(userId)]['cvar']*(-100), 1)}%"
                mc.set("user_dict", global_dict[int(userId)])
                del global_dict[int(userId)]
                return
            weights = ec.clean_weights()
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(request.form.get("funds")))
            alloc, global_dict[int(userId)]['leftover_cvar'] = da.lp_portfolio()
            fig = px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
            fig.update_layout(title="allocation distribution")
            global_dict[int(userId)]['pie_json_cvar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            global_dict[int(userId)]['alloc_cvar'] = alloc
            global_dict[int(userId)]['target_CVaR_exp_rtn'], global_dict[int(userId)]['target_CVaR_cond_val_risk'] = ec.portfolio_performance()
            global_dict[int(userId)]['finished'] = 'True'
            mc.set("user_dict", global_dict[int(userId)])
            del global_dict[int(userId)]
            t1.terminate()
            return
        t1 = kthread.KThread(target=operation, args=[global_dict, session])
        t1.start()

        @copy_current_request_context
        def enter_sql_data(nasdaq_exchange_info, tickers):
            for ticker in tickers:
                ticker=ticker.upper()
                if any(sublist[1]==ticker in sublist for sublist in nasdaq_exchange_info) is False:
                    try:
                        ticker_ln = yf.Ticker(ticker).stats()["price"].get('longName')
                    except:
                        continue
                    if not ticker_ln:
                        ticker_ln = ticker
                    ticker_list=[ticker_ln, ticker]
                    new_stock=Stocks(ticker, ticker_ln)
                    db.session.add(new_stock)
                    db.session.commit()
                    nasdaq_exchange_info.extend([ticker_list])
                    t2.terminate()
                    return
        #global nasdaq_exchange_info
        t2 = kthread.KThread(target=enter_sql_data, args=[nasdaq_exchange_info, tickers])
        t2.start()
        return render_template("loading.html")
    else:
        try:
            mc.delete("user_dict")
        except:
            pass
        userId = session['user_id']
        cached_symbols = mc.get(str(userId) + "_symbols") if mc.get(str(userId) + "_symbols") else ''
        start_cached = mc.get(str(userId)+'start_date') if mc.get(str(userId)+'start_date') else 0
        funds_cached = mc.get(str(userId) + '_funds') if mc.get(str(userId) + '_funds') else 0
        vol_cached = mc.get(str(userId)+'_volatility') if mc.get(str(userId)+'_volatility') else 0
        gamma_cached = mc.get(str(userId)+'_gamma') if mc.get(str(userId)+'_gamma') else 0
        cvar_cached = mc.get(str(userId)+'_cvar') if mc.get(str(userId)+'_cvar') else 0
        return_cached = mc.get(str(userId)+'_return') if mc.get(str(userId)+'_return') else 0
        availableCash=db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
        return render_template("build.html", availableCash=round(availableCash, 4), GBP=GBPtoUSD(), nasdaq_exchange_info=nasdaq_exchange_info, return_cached = return_cached, cvar_cached = cvar_cached, gamma_cached = gamma_cached, vol_cached = vol_cached, funds_cached = funds_cached, start_cached = start_cached, cached_symbols=cached_symbols, top_50_crypto=top_50_crypto, top_world_stocks=top_world_stocks, top_US_stocks=top_US_stocks, top_div=top_div, win_loss_signal = win_loss_signal, win_loss_trend=win_loss_trend)


@app.route('/result')
@login_required
#@profile(stream=fp)
def result():
    user_dict = mc.get("user_dict")
    userId = session['user_id']
    try:
        return render_template("built.html", num_small=user_dict['num_small'], plot_json_weights_min_vol_long=user_dict['plot_json_weights_min_vol_long'], av_min_vol_long=user_dict['av_min_vol_long'], leftover_min_vol_long=user_dict['leftover_min_vol_long'], alloc_min_vol_long = user_dict['alloc_min_vol_long'], plot_json_dist_min_vol_long=user_dict['plot_json_dist_min_vol_long'], av = user_dict['av'], leftover_min_vol_long_short=user_dict['leftover_min_vol_long_short'], alloc_min_vol_long_short=user_dict['alloc_min_vol_long_short'], ret=user_dict['ret'],gamma=user_dict['gamma'],volatility=user_dict['volatility'], perf_L2=user_dict['perf_L2'], perf_semi_v=user_dict['perf_semi_v'], alloc_L2=user_dict['alloc_L2'], alloc_semi_v=user_dict['alloc_semi_v'], plot_json_graph=user_dict['plot_json_graph'], plot_json_Ledoit_Wolf=user_dict['plot_json_Ledoit_Wolf'], plot_json_weight_min_vol_long_short=user_dict['plot_json_weight_min_vol_long_short'], plot_json_L2_weights=user_dict['plot_json_L2_weights'], plot_json_L2_port = user_dict['plot_json_L2_port'], plot_json_semi_v = user_dict['plot_json_semi_v'], leftover_L2=user_dict['leftover_L2'], leftover_semi_v=user_dict['leftover_semi_v'],listofna=(', '.join(user_dict['listofna'])), min_cvar_rtn = user_dict['target_CVaR_exp_rtn'], min_cvar_risk = user_dict['target_CVaR_cond_val_risk'], var = user_dict['var'], cvar = user_dict['cvar'], plot_json_cvar=user_dict['plot_json_cvar'], pie_json_cvar=user_dict['pie_json_cvar'], cvar_value=user_dict['cvar_value'], alloc_cvar = user_dict['alloc_cvar'], leftover_cvar = user_dict['leftover_cvar'])
    except:
        try:
            return_error = str(user_dict['error'])
            flash(return_error)
            return redirect("/build")
            mc.delete("user_dict")
        except:
            return redirect("/build")
            mc.delete("user_dict")

mc.delete("user_dict")
@app.route('/result_alloc')
@login_required
def result_alloc():
    try:
        time.sleep(2)
        return redirect("/")
    except:
        try:
            return_error = str(user_dict['error'])
            flash(return_error)
            return redirect("/build")
        except:
            return redirect("/build")

@app.route('/status')
@login_required
#@profile(stream=fp)
def thread_status():
    userId = session['user_id']
    user_dict = mc.get("user_dict")
    if user_dict:
        return jsonify(dict(status=('finished' if (user_dict['finished'] == 'True') else 'running')))
    else:
        return render_template('loading.html')

@app.route("/allocation", methods=["POST"])
@login_required
def allocation():
    @copy_current_request_context
    def start_allocation(user_dict, session):
        user_dict['finished'] = 'False'
        alloc=user_dict[request.form.get('form_name')]
        for key, value in alloc.items():
            price=price_lookup(key)
            amount=value
            availableCash=Users.query.filter_by(id=session["user_id"]).first().cash
            sharesPrice = price * amount
            if sharesPrice > availableCash:
                user_dict['error'] = "Not enough money to buy:" + str(key)
                user_dict['finished'] = 'True'
                return
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
        user_dict['finished'] = 'True'
        t3.terminate()
        mc.delete("user_dict")
    user_dict = mc.get("user_dict")
    userId = session['user_id']
    t3 = kthread.KThread(target=start_allocation, args=[user_dict, session], name=str(userId)+'_allocation_thread')
    t3.start()
    return render_template("loading_alloc.html")

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
        symbols = request.form.get("symbols")
        symbols = list(set(symbols.split()))
        sp500_prices = yf.download('^GSPC', start=datetime.strptime(request.form.get("end"), '%Y-%m-%d') - timedelta(days=365), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].reset_index()
        investment = 10000
        values = [investment]
        for i in range(1, len(sp500_prices)):
            investment *= sp500_prices['Adj Close'][i] / sp500_prices['Adj Close'][i - 1]
            values.append(investment)
        sp500_prices['Adj Close'] = values
        sp500_prices = sp500_prices.set_index('Date')
        try:
            df = yf.download(symbols, start=request.form.get("start"), end=request.form.get("end"), auto_adjust = False, prepost = False, threads = True, proxy = None)["Adj Close"].dropna(axis=1, how='all').sort_values('Date')
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
        # Select the columns that end with '.L'
        cols = df.columns[df.columns.str.endswith('.L')]
        # Use numpy.where to get the indices of the elements that need to be converted
        idx = np.where(df[cols].columns.str.endswith('.L'))[0]
        # Use the indices to index into the dataframe and apply the conversion
        df.iloc[:, idx] = df.iloc[:, idx] * GBPtoUSD()
        # Subtract one year from today's date
        today = datetime.strptime(request.form.get("end"), '%Y-%m-%d')
        one_year_ago = today - timedelta(days=365)
        fig = px.line(sp500_prices, x=sp500_prices.index, y='Adj Close')
        #fig = fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'S&P500', title_x = 0.5)
        plot_portfolio_performance_sp500 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        df = df.reset_index()
        # Filter the df dataframe to only include rows with a 'Date' value that is greater than or equal to one_year_ago
        future_prices = df[df['Date'] >= one_year_ago].set_index('Date')
        # Assign the remaining rows in the df dataframe to the future_prices_df dataframe
        prices = df[df['Date'] < one_year_ago].set_index('Date')
        prices = prices.loc[:,prices.iloc[-2,:].notna()]
        for col in prices:
            if pd.isnull(prices[col].iloc[-1]):
                if pd.notnull(prices[col].iloc[-2]):
                    prices[col].iloc[-1] = prices[col].iloc[-2]
                else:
                    prices = prices.drop(col, axis=1)
        selected_values = request.form.getlist('test_model')
        results = {"models": selected_values, "plot_portfolio_performance_sp500": plot_portfolio_performance_sp500}
        for method in selected_values:
            if method == "minimum volatility":
                S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
                ef = EfficientFrontier(None, S)
                try:
                    ef.min_volatility()
                    weights = ef.clean_weights()
                except ValueError as e:
                    error = str(e)
                    flash(error)
                    return redirect("/test")
                try:
                    latest_prices = prices.iloc[-1]
                except IndexError:
                    error = "There is an issue with Yahoo API please try again later"
                    flash(error)
                    return redirect("/test")
                # prices as of the day you are allocating
                if 10000 < float(latest_prices.min()):
                    error = "Amount is not high enough to cover the lowest priced stock"
                    flash(error)
                    return redirect("/test")
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
                except TypeError:
                        delisted = prices.columns[df.iloc[-1].isnull()]
                        delisted= ", ".join(delisted)
                        error = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                        flash(error)
                        return redirect("/test")
                alloc, leftover = da.lp_portfolio()
                nu = pd.Series(alloc)
                fig = px.bar(nu, orientation='h')
                #fig = px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
                fig.update_traces(textposition='inside')
                fig.update_layout(title_text='Suggested Portfolio Allocation for min volatility (long)', title_x=0.5, showlegend = False, yaxis_title="amount", xaxis_title="ticker")
                plot_json_dist_min_vol_long = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                future_prices_min=future_prices.loc[:, future_prices.columns.isin(list(alloc.keys()))]
                #create a total value df where we can see the value of the portfolio on each day
                portfolio_performance = future_prices_min.dot(pd.Series(alloc))+leftover
                # Get the first date in the series
                first_date = portfolio_performance.index[0]
                # Calculate the date that is one day before the first date
                previous_date = first_date - pd.Timedelta(days=1)
                # Create a new series with a single row
                new_row = pd.Series([10000], index=[previous_date])
                # Append the new row to the beginning of the existing series
                portfolio_performance = new_row.append(portfolio_performance)
                fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance)
                #fig = fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'Minimun volatility long', title_x = 0.5)
                plot_portfolio_performance_min_vol = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                max_profit_min_vol=round(portfolio_performance.max(), 2)
                max_profit_pct_min_vol=round((portfolio_performance.max()-10000)/100, 2)
                max_profit_value=portfolio_performance.max()-10000
                profit_date = portfolio_performance.idxmax()
                profit_date_min_vol=profit_date.strftime("%Y-%m-%d")
                num_days_min_vol = (profit_date - portfolio_performance.index[0]).days
                method = "minimum volatility long"
                end_date=request.form.get("end")
                new_test=Test(request.form.get("start"), end_date, list(alloc.keys()), session["user_id"], profit_date, method, max_profit_value, null(), null(), num_days_min_vol)
                db.session.add(new_test)
                db.session.commit()
                results.update({"plot_portfolio_performance_min_vol": plot_portfolio_performance_min_vol,
                                "plot_json_dist_min_vol_long": plot_json_dist_min_vol_long,
                                 "max_profit_pct_min_vol": max_profit_pct_min_vol,
                                 "max_profit_min_vol": max_profit_min_vol,
                                 "num_days_min_vol":num_days_min_vol,
                                 "profit_date_min_vol": profit_date_min_vol,
                                 "listofna": listofna})
                #calulating long/short for min volatility without expected returns
                ef = EfficientFrontier(None, S, weight_bounds=(-1, 1))
                try:
                    ef.min_volatility()
                    weights = ef.clean_weights()
                except ValueError as e:
                    error = str(e)
                    flash(error)
                    return redirect("/test")
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000, short_ratio=float(request.form.get("short_ratio"))/100)
                except TypeError:
                        delisted = prices.columns[df.iloc[-1].isnull()]
                        delisted= ", ".join(delisted)
                        error = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                        flash(error)
                        return redirect("/test")
                alloc, leftover = da.lp_portfolio()
                alloc_to_graph = {k: abs(v*(1-float(request.form.get("short_ratio"))/100)) if v>0 else v for k, v in alloc.items()}
                nu = pd.Series(alloc_to_graph)
                fig = px.bar(nu, orientation='h')
                fig.update_layout(title_text='Suggested Portfolio Allocation for min volatility (long/short)', title_x=0.5, showlegend = False, yaxis_title="amount", xaxis_title="ticker")
                plot_json_dist_min_vol_short = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                leftover=leftover*(1-float(request.form.get("short_ratio"))/100)
                future_prices_min_long_short=future_prices.loc[:, future_prices.columns.isin(list(alloc.keys()))]
                #extract the long positions of the portfolio
                long_positions = {k:v*(1-float(request.form.get("short_ratio"))/100) for k,v in alloc.items() if v>0}
                #extract the short positions of the portfolio
                short_positions = {k:-1*v for k,v in alloc.items() if v<0}
                #changing the negative allocations to positive for later merge and .dot operations
                alloc = {k: abs(v*(1-float(request.form.get("short_ratio"))/100)) if v>0 else abs(v) for k, v in alloc.items()}
                #extract the prices for the stocks with long  positions
                long_positions_portfolio = future_prices.loc[:, future_prices.columns.isin(list(long_positions.keys()))]
                #extract the prices for the stocks with short positions
                short_positions_portfolio = future_prices.loc[:, future_prices.columns.isin(list(short_positions.keys()))]
                #creating a pct_chnage column to reverse the order of the profit/loss (if the stock goes down the profit goes up)
                short_positions_portfolio[short_positions_portfolio.columns+'_pct_change'] = short_positions_portfolio.pct_change()*-1
                #multiplying the price with the pct_change
                for col in short_positions_portfolio.columns:
                    if col.endswith('_pct_change'):
                        continue
                    for i in range(1, len(short_positions_portfolio)):
                        short_positions_portfolio.iloc[i, short_positions_portfolio.columns.get_loc(col)] = short_positions_portfolio.iloc[i-1, short_positions_portfolio.columns.get_loc(col)] + short_positions_portfolio.iloc[i-1, short_positions_portfolio.columns.get_loc(col)] * short_positions_portfolio.iloc[i, short_positions_portfolio.columns.get_loc(col+'_pct_change')]
                #dropping the _pct_change columns
                short_positions_portfolio.drop(columns=[col for col in short_positions_portfolio.columns if col.endswith("_pct_change")], inplace=True)
                #merging the two dfs
                portfolio_performance = pd.merge(short_positions_portfolio, long_positions_portfolio, on='Date', how='inner')
                portfolio_performance = portfolio_performance.dot(pd.Series(alloc))+leftover
                # Get the first date in the series
                first_date = portfolio_performance.index[0]
                # Calculate the date that is one day before the first date
                previous_date = first_date - pd.Timedelta(days=1)
                # Create a new series with a single row
                new_row = pd.Series([10000], index=[previous_date])
                # Append the new row to the beginning of the existing series
                portfolio_performance = new_row.append(portfolio_performance)
                fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance)
                #fig = fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'Minimun volatility long/short', title_x = 0.5)
                plot_portfolio_performance_min_vol_short = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                max_profit_min_vol_short=round(portfolio_performance.max(), 2)
                max_profit_pct_min_vol_short=round((portfolio_performance.max()-10000)/100, 2)
                max_profit_value=portfolio_performance.max()-10000
                profit_date = portfolio_performance.idxmax()
                profit_date_min_vol_short=profit_date.strftime("%Y-%m-%d")
                num_days_min_vol_short = (profit_date - portfolio_performance.index[0]).days
                method = "minimum volatility short"
                end_date=request.form.get("end")
                new_test=Test(request.form.get("start"), end_date, list(alloc.keys()), session["user_id"], profit_date, method, max_profit_value, null(), null(), num_days_min_vol)
                db.session.add(new_test)
                db.session.commit()
                results.update({"plot_portfolio_performance_min_vol_short": plot_portfolio_performance_min_vol_short,
                                "plot_json_dist_min_vol_short": plot_json_dist_min_vol_short,
                                 "max_profit_pct_min_vol_short": max_profit_pct_min_vol_short,
                                 "max_profit_min_vol_short": max_profit_min_vol_short,
                                 "num_days_min_vol_short":num_days_min_vol_short,
                                 "profit_date_min_vol_short": profit_date_min_vol_short})
            if method == "mean variance":
                try:
                    mu = pypfopt.expected_returns.capm_return(prices)
                    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
                    ef = EfficientFrontier(mu, S)
                    ef.add_objective(objective_functions.L2_reg, gamma=(float(request.form.get("gamma"))))  # gamme is the tuning parameter
                    ef.efficient_risk(float(request.form.get("volatility"))/100)
                    weights = ef.clean_weights()
                    #finding zero weights
                    num_small = len([k for k in weights if weights[k] <= 1e-4])
                    num_small_mean_var = str(f"{num_small}/{len(ef.tickers)} tickers have zero weight")
                    port_perf_mean_var = ef.portfolio_performance()
                except Exception as e:
                    error = str(e)
                    flash(error)
                    return redirect("/test")
                #if we want to buy the portfolio mentioned above
                try:
                    latest_prices = prices.iloc[-1]
                except IndexError:
                    error = "There is an issue with Yahoo API please try again later"
                    flash(error)
                    return redirect("/test")
                # prices as of the day you are allocating
                if 10000 < float(latest_prices.min()):
                    error = "Amount is not high enough to cover the lowest priced stock"
                    flash(error)
                    return redirect("/test")
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
                except TypeError:
                    delisted = prices.columns[df.iloc[-1].isnull()]
                    delisted= ", ".join(delisted)
                    error = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                    flash(error)
                    return redirect("/test")
                alloc, leftover = da.lp_portfolio()
                fig= px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
                fig.update_traces(textposition='inside')
                fig.update_layout(width=500, height=500, uniformtext_minsize=12, font=dict(size=10), uniformtext_mode='hide', title_text='Suggested Portfolio Distribution using Capital Asset Pricing Model', title_x=0.5)
                plot_json_dist_mean_var = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                future_prices_mean=future_prices.loc[:, future_prices.columns.isin(list(alloc.keys()))]
                #create a total value df where we can see the value of the portfolio on each day
                portfolio_performance = future_prices_mean.dot(pd.Series(alloc))+leftover
                # Get the first date in the series
                first_date = portfolio_performance.index[0]
                # Calculate the date that is one day before the first date
                previous_date = first_date - pd.Timedelta(days=1)
                # Create a new series with a single row
                new_row = pd.Series([10000], index=[previous_date])
                # Append the new row to the beginning of the existing series
                portfolio_performance = new_row.append(portfolio_performance)
                fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance)
                #fig = fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'mean variance & L2', title_x = 0.5)
                plot_portfolio_performance_mean_var = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                max_profit_mean_var=round(portfolio_performance.max(), 2)
                max_profit_pct_mean_var=round((portfolio_performance.max()-10000)/100, 2)
                max_profit_value=portfolio_performance.max()-10000
                profit_date = portfolio_performance.idxmax()
                profit_date_mean_var=profit_date.strftime("%Y-%m-%d")
                num_days_mean_var = (profit_date - portfolio_performance.index[0]).days
                method = "mean variance"
                end_date=request.form.get("end")
                new_test=Test(request.form.get("start"), end_date, list(alloc.keys()), session["user_id"], profit_date, method, max_profit_value, null(), null(), num_days_mean_var)
                db.session.add(new_test)
                db.session.commit()
                results.update({"num_small_mean_var": num_small_mean_var,
                                "port_perf_mean_var": port_perf_mean_var,
                                 "plot_json_dist_mean_var": plot_json_dist_mean_var,
                                 "plot_portfolio_performance_mean_var": plot_portfolio_performance_mean_var,
                                 "max_profit_mean_var":max_profit_mean_var,
                                 "max_profit_pct_mean_var": max_profit_pct_mean_var,
                                 "profit_date_mean_var": profit_date_mean_var,
                                 "num_days_mean_var": num_days_mean_var})
            if method == "semi-variance":
                try:
                    mu = pypfopt.expected_returns.capm_return(prices)
                    returns = pypfopt.expected_returns.returns_from_prices(prices)
                    returns = returns.dropna()
                    es = EfficientSemivariance(mu, returns)
                    try:
                        es.efficient_return(float(request.form.get("return"))/100)
                    except ValueError as e:
                        error = str(e)
                        flash(error)
                        return redirect("/test")
                    port_perf_semi_var=es.portfolio_performance()
                    weights = es.clean_weights()
                except IndexError as e:
                    flash(e)
                    return redirect("/test")
                try:
                    latest_prices = prices.iloc[-1]
                except IndexError:
                    error = "There is an issue with Yahoo API please try again later"
                    flash(error)
                    return redirect("/test")
                # prices as of the day you are allocating
                if 10000 < float(latest_prices.min()):
                    error = "Amount is not high enough to cover the lowest priced stock"
                    flash(error)
                    return redirect("/test")
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
                except TypeError:
                    delisted = prices.columns[df.iloc[-1].isnull()]
                    delisted= ", ".join(delisted)
                    error = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                    flash(error)
                    return redirect("/test")
                alloc, leftover = da.lp_portfolio()
                fig= px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
                fig.update_traces(textposition='inside')
                fig.update_layout(width=500, height=500, uniformtext_minsize=12, font=dict(size=10), uniformtext_mode='hide', title_text='Suggested Portfolio Distribution using Capital Asset Pricing Model', title_x=0.5)
                plot_json_dist_semi_var = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                future_prices_semi=future_prices.loc[:, future_prices.columns.isin(list(alloc.keys()))]
                #create a total value df where we can see the value of the portfolio on each day
                portfolio_performance = future_prices_semi.dot(pd.Series(alloc))+leftover
                # Get the first date in the series
                first_date = portfolio_performance.index[0]
                # Calculate the date that is one day before the first date
                previous_date = first_date - pd.Timedelta(days=1)
                # Create a new series with a single row
                new_row = pd.Series([10000], index=[previous_date])
                # Append the new row to the beginning of the existing series
                portfolio_performance = new_row.append(portfolio_performance)
                fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance)
                #fig = fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'semi variance', title_x = 0.5)
                plot_portfolio_performance_semi_var = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                max_profit_semi_var=round(portfolio_performance.max(), 2)
                max_profit_pct_semi_var=round((portfolio_performance.max()-10000)/100, 2)
                max_profit_value=portfolio_performance.max()-10000
                profit_date = portfolio_performance.idxmax()
                profit_date_semi_var=profit_date.strftime("%Y-%m-%d")
                num_days_semi_var = (profit_date - portfolio_performance.index[0]).days
                method = "semi variance"
                end_date=request.form.get("end")
                new_test=Test(request.form.get("start"), end_date, list(alloc.keys()), session["user_id"], profit_date, method, max_profit_value, null(), null(), num_days_semi_var)
                db.session.add(new_test)
                db.session.commit()
                results.update({"port_perf_semi_var": port_perf_semi_var,
                                 "plot_json_dist_semi_var": plot_json_dist_semi_var,
                                 "plot_portfolio_performance_semi_var": plot_portfolio_performance_semi_var,
                                 "max_profit_semi_var":max_profit_semi_var,
                                 "max_profit_pct_semi_var": max_profit_pct_semi_var,
                                 "profit_date_semi_var": profit_date_semi_var,
                                 "num_days_semi_var": num_days_semi_var})
            if method == "CVaR":
                mu = pypfopt.expected_returns.capm_return(prices)
                returns =pypfopt.expected_returns.returns_from_prices(prices).dropna()
                cvar_value=request.form.get("cvar")
                ec = EfficientCVaR(mu, returns)
                try:
                    ec.efficient_risk(target_cvar=float(request.form.get("cvar"))/100)
                except:
                    error = f"Please enter CVaR higher than {round(global_dict[int(userId)]['cvar']*(-100), 1)}%"
                    flash(error)
                    return redirect("/test")
                weights = ec.clean_weights()
                try:
                    latest_prices = prices.iloc[-1]
                except IndexError:
                    error = "There is an issue with Yahoo API please try again later"
                    flash(error)
                    return redirect("/test")
                # prices as of the day you are allocating
                if 10000 < float(latest_prices.min()):
                    error = "Amount is not high enough to cover the lowest priced stock"
                    flash(error)
                    return redirect("/test")
                try:
                    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
                except TypeError:
                    delisted = prices.columns[df.iloc[-1].isnull()]
                    delisted= ", ".join(delisted)
                    error = "Can't get latest prices for the following stock/s, please remove to contiue :" + delisted
                    flash(error)
                    return redirect("/test")
                alloc, leftover = da.lp_portfolio()
                fig= px.pie(alloc.keys(), values=alloc.values(), names=alloc.keys())
                fig.update_traces(textposition='inside')
                fig.update_layout(width=500, height=500, uniformtext_minsize=12, font=dict(size=10), uniformtext_mode='hide', title_text='Suggested Portfolio Distribution CVaR', title_x=0.5)
                plot_json_dist_cvar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                future_prices_cvar=future_prices.loc[:, future_prices.columns.isin(list(alloc.keys()))]
                #create a total value df where we can see the value of the portfolio on each day
                portfolio_performance = future_prices_cvar.dot(pd.Series(alloc))+leftover
                # Get the first date in the series
                first_date = portfolio_performance.index[0]
                # Calculate the date that is one day before the first date
                previous_date = first_date - pd.Timedelta(days=1)
                # Create a new series with a single row
                new_row = pd.Series([10000], index=[previous_date])
                # Append the new row to the beginning of the existing series
                portfolio_performance = new_row.append(portfolio_performance)
                fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance)
                #fig = fig.update_xaxes(rangeslider_visible=True)
                fig.update_layout(yaxis_title="Portfolio value",width=1350, height=700, title_text = 'CVaR', title_x = 0.5)
                plot_portfolio_performance_cvar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                target_CVaR_exp_rtn, target_CVaR_cond_val_risk = ec.portfolio_performance()
                max_profit_cvar=round(portfolio_performance.max(), 2)
                max_profit_pct_cvar=round((portfolio_performance.max()-10000)/100, 2)
                max_profit_value=portfolio_performance.max()-10000
                profit_date = portfolio_performance.idxmax()
                profit_date_cvar=profit_date.strftime("%Y-%m-%d")
                num_days_cvar = (profit_date - portfolio_performance.index[0]).days
                method = "CVaR"
                end_date=request.form.get("end")
                new_test=Test(request.form.get("start"), end_date, list(alloc.keys()), session["user_id"], profit_date, method, max_profit_value, null(), null(), num_days_cvar)
                db.session.add(new_test)
                db.session.commit()
                results.update({"plot_json_dist_cvar": plot_json_dist_cvar,
                                "plot_portfolio_performance_cvar": plot_portfolio_performance_cvar,
                                "max_profit_cvar":max_profit_cvar,
                                "max_profit_pct_cvar": max_profit_pct_cvar,
                                "profit_date_cvar": profit_date_cvar,
                                "num_days_cvar": num_days_cvar,
                                "target_CVaR_exp_rtn": target_CVaR_exp_rtn,
                                "target_CVaR_cond_val_risk": target_CVaR_cond_val_risk})
        # Extract the objects with keys that start with "plot_portfolio_performance"
        plot_portfolio_performance_objects = [value for key, value in results.items() if key.startswith("plot_portfolio_performance")]
        # Check if there are 2 or more objects
        if len(plot_portfolio_performance_objects) >= 2:
            fig = subplots.make_subplots(rows=len(plot_portfolio_performance_objects), cols=1)
            for i in range(len(plot_portfolio_performance_objects)):

                # Parse the string into a JSON object
                json_object = json.loads(plot_portfolio_performance_objects[i])
                plot_name=json_object["layout"]["title"]["text"]

                 # Get the data and layout from the px.line object
                data = json_object["data"]
                layout = json_object["layout"]

                # Extract the x and y data from the data object
                x = [point for point in data[0]["x"]]
                y = [point for point in data[0]["y"]]

                trace = go.Scatter(x=x, y=y, name=layout["title"]["text"])
                fig.add_trace(trace, row=1, col=1)
            fig.update_layout(yaxis_title="Portfolio value",width=1350, height=len(plot_portfolio_performance_objects) * 625, title_text = 'Portfolio Performance', title_x = 0.5,
                            shapes=[
                                    {
                                        'type': 'line',
                                        'x0': 0,
                                        'y0': 10000,
                                        'x1': 1,
                                        'y1': 10000,
                                        'xref': 'paper',
                                        'line': {
                                            'color': 'red',
                                            'dash': 'dash',
                                        },
                                    },
                                ])
            merged_graphs=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            results.update({"merged_graphs": merged_graphs})
        return render_template("test_result.html",results=results)
    else:
        arr = np.array(nasdaq_exchange_info)
        nasdaq_exchange_info_tickers = arr[:, 1]
        nasdaq_exchange_info_tickers = nasdaq_exchange_info_tickers.tolist()
        return render_template("test.html", win_loss_signal=win_loss_signal, win_loss_trend=win_loss_trend, top_div=top_div, top_50_crypto=top_50_crypto, top_world_stocks=top_world_stocks, top_US_stocks=top_US_stocks, nasdaq_exchange_info = nasdaq_exchange_info, nasdaq_exchange_info_tickers=nasdaq_exchange_info_tickers)


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
