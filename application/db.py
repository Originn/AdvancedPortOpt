from flask import current_app as app
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)


class Records(db.Model):
    symbol=db.Column(db.Text)
    number_of_shares=db.Column(db.Integer)
    transaction_type=db.Column(db.Text)
    purchase_p=db.Column(db.Float)
    user_id=db.Column(db.Integer, primary_key=True)
    price=db.Column(db.Float)
    execution_time=db.Column(db.DateTime)
    last_split=db.Column(db.Text)

    def __init__(self, user_id, symbol, number_of_shares, transaction_type, purchase_p, price, execution_time, last_split):
        self.user_id = user_id
        self.symbol = symbol
        self.number_of_shares = number_of_shares
        self.transaction_type = transaction_type
        self.purchase_p = purchase_p
        self.price = price
        self.execution_time = execution_time
        self.last_split = last_split

class History(db.Model):
    status = db.Column(db.Text)
    symbol = db.Column(db.String)
    price = db.Column(db.Float)
    number_of_shares = db.Column(db.Integer)
    time = db.Column(db.TIMESTAMP)
    user_id = db.Column(db.Integer, primary_key=True)
    gain_loss=db.Column(db.Float)
    cml_cost=db.Column(db.Float)
    cash_flow=db.Column(db.Float)
    avg_price = db.Column(db.Float)
    cost_unit = db.Column(db.Float)
    cost_transact = db.Column(db.Float)
    cml_units = db.Column(db.Integer)

    def __init__(self, user_id, symbol, price, number_of_shares, time, status, gain_loss, cml_cost, cash_flow, avg_price, cost_unit, cost_transact, cml_units):
        self.user_id = user_id
        self.symbol = symbol
        self.price = price
        self.number_of_shares = number_of_shares
        self.time = time
        self.status = status
        self.gain_loss = gain_loss
        self.cml_cost = cml_cost
        self.cash_flow = cash_flow
        self.avg_price = avg_price
        self.cost_unit = cost_unit
        self.cost_transact = cost_transact
        self.cml_units = cml_units

class Users(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String)
    hash=db.Column(db.Text)
    cash=db.Column(db.Float)

    def __init__(self, username, hash, cash):
        self.username = username
        self.hash = hash
        self.cash = cash

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
    profit_date=db.Column(db.Date, default=None)
    method=db.Column(db.String)
    max_profit_value=db.Column(db.Integer)
    target_profit=db.Column(db.Integer, default=None)
    gamma=db.Column(db.Float)

    def __init__(self, start_date, end_date, symbols, profit_loss, user_id, profit_date, method, max_profit_value, target_profit, gamma):
        self.user_id = user_id
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.profit_loss = profit_loss
        self.profit_date=profit_date
        self.method=method
        self.max_profit_value=max_profit_value
        self.target_profit=target_profit
        self.gamma=gamma

class Stocks(db.Model):
    symbol = db.Column(db.Text, primary_key=True, unique=True)
    shortname = db.Column(db.Text)

    def __init__(self, symbol, shortname):
        self.symbol = symbol
        self.shortname = shortname
