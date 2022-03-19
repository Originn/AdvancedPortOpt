import os
import requests
import urllib.parse
import yfinance as yf
import pandas as pd
from operator import itemgetter
from flask import redirect, render_template, request, session
from functools import wraps


def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

#using yahoo finance as a source for stock data (yahoo have much more tickers and does not have limit on searchs)
#however, it's very slow, so I'll try to implement a faster search using multiple threads.

def lookup(symbol):
    try:
        #response = yf.Ticker(symbol).stats()["price"]
        keys = ['shortName', 'regularMarketPrice', 'symbol']
        filtered_response = {k:yf.Ticker(symbol).stats()["price"].get(k) for k in keys}
        return filtered_response
    except (KeyError, TypeError, ValueError):
        return None


def usd(value):
    """Format value as USD."""
    try:
        return f"${value:,.2f}"
    except TypeError:
        return redirect ("/")

def gbp(value):
    """Format value as USD."""
    try:
        return f"£{value:,.2f}"
    except TypeError:
        return redirect ("/")

class RealTimeCurrencyConverter():
    def __init__(self,url):
        self.data = requests.get(url).json()
        self.currencies = self.data['rates']

    def convert(self, from_currency, to_currency, amount):
        initial_amount = amount
        if from_currency != 'USD' :
            amount = amount / self.currencies[from_currency]

        # limiting the precision to 4 decimal places
        amount = round(amount * self.currencies[to_currency], 2)
        return amount

def GBPtoUSD():
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    converter = RealTimeCurrencyConverter(url)
    GBPvalue=converter.convert('GBP','USD', 1)
    return GBPvalue

def contains_multiple_words(s):
  return len(s.split()) > 1
