import os
import requests, bmemcached
import yfinance as yf
import pandas as pd
from flask import redirect, session
from functools import wraps
from io import BytesIO
from ftplib import FTP
from apscheduler.schedulers.background import BackgroundScheduler

#set memcache in Heroku
servers = os.environ.get('MEMCACHIER_SERVERS', '').split(',')
user = os.environ.get('MEMCACHIER_USERNAME', '')
passw = os.environ.get('MEMCACHIER_PASSWORD', '')

mc = bmemcached.Client(servers, username=user, password=passw)

mc.enable_retry_delay(True)

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
def clean_header(df):
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '').str.replace('(', '').str.replace(')', '').str.replace(' ', '_').str.replace('_/_', '/')

def price_lookup(symbol):
    try:
        filtered_response=round(yf.download(tickers=symbol, period='1m')['Adj Close'][0], 2)
        return filtered_response
    except (KeyError, TypeError, ValueError):
        return None

def lookup(symbol):
    try:
        keys=['shortName', 'regularMarketPrice', 'symbol']
        filtered_response={k:yf.Ticker(symbol).stats()["price"].get(k) for k in keys}
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
    """Format value as GBP."""
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

        # limiting the precision to 2 decimal places
        amount = round(amount * self.currencies[to_currency], 2)
        return amount

def GBPtoUSD():
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    converter = RealTimeCurrencyConverter(url)
    GBPvalue=converter.convert('GBP','USD', 1)
    return GBPvalue

def contains_multiple_words(s):
  return len(s.split()) > 1

#function that build the list of tickers
def symbol_search():

    flo = BytesIO()

    directory = 'symboldirectory'
    filenames = ('otherlisted.txt', 'nasdaqlisted.txt')

    ftp = FTP('ftp.nasdaqtrader.com')
    ftp.login()
    ftp.cwd(directory)

    #Create pandas dataframes from the nasdaqlisted and otherlisted files.
    for item in filenames:
        nasdaq_exchange_info=[]
        ftp.retrbinary('RETR ' + item, flo.write)
        flo.seek(0)
        nasdaq_exchange_info.append(pd.read_fwf(flo))
    ftp.quit()

    # Create pandas dataframes from the nasdaqlisted and otherlisted files.
    nasdaq_exchange_info=pd.concat(nasdaq_exchange_info, axis=1)
    nasdaq_exchange_info[['symbol', 'name', 'Exchange', 'Symbol', 'etf', 'Lot_size', 'Test', 'NASDAQ_Symbol']]=nasdaq_exchange_info['ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol'].str.split('|', expand=True)
    nasdaq_exchange_info=nasdaq_exchange_info.drop(nasdaq_exchange_info.columns[[0]], axis=1).dropna()
    nasdaq_exchange_info=nasdaq_exchange_info[(nasdaq_exchange_info['Test'] != 'Y') & (nasdaq_exchange_info['symbol'] != 'Y') & (~nasdaq_exchange_info.symbol.str.contains('symbol', 'file')) & (~nasdaq_exchange_info.name.str.contains('%', 'arrant'))]
    nasdaq_exchange_info=nasdaq_exchange_info.drop(['Symbol', 'Exchange', 'Lot_size', 'Test', 'NASDAQ_Symbol', 'etf'], axis = 1)
    nasdaq_exchange_info=nasdaq_exchange_info[['name', 'symbol']].values.tolist()
    nasdaq_exchange_info_dict=dict(map(reversed, nasdaq_exchange_info))
    return mc.set("nasdaq_exchange_info", nasdaq_exchange_info), mc.set("nasdaq_exchange_info_dict", nasdaq_exchange_info_dict)


scheduler = BackgroundScheduler(timezone="Europe/London")
# Runs from Monday to Friday at 5:30 (am)
scheduler.add_job(
    func=symbol_search,
    trigger="cron",
    max_instances=1,
    day_of_week='mon-fri',
    hour=5,
    minute=30,
)
scheduler.start()
