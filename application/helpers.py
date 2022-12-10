import os
import requests, bmemcached
import yfinance as yf
import pandas as pd
from flask import redirect, session
from functools import wraps
from io import BytesIO
from ftplib import FTP
from apscheduler.schedulers.background import BackgroundScheduler
from flask import current_app as app
from bs4 import BeautifulSoup
from requests import get
from string import *

sched = BackgroundScheduler()

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
  return len(s) > 1

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

app1=app._get_current_object()
def stock_splits_update(*args):
    with app1.app_context():
        from models import db, Records
        from sqlalchemy import func, cast, Date
        stocks=[[s, et, ls] for s, et, ls in db.session.query(Records.symbol, func.to_char(Records.execution_time.cast(Date), 'yyyy-mm-dd'), Records.last_split).distinct().all()]
        for stock, date, ls in stocks:
            splits=yf.Ticker(stock).splits
            last_split_amount=splits.tail(1)[0]
            last_split_date=splits.tail(1).index[0].strftime('%Y-%m-%d')
            if ls is None:
                db.session.query(Records).filter(Records.symbol==stock).update({'last_split':last_split_date})
            elif ls < last_split_date:
                db.session.query(Records).filter(Records.symbol==stock).update({'last_split':last_split_date})
                db.session.query(Records).filter(Records.symbol==stock, func.to_char(Records.execution_time.cast(Date), 'yyyy-mm-dd')<last_split_date).update({'number_of_shares': Records.number_of_shares*last_split_amount, 'purchase_p': Records.purchase_p/last_split_amount}, synchronize_session='fetch')
            db.session.commit()


@sched.scheduled_job('cron',timezone="Europe/London", day_of_week='mon-fri', hour=5, minute=30)
def get_list_win_loss():
    #getting list of top 100 crypto currencies
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    win_loss_symbols = []
    win_loss_symbols_Url = 'https://finviz.com/'
    r= get(win_loss_symbols_Url, headers=headers)
    data=r.text
    soup=BeautifulSoup(data, 'html.parser')
    for listing in soup.find_all('a', attrs={'class':'tab-link'}):
        win_loss_symbols.append(listing.get_text())
    win_loss_signal = win_loss_symbols[:38]
    for listing in soup.find_all('a', attrs={'class':'tab-link-nw'}):
        win_loss_signal.append(listing.get_text())
    win_loss_signal = [word for word in win_loss_signal if all([letter in ascii_uppercase for letter in word])]
    win_loss_signal = sorted(set(win_loss_signal))
    win_loss_trend = win_loss_symbols[38:102]
    win_loss_trend = sorted(set(win_loss_trend))
    mc.set("win_loss_signal", win_loss_signal)
    mc.set("win_loss_trend", win_loss_trend)

@sched.scheduled_job('cron',timezone="Europe/London", day_of_week='mon-fri', hour=5, minute=28)
#gathering top 40 matket cap stocks with dividend higher then 10%
def top_40_mcap_world_higher_then_10pc_div():
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    div_symbols = []
    for i in range(1, 41, 20):
        div_symbolsUrl = 'https://finviz.com/screener.ashx?v=111&f=fa_div_veryhigh&o=-marketcap&r=' + str(i)
        r= get(div_symbolsUrl, headers=headers)
        data=r.text
        soup=BeautifulSoup(data, 'html.parser')
        for listing in soup.find_all('a', attrs={'class':'screener-link-primary'}):
            div_symbols.append(listing.get_text())
    return mc.set("top_div", div_symbols)

@sched.scheduled_job('cron',timezone="Europe/London", day_of_week='mon-fri', hour=5, minute=26)
def get_list_of_crypto_currencies():
    #getting list of top 100 crypto currencies
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    CryptoCurrenciesUrl = 'https://finance.yahoo.com/crypto/?count=50&offset=0'
    r= get(CryptoCurrenciesUrl, headers=headers)
    data=r.text
    soup=BeautifulSoup(data, 'html.parser')
    crypto_symbols = []
    for listing in soup.find_all('a', attrs={'data-test':'quoteLink'}):
        crypto_symbols.append(listing.get_text())
    #removing stable coins
    unwanted = ['USDC-USD', 'BUSD-USD', 'DAI-USD', 'USDP-USD', 'FRAX-USD', 'USDT-USD']
    for ele in sorted(unwanted, reverse = True):
        try:
            crypto_symbols.remove(ele)
        except:
            pass
    return mc.set("top_50_crypto", crypto_symbols)

@sched.scheduled_job('cron',timezone="Europe/London", day_of_week='mon-fri', hour=5, minute=24)
def get_list_of_top_world():
    #getting list of top 100 crypto currencies
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    world_symbols = []
    for i in range(1, 61, 20):
        world_symbolsUrl = 'https://finviz.com/screener.ashx?v=111&o=-marketcap&r=' + str(i)
        r= get(world_symbolsUrl, headers=headers)
        data=r.text
        soup=BeautifulSoup(data, 'html.parser')
        for listing in soup.find_all('a', attrs={'class':'screener-link-primary'}):
            world_symbols.append(listing.get_text())
    return mc.set("top_world", world_symbols)

@sched.scheduled_job('cron',timezone="Europe/London", day_of_week='mon-fri', hour=5, minute=22)
def get_list_of_top_US():
    #getting list of top 100 crypto currencies
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    US_symbols = []
    for i in range(1, 41, 20):
        US_symbolsUrl = 'https://finviz.com/screener.ashx?v=111&f=geo_usa&o=-marketcap&r=' + str(i)
        r= get(US_symbolsUrl, headers=headers)
        data=r.text
        soup=BeautifulSoup(data, 'html.parser')
        for listing in soup.find_all('a', attrs={'class':'screener-link-primary'}):
            US_symbols.append(listing.get_text())
    return mc.set("top_US", US_symbols)

sched.start()
