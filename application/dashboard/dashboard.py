import dash, math
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
from dash import dash_table
from flask import current_app as app
from flask_sqlalchemy import SQLAlchemy
from flask import session, redirect, flash
from models import db, Records, History, Users
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
import pandas_datareader.data as web
from helpers import clean_header, price_lookup, GBPtoUSD, gbp
import plotly.graph_objects as go
import plotly.express as px
from werkzeug.routing import Map
from sqlalchemy import func
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import yfinance as yf
import numpy as np
from pandas_datareader import data


def init_dashboard(server):
    try:
        user = session.get("user_name")
    except:
        user = 'Random user'
    """Create a Plotly Dash dashboard."""
    try:
        history = db.session.query(History.symbol, History.cml_units, History.price, History.time, History.gain_loss, History.cml_cost, History.cash_flow, History.avg_price).filter_by(user_id=session["user_id"]).all()
    except:
        history=[]
    try:
        stocks = db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares'), func.avg(Records.purchase_p).label('purchase_p')).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()
    except:
        stocks = []
    try:
        availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
    except:
        availableCash = 0
    totalPortValue = 0
    totalprolos = 0

    today = datetime.today().date()
    start_sp = today - BDay(220)
    sp500 = web.get_data_yahoo('^GSPC', start_sp, today)
    new_row = web.get_data_yahoo('^GSPC', period='1m')
    sp500= pd.concat([sp500, new_row])
    print(new_row)
    clean_header(sp500)
    print(sp500)
    sp500_empty = sp500[['adj_close', 'open']].reset_index()
    sp500_empty = sp500_empty.drop_duplicates(subset='Date', keep='first')
    sp500_empty['sp500_diff'] = (sp500_empty['adj_close'].diff()).round(2)
    sp500_empty['daily_return'] = ((sp500_empty['adj_close']/sp500_empty['adj_close'].shift(1)) - 1).round(4)*100

    kpi_sp500_1d_pct = sp500_empty.tail(1).daily_return.iloc[0]
    #open of first day in the timeframe / adj_close of last day
    kpi_sp500_7d_pct = (1 - (sp500_empty.tail(7).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
    kpi_sp500_15d_pct = (1 - (sp500_empty.tail(15).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
    kpi_sp500_30d_pct = (1 - (sp500_empty.tail(30).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
    kpi_sp500_200d_pct = (1 - (sp500_empty.tail(200).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100

    CHART_THEME = 'plotly_white'

    indicators_sp500 = go.Figure()
    indicators_sp500.layout.template = CHART_THEME

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = kpi_sp500_1d_pct,
        number = {'suffix': " %"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>1 Day</span>"},
        domain = {'row': 0, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = kpi_sp500_7d_pct,
        number = {'suffix': " %"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
        domain = {'row': 1, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = kpi_sp500_15d_pct,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
        domain = {'row': 2, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = kpi_sp500_30d_pct,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
        domain = {'row': 3, 'column': 0}))

    indicators_sp500.add_trace(go.Indicator(
        mode = "number+delta",
        value = kpi_sp500_200d_pct,
        number = {'suffix': " %"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
        domain = {'row': 4, 'column': 1}))

    indicators_sp500.update_layout(
        grid = {'rows': 5, 'columns': 1, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30)
    )

    indicators_ptf = go.Figure()
    indicators_ptf.layout.template = CHART_THEME
    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = availableCash,
        number = {'prefix': " $"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>Cash</span>"},
        domain = {'row': 0, 'column': 0}))

    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = totalPortValue,
        number = {'prefix': "$"},
        title = {"text": "<br><span style='font-size:0.7em;color:gray'>Total Value</span>"},
        domain = {'row': 1, 'column': 0}))

    indicators_ptf.add_trace(go.Indicator(
        mode = "number+delta",
        value = totalprolos,
        number = {'prefix': "$"},
        title = {"text": "<span style='font-size:0.7em;color:gray'>Total Profit/Loss</span>"},
        domain = {'row': 2, 'column': 0}))

    indicators_ptf.update_layout(
        height=550,
        grid = {'rows': 5, 'columns': 1, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30)
    )

    chart_ptfvalue = go.Figure()  # generating a figure that will be updated in the following lines
    chart_ptfvalue.add_trace(go.Scatter(x=sp500_empty.Date, y=sp500_empty.adj_close,
                        mode='lines',  # you can also use "lines+markers", or just "markers"
                        name='S&P 500 Value'))
    chart_ptfvalue.layout.template = CHART_THEME
    chart_ptfvalue.layout.height=500
    chart_ptfvalue.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # this will help you optimize the chart space
    chart_ptfvalue.update_layout(
        title='S&P 500 Value (USD $)',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Value: $ USD',
            titlefont_size=14,
            tickfont_size=12,
            ))

    fig_growth2 = go.Figure()
    fig_growth2.layout.template = CHART_THEME

    fig_growth2.add_trace(go.Bar(
        x=sp500_empty.Date,
        y=sp500_empty.daily_return,
        name='S&P 500',
    ))
    fig_growth2.update_layout(barmode='group')
    fig_growth2.layout.height=300
    fig_growth2.update_layout(margin = dict(t=50, b=50, l=25, r=25))
    fig_growth2.update_layout(
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='% change',
            titlefont_size=13,
            tickfont_size=12,
            ))

    fig_growth2.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99))

    donut_top = go.Figure()
    donut_top.layout.template = CHART_THEME
    donut_top.add_trace(go.Pie(labels=['No Stocks'], values=[0]))
    donut_top.update_traces(hole=.7, hoverinfo="label+value+percent")
    donut_top.update_traces(textposition='outside', textinfo='label+value')
    donut_top.update_layout(showlegend=False)
    donut_top.update_layout(margin = dict(t=50, b=50, l=25, r=25))

    dash_app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashapp/', external_stylesheets=[dbc.themes.FLATLY])

    dash_app.layout = html.Div([dcc.Location(id='page-content'),
    dbc.Container(
    [
        dcc.Loading([
            dbc.Row(dbc.Col(html.H2('PORTFOLIO OVERVIEW', className='text-center text-primary, mb-3'))),  # header row
            dbc.Row(dbc.Col(html.H4(f'Hello {user}', className='text-center text-primary, mb-3', id='greet'))),
            dbc.Row([  # start of second row
                dbc.Col([  # first column on second row
                html.H5('Total Portfolio Value ($USD)', className='text-center'),
                dcc.Graph(id='chrt-portfolio-main',
                        figure=chart_ptfvalue,
                        style={'height':550}),
                html.Hr(),
                ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
                dbc.Col([  # second column on second row
                html.H5('Portfolio Return', className='text-center'),
                html.Div(dcc.Graph(id='indicators-ptf',
                        figure=indicators_ptf,
                        style={"maxHeight": "550px", 'overflowY': 'scroll'})),
                html.Hr()
                ], width={'size': 2, 'offset': 0, 'order': 2}),  # width second column on second row
                dbc.Col([  # third column on second row
                html.H5('S&P500', className='text-center'),
                dcc.Graph(id='indicators-sp',
                        figure=indicators_sp500,
                        style={'height':550}),
                html.Hr()
                ], width={'size': 2, 'offset': 0, 'order': 3}),  # width third column on second row
            ]),  # end of second row

            dbc.Row([  # start of third row
                dbc.Col([  # first column on third row
                    html.H5('Daily Return (%)', className='text-center'),
                    dcc.Graph(id='chrt-portfolio-secondary',
                        figure=fig_growth2,
                        style={'height':380}),
                ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
                dbc.Col([  # second column on third row
                    html.H5('Top 15 Holdings', className='text-center'),
                    dcc.Graph(id='pie-top15',
                        figure = donut_top,
                        style={'height':380}),
                ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on second row
            ])])  # end of third row

    ], fluid=True)])

    @dash_app.callback(
                      Output('greet', 'children'),
                      Output('chrt-portfolio-main', 'figure'),
                      Output('indicators-ptf', 'figure'),
                      Output('chrt-portfolio-secondary', 'figure'),
                      Output('pie-top15', 'figure'),
                      [Input('greet', 'pathname')])
    def load_user_dash(input1):
        if session.get('user_name'):
            user = session.get('user_name')
        try:
            stocks = db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares'), func.avg(Records.purchase_p).label('purchase_p')).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()
        except:
            stocks = []
        if stocks == []:
            try:
                availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
            except:
                availableCash = 0
            totalPortValue = 0
            totalprolos = 0

            today = datetime.today().date()
            start_sp = today - BDay(220)

            kpi_sp500_1d_pct = sp500_empty.tail(1).daily_return.iloc[0]
            #open of first day in the timeframe / adj_close of last day
            kpi_sp500_7d_pct = (1 - (sp500_empty.tail(7).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
            kpi_sp500_15d_pct = (1 - (sp500_empty.tail(15).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
            kpi_sp500_30d_pct = (1 - (sp500_empty.tail(30).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100
            kpi_sp500_200d_pct = (1 - (sp500_empty.tail(200).open.iloc[0]/sp500_empty.adj_close.iloc[-1])).round(4)*100

            CHART_THEME = 'plotly_white'

            indicators_sp500 = go.Figure()
            indicators_sp500.layout.template = CHART_THEME

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_1d_pct,
                number = {'suffix': " %"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>1 Day</span>"},
                domain = {'row': 0, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_7d_pct,
                number = {'suffix': " %"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
                domain = {'row': 1, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_15d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
                domain = {'row': 2, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_30d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
                domain = {'row': 3, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_200d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
                domain = {'row': 4, 'column': 1}))

            indicators_sp500.update_layout(
                grid = {'rows': 5, 'columns': 1, 'pattern': "independent"},
                margin=dict(l=50, r=50, t=30, b=30)
            )

            indicators_ptf = go.Figure()
            indicators_ptf.layout.template = CHART_THEME
            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = availableCash,
                number = {'prefix': " $"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>Cash</span>"},
                domain = {'row': 0, 'column': 0}))

            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = totalPortValue,
                number = {'prefix': "$"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>Total Value</span>"},
                domain = {'row': 1, 'column': 0}))

            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = totalprolos,
                number = {'prefix': "$"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>Total Profit/Loss</span>"},
                domain = {'row': 2, 'column': 0}))

            indicators_ptf.update_layout(
                height=550,
                grid = {'rows': 5, 'columns': 1, 'pattern': "independent"},
                margin=dict(l=50, r=50, t=30, b=30)
            )

            chart_ptfvalue = go.Figure()  # generating a figure that will be updated in the following lines
            chart_ptfvalue.add_trace(go.Scatter(x=sp500_empty.Date, y=sp500_empty.adj_close,
                                mode='lines',  # you can also use "lines+markers", or just "markers"
                                name='S&P 500 Value'))
            chart_ptfvalue.layout.template = CHART_THEME
            chart_ptfvalue.layout.height=500
            chart_ptfvalue.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # this will help you optimize the chart space
            chart_ptfvalue.update_layout(
                title='S&P 500 Value (USD $)',
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='Value: $ USD',
                    titlefont_size=14,
                    tickfont_size=12,
                    ))

            fig_growth2 = go.Figure()
            fig_growth2.layout.template = CHART_THEME

            fig_growth2.add_trace(go.Bar(
                x=sp500_empty.Date,
                y=sp500_empty.daily_return,
                name='S&P 500',
            ))
            fig_growth2.update_layout(barmode='group')
            fig_growth2.layout.height=300
            fig_growth2.update_layout(margin = dict(t=50, b=50, l=25, r=25))
            fig_growth2.update_layout(
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='% change',
                    titlefont_size=13,
                    tickfont_size=12,
                    ))

            fig_growth2.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99))

            donut_top = go.Figure()
            donut_top.layout.template = CHART_THEME
            donut_top.add_trace(go.Pie(labels=['No Stocks'], values=[0]))
            donut_top.update_traces(hole=.7, hoverinfo="label+value+percent")
            donut_top.update_traces(textposition='outside', textinfo='label+value')
            donut_top.update_layout(showlegend=False)
            donut_top.update_layout(margin = dict(t=50, b=50, l=25, r=25))
        else:
            history = db.session.query(History.symbol, History.cml_units, History.number_of_shares, History.price, History.time, History.gain_loss, History.cml_cost, History.cash_flow, History.avg_price).filter_by(user_id=session["user_id"]).all()
            records = db.session.query(Records.symbol, Records.number_of_shares).filter_by(user_id=session["user_id"]).all()
            all_transactions=pd.DataFrame(history)
            all_records = pd.DataFrame(records)
            all_transactions=all_transactions.loc[all_transactions['symbol'].isin(all_records['symbol'])]
            all_transactions = all_transactions.sort_values('time')

            all_transactions_out = pd.DataFrame()
            for sym, sub_df in all_transactions.groupby('symbol'):
                zero_dates = sub_df[(sub_df['cml_units'] == 0)]['time']
                if not zero_dates.empty:
                    last_zero_date = zero_dates.values[-1]
                else:
                    last_zero_date = pd.to_datetime(0)
                all_transactions_out = pd.concat([all_transactions_out, sub_df[sub_df['time'] > last_zero_date]])

            all_transactions = all_transactions_out.copy()
            all_tickers = set(list(stock for stock in all_transactions['symbol']))
            today = datetime.today().date()
            end_stocks = today+pd.DateOffset(1)
            start_stocks = all_transactions.time.min().date()

            from multiprocessing import Process, Queue
            def get(tickers, all_transactions, enddate):
                def data(ticker):
                    def premarket1(ticker):
                        try:
                            pmarket = float(web.get_quote_yahoo(ticker)['preMarketPrice'])
                            return pmarket
                        except:
                            return None
                    df = web.get_data_yahoo(ticker, start=all_transactions[all_transactions['symbol']==ticker]['time'].min().date()-BDay(0), end=enddate)
                    if str(today) not in str(df.reset_index().Date):
                        new_data = pd.DataFrame(df[-1:].values, index=[today], columns=df.columns)
                        df = pd.concat([df, new_data])
                    if ticker.endswith(".L"):
                        df.loc[:,'Adj Close'] = df.loc[:,'Adj Close']*GBPtoUSD()
                    premarket = premarket1(ticker)
                    if premarket:
                        df.iloc[-1, 4] = premarket
                        return df
                    else:
                        return df
                datas = map(data, tickers)
                return(pd.concat(datas, keys=tickers, names=['ticker', 'date']))

            all_data = get(all_tickers, all_transactions, end_stocks)
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            MEGA_DICT = {}
            min_date = min(stock['time'] for stock in history).strftime('%Y-%m-%d')
            min_date=datetime.strptime(min_date, '%Y-%m-%d')
            min_inds = all_transactions.groupby("symbol", sort=False).time.idxmin()
            result = all_transactions.copy()
            result=result.rename(columns={'time' : 'date', 'symbol': 'ticker'})
            result.drop('price', inplace= True, axis=1)
            result['date']=result['date'].dt.date.astype('datetime64')
            TX_COLUMNS = ['date', 'ticker', 'cash_flow', 'cml_units', 'cml_cost', 'gain_loss']
            tx_filt = result[TX_COLUMNS]

            for ticker in all_tickers:
                prices_df = all_data[all_data.index.get_level_values('ticker').isin([ticker])].reset_index()
                PX_COLS = ['date', 'Adj Close']
                prices_df = prices_df[prices_df.date >= min_date-BDay(1)][PX_COLS].set_index(['date'])
                # Making sure we get sameday transactions
                tx_df = tx_filt[tx_filt.ticker==ticker].groupby('date').agg({'cash_flow': 'sum',
                                                                    'cml_units': 'last',
                                                                    'cml_cost': 'last',
                                                                    'gain_loss': 'sum'})
                # Merging price history and transactions dataframe

                tx_and_prices = pd.merge(prices_df, tx_df, how='outer', left_index=True, right_index=True).fillna(0)

                # This is to fill the days that were not in our transaction dataframe
                tx_and_prices['cml_units'] = tx_and_prices['cml_units'].replace(to_replace=0, method='ffill')
                tx_and_prices['cml_cost'] = tx_and_prices['cml_cost'].replace(to_replace=0, method='ffill')
                tx_and_prices['gain_loss'] = tx_and_prices['gain_loss'].replace(to_replace=0, method='ffill')
                # Cumulative sum for the cashflow
                tx_and_prices['cashflow'] = tx_and_prices['cash_flow'].cumsum()
                tx_and_prices['avg_price'] = (tx_and_prices['cml_cost']/tx_and_prices['cml_units'])
                tx_and_prices['mktvalue'] = (tx_and_prices['cml_units']*tx_and_prices['Adj Close'])
                if tx_and_prices['Adj Close'].iloc[-1]==0:
                    tx_and_prices['mktvalue'].iloc[-1] = tx_and_prices['cml_cost'].iloc[-1]
                tx_and_prices = tx_and_prices.add_prefix(ticker+'_')
                # Once we're happy with the dataframe, add it to the dictionary
                MEGA_DICT[ticker] = tx_and_prices.round(3)

            #portfilio df
            MEGA_DF = pd.concat(MEGA_DICT.values(), axis=1).reset_index()

            MEGA_DF['date'] = pd.to_datetime(MEGA_DF['date'])
            MEGA_DF.set_index('date', inplace=True)
            portf_allvalues = MEGA_DF.filter(regex='mktvalue').fillna(0)

            portf_allvalues['portf_value'] = portf_allvalues.sum(axis=1) # summing all market values

            # # For the S&P500 price return
            # start_sp = today - BDay(200)
            # #
            # if today not in sp500.index:
            #     new_data = pd.DataFrame(sp500[-1:].values, index=[today], columns=sp500.columns)
            #     sp500 = pd.concat([sp500, new_data])
            # clean_header(sp500)
            # sp500=sp500[~sp500.index.duplicated(keep='first')]

            #getting the pct change
            portf_allvalues = portf_allvalues.join(sp500[['adj_close', 'open']], how='outer')
            #portf_allvalues = portf_allvalues.drop_duplicates()
            portf_allvalues= portf_allvalues.rename(columns={'adj_close': 'sp500_mktvalue'})
            portf_allvalues['ptf_value_pctch'] = (portf_allvalues['portf_value'].pct_change()*100).round(2)
            portf_allvalues['sp500_pctch'] = (portf_allvalues['sp500_mktvalue'].pct_change()*100).round(2)
            portf_allvalues['ptf_value_diff'] = (portf_allvalues['portf_value'].diff()).round(2)
            portf_allvalues['sp500_diff'] = (portf_allvalues['sp500_mktvalue'].diff()).round(2)
            portf_allvalues['sp500_mktvalue'].fillna(method='ffill', inplace=True)

            # KPI's for S&P500

            kpi_sp500_1d_pct = ((portf_allvalues.sp500_mktvalue.iloc[-1] - portf_allvalues.tail(1).open.iloc[0])/portf_allvalues.tail(1).open.iloc[0]).round(4)*100
            kpi_sp500_7d_pct = ((portf_allvalues.sp500_mktvalue.iloc[-1] - portf_allvalues.tail(5).open.iloc[0])/portf_allvalues.tail(5).open.iloc[0]).round(4)*100
            kpi_sp500_15d_pct = ((portf_allvalues.sp500_mktvalue.iloc[-1] - portf_allvalues.tail(15).open.iloc[0])/portf_allvalues.tail(15).open.iloc[0]).round(4)*100
            kpi_sp500_30d_pct = ((portf_allvalues.sp500_mktvalue.iloc[-1] - portf_allvalues.tail(30).open.iloc[0])/portf_allvalues.tail(30).open.iloc[0]).round(4)*100
            kpi_sp500_200d_pct = ((portf_allvalues.sp500_mktvalue.iloc[-1] - portf_allvalues.tail(200).open.iloc[0])/portf_allvalues.tail(200).open.iloc[0]).round(4)*100

            initial_date = min_date  # do not use anything earlier than your first trade
            plotlydf_portfval = portf_allvalues[portf_allvalues.index >= initial_date]
            plotlydf_portfval = plotlydf_portfval[['portf_value', 'sp500_mktvalue', 'ptf_value_pctch', 'sp500_pctch', 'ptf_value_diff', 'sp500_diff']].reset_index().round(2)
            # calculating cumulative growth since initial date
            plotlydf_portfval['ptf_growth'] = (plotlydf_portfval.portf_value.pct_change()*100).round(2)
            plotlydf_portfval['sp500_growth'] = (plotlydf_portfval.sp500_mktvalue.pct_change()*100).round(2)

            plotlydf_portfval.rename(columns={'index': 'date'}, inplace=True)  # needed for later

            all_transactions_mod = all_transactions[['time','cash_flow']].copy()
            all_transactions_mod['time'] = all_transactions_mod.time.dt.strftime('%Y-%m-%d')
            all_transactions_mod.rename(columns={'time': 'date'}, inplace=True)
            all_transactions_mod['date']= pd.to_datetime(all_transactions_mod['date'], format='%Y-%m-%d')
            all_transactions_mod = all_transactions_mod.groupby('date').agg({'cash_flow': 'sum'})
            plotlydf_portfval = pd.merge(plotlydf_portfval, all_transactions_mod, on='date', how='left')

            plotlydf_portfval['cash_flow'] = plotlydf_portfval['cash_flow'].fillna(0).cumsum()
            plotlydf_portfval['ptf_growth_wo_purchases'] = plotlydf_portfval['portf_value'] + plotlydf_portfval['cash_flow']
            plotlydf_portfval['ptf_value_pctch_wo_purchases'] = (((plotlydf_portfval['ptf_growth_wo_purchases']/plotlydf_portfval['portf_value'])*100).diff()).round(2)


            if math.isnan(plotlydf_portfval.iloc[-1]['sp500_growth']) == True:
                plotlydf_portfval = plotlydf_portfval[:-1]


            stocks = db.session.query(Records.symbol, func.sum(Records.number_of_shares).label('sumshares'), func.avg(Records.purchase_p).label('purchase_p')).filter_by(user_id=session["user_id"]).group_by(Records.symbol).all()

            if not stocks:
                stocks = []
                availableCash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
                grandTotal = availableCash
                totalPortValue = 0
                totalprolos = 0
                portfolio_pct = 0

            else:
                stocks = [r._asdict() for r in stocks]
                cash = db.session.query(Users.cash).filter_by(id=session["user_id"]).first().cash
                totalPortValue = 0
                totalprolos = 0
                portfolio_pct = 0


                #building the index
                for stock in stocks:
                    price = price_lookup(stock['symbol'])
                    #check if the stock is listed in the UK
                    if ".L" in stock["symbol"]:
                        #if it is - convert the price from GBP to USD
                        price=GBPtoUSD()*price
                    stock['ap'] = (stock['sumshares'] * price)/stock['sumshares']
                    stock['total'] = stock['sumshares'] * price
                    stock['perc_change'] = round(((stock['ap'] - stock['purchase_p'])/stock['purchase_p'])*100, 3)
                    stock['prolos'] = round((stock['perc_change']/100)*stock['total'], 2)
                    totalprolos += stock['prolos']
                    totalPortValue += stock['sumshares'] * price


                availableCash = cash
                grandTotal = availableCash + totalPortValue


                #building the portfolio pct change
                for stock in stocks:
                    portfolio_pct += ((stock['total']/totalPortValue)*stock['perc_change'])

                portfolio_pct = round(portfolio_pct, 2)

            CHART_THEME = 'plotly_white'  # others include seaborn, ggplot2, plotly_dark

            chart_ptfvalue = go.Figure()  # generating a figure that will be updated in the following lines
            chart_ptfvalue.add_trace(go.Scatter(x=plotlydf_portfval.date, y=plotlydf_portfval.portf_value,
                                mode='lines',  # you can also use "lines+markers", or just "markers"
                                name='Portfolio Value'))
            chart_ptfvalue.add_trace(go.Scatter(x=portf_allvalues.query('index >= @min_date').index, y=portf_allvalues.query('index >= @min_date').sp500_mktvalue,
                                mode='lines',  # you can also use "lines+markers", or just "markers"
                                name='S&P Value'))
            chart_ptfvalue.layout.template = CHART_THEME
            chart_ptfvalue.layout.height=500
            chart_ptfvalue.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # this will help you optimize the chart space
            chart_ptfvalue.update_layout(
                title='Global Portfolio Value (USD $)',
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='Value: $ USD',
                    titlefont_size=14,
                    tickfont_size=12,
                    ))
            chart_ptfvalue.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.99,
                xanchor="right",
                x=1
            ))

            fig2 = go.Figure(data=[
                go.Bar(name='Portfolio', x=plotlydf_portfval['date'], y=plotlydf_portfval['ptf_value_pctch_wo_purchases']),
                go.Bar(name='SP500', x=plotlydf_portfval['date'], y=plotlydf_portfval['sp500_pctch'])
            ])
            # Change the bar mode
            fig2.update_layout(barmode='group')
            fig2.layout.template = CHART_THEME
            fig2.layout.height=300
            fig2.update_layout(margin = dict(t=50, b=50, l=25, r=25))
            fig2.update_layout(
                title='% variation - Portfolio vs SP500',
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='% change',
                    titlefont_size=14,
                    tickfont_size=12,
                    ))
            fig2.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99))

            df = plotlydf_portfval[['date', 'ptf_growth', 'sp500_growth']].copy().round(3)
            df['month'] = df.date.dt.month_name()  # date column should be formatted as datetime
            df['weekday'] = df.date.dt.day_name()  # could be interesting to analyze weekday returns later
            df['year'] = df.date.dt.year
            df['weeknumber'] = df.date.dt.isocalendar().week    # could be interesting to try instead of timeperiod
            df['timeperiod'] = pd.to_datetime(df['date'], format='%d-%b')


            # getting the percentage change for each period. the first period will be NaN
            sp = (df.reset_index().groupby('timeperiod').last()['sp500_growth'].pct_change()*100).round(2)
            ptf = df.reset_index().groupby('timeperiod').last()['ptf_growth'].pct_change()*100
            plotlydf_growth_compare = pd.merge(ptf, sp, on='timeperiod').reset_index().round(3)
            pd.set_option('display.max_columns', None)



            fig_growth2 = go.Figure()
            fig_growth2.layout.template = CHART_THEME
            fig_growth2.add_trace(go.Bar(
                x=plotlydf_growth_compare.timeperiod,
                y=plotlydf_portfval.ptf_value_pctch_wo_purchases,
                name='Portfolio'
            ))
            fig_growth2.add_trace(go.Bar(
                x=plotlydf_growth_compare.timeperiod,
                y=plotlydf_portfval.sp500_growth,
                name='S&P 500',
            ))
            fig_growth2.update_layout(barmode='group')
            fig_growth2.layout.height=300
            fig_growth2.update_layout(margin = dict(t=50, b=50, l=25, r=50))
            fig_growth2.update_layout(
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='% change',
                    titlefont_size=13,
                    tickfont_size=12,
                    ))

            fig_growth2.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.99,
                xanchor="right",
                x=1
            ))

            indicators_ptf = go.Figure()
            indicators_ptf.layout.template = CHART_THEME
            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = totalprolos,
                number = {'prefix': " $"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>Profit/Loss</span>"},
                delta = {'position': "bottom", 'reference': totalprolos - portfolio_pct/100, 'relative': False, 'valueformat': ".2%"},
                domain = {'row': 0, 'column': 0}))

            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = availableCash,
                number = {'prefix': "$"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>Available Cash</span>"},
                domain = {'row': 1, 'column': 0}))

            indicators_ptf.add_trace(go.Indicator(
                mode = "number+delta",
                value = grandTotal,
                number = {'prefix': "$"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>Total Value</span>"},
                delta = {'position': "bottom", 'reference': 10000, 'relative': False},
                domain = {'row': 2, 'column': 0}))

            row_n=3
            for stock in stocks:
                indicators_ptf.add_trace(go.Indicator(
                    mode = "number+delta",
                    value = stock['prolos'],
                    number = {'prefix': "$"},
                    title = {"text": stock['symbol']},
                    delta = {'position': "bottom", 'reference': stock['prolos'] - stock['perc_change']/100, 'relative': False, 'valueformat': ".2%"},
                    domain = {'row': row_n, 'column': 0}))
                row_n += 1

            indicators_ptf.update_layout(
                height=len(all_tickers)*110 + 330,
                grid = {'rows': row_n, 'columns': 1, 'pattern': "independent"},
                margin=dict(l=50, r=50, t=30, b=30)
            )


            #indicators_ptf.update_layout(autosize=True,height=600,grid={'rows': 6, 'columns': 1, 'pattern': "independent"},margin=dict(l=50, r=50, t=30, b=30))

            indicators_sp500 = go.Figure()
            indicators_sp500.layout.template = CHART_THEME

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_1d_pct,
                number = {'suffix': " %"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>1 Day</span>"},
                domain = {'row': 0, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_7d_pct,
                number = {'suffix': " %"},
                title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
                domain = {'row': 1, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_15d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
                domain = {'row': 2, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_30d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
                domain = {'row': 3, 'column': 0}))

            indicators_sp500.add_trace(go.Indicator(
                mode = "number+delta",
                value = kpi_sp500_200d_pct,
                number = {'suffix': " %"},
                title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
                domain = {'row': 4, 'column': 1}))

            indicators_sp500.update_layout(
                grid = {'rows': 5, 'columns': 1, 'pattern': "independent"},
                margin=dict(l=50, r=50, t=30, b=30)
            )


            #prices for top stocks
            last_positions = all_transactions.groupby(['symbol']).agg({'cml_units': 'last', 'cml_cost': 'last', 'gain_loss': 'sum', 'cash_flow': 'sum'}).reset_index()

            curr_prices = []
            for tick in last_positions['symbol']:

                price = price_lookup(tick)
                curr_prices.append(price)

            last_positions['price'] = curr_prices
            last_positions['current_value'] = (last_positions.price * last_positions.cml_units).round(2)
            last_positions['avg_price'] = (last_positions.cml_cost / last_positions.cml_units).round(2)
            last_positions = last_positions.sort_values(by='current_value', ascending=False)

            donut_top = go.Figure()
            donut_top.layout.template = CHART_THEME
            donut_top.add_trace(go.Pie(labels=last_positions.head(15).symbol, values=last_positions.head(15).current_value))
            donut_top.update_traces(hole=.7, hoverinfo="label+value+percent")
            donut_top.update_traces(textposition='outside', textinfo='label+value')
            donut_top.update_layout(showlegend=False)
            donut_top.update_layout(margin = dict(t=50, b=50, l=25, r=25))

        return dbc.Col(html.H4(f'Welcome back, {user}', className='text-center text-primary, mb-3', id='greet')), chart_ptfvalue, indicators_ptf, fig_growth2, donut_top


    return dash_app.server
