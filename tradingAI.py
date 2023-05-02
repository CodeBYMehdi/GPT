'''

                                                    Made by @mehdibj

'''



# Finance modules

import yfinance as yf

# Import the IB modules

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.order import *
from ib_insync import *


# Import classic modules

import openai
import requests
import exchange
import json
import time
import datetime
import logging
import uuid
import numpy as np
import ta
import pandas as pd
import threading
import pandas_datareader.data as web
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import the machine learning

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import Deep Learning modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

# Import the dataset module

from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries

# Import the technical indicators module

from ta.trend import MACD
from ta.volatility import BollingerBands

# Initialize openai API key

openai.api_key = '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'

# Initialize Alpha Vantage API key

alphavantage_api_key = 'QUJ00N0C3VLU7NKC'



class Market:

    def __init__(self, symbol, yahoo_ticker, currency='EURUSD', hist_window=365):
        self.symbol = symbol
        self.yahoo_ticker = yahoo_ticker
        self.currency = currency
        self.hist_window = hist_window
        

    def fx_price(self, real_time=True):
        if real_time:
            fx = ForeignExchange(key=alphavantage_api_key, output_format='pandas')
            data, meta_data = fx.get_currency_exchange_rate(from_currency='EUR', to_currency='USD')
            price = data['5. Exchange Rate'][0]
        else:
            fx = ForeignExchange(key=alphavantage_api_key, output_format='pandas')
            data, meta_data = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='compact')
            price = data['4. close'][-1]
        return float(price)

    
    def stock_price(self):
        msft = yf.Ticker('MSFT')
        hist_data = msft.history(period='max')
        hist_data = hist_data[['Close']]
        print(hist_data)

        # Get real time data for MSFT stock
        msft_info = msft.info
        time.sleep(5)
        current_price = msft_info["regularMarketOpen"]
        print(current_price)

        return hist_data, current_price
        



class Balance:


    def calculate_units(self, portfolio_value, price):
        # calculate the maximum number of units that can be bought with the portfolio value and the price
        balance = self.get_balance()
        max_units = balance / price
        return int(min(max_units, portfolio_value / price))



    def get_balance(self):
        # implement a function to get the account balance from the IB API
        # here is an example implementation that assumes that the account currency is EUR
        class MyEWrapper(EWrapper):
            def __init__(self):
                self.account_balance = None

            def accountSummary(self, reqId, account, tag, value, currency):
                if tag == "TotalCashValue":
                    self.account_balance = value
                    
                    









class NNTS:
    def __init__(self, lookback, units, dropout, epochs, batch_size):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def _prepare_data(self, data):
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def _build_model(self, X):
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model

    def generate_signals(self, data, strategy):
        X, y = self._prepare_data(data)
        model = self._build_model(X)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        y_pred = model.predict(X)
        signals = np.zeros(len(data))
        signals[self.lookback:] = np.where(y_pred > y, 1, -1)
        
        if strategy == 'buy':
            signals[signals != 1] = 0
        elif strategy == 'sell':
            signals[signals != -1] = 0
        else:
            signals = np.zeros(len(data))
        
        return signals










class DataProcessor:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        # Drop any rows with NaN values
        data.dropna(inplace=True)

        # Add technical indicators
        data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume")

        # Create X and y
        X = data[self.feature_columns].values
        y = np.where(data["close"].shift(-1) > data["close"], 1, -1)
        y = y[:-1]

        # Scale the data
        X = self.scaler.fit_transform(X)

        return X, y





        
class RiskManager:
    def __init__(self, max_risk_per_trade=0.5, max_open_positions=500):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_positions = max_open_positions
        self.open_positions = []
        self.current_equity = 0
        self.total_risk = 0
    
    def update_equity(self, equity):
        self.current_equity = equity
        self.equity.append(self.current_equity)
    
    def calculate_risk(self, position_size, stop_loss):
        return position_size * stop_loss
    
    def can_open_position(self):
        return len(self.open_positions) < self.max_open_positions
    
    def can_afford_position(self, position_size):
        return self.current_equity * self.max_risk_per_trade >= self.calculate_risk(position_size)
    
    def open_position(self, position, take_profit=None):
        if not self.can_open_position():
            raise Exception("Cannot open position: max open positions reached.")
        if not self.can_afford_position(position['size']):
            raise Exception("Cannot open position: insufficient funds or excessive risk.")
        
        position['take_profit'] = take_profit
        self.open_positions.append(position)
        self.total_risk += self.calculate_risk(position['size'], position['stop_loss'])
        self.current_equity -= self.calculate_risk(position['size'], position['stop_loss'])
    
    def close_position(self, position):
        if position not in self.open_positions:
            raise Exception("Cannot close position: not found in open positions.")
        self.open_positions.remove(position)
        self.total_risk -= self.calculate_risk(position['size'], position['stop_loss'])
        self.current_equity += position['profit']
    
    def update_position(self, position, current_price):
        position['profit'] = (current_price - position['entry_price']) * position['size'] * position['direction']
        if position['direction'] == 1:
            position['stop_loss'] = max(position['stop_loss'], position['entry_price'] - position['profit'])
            if position['take_profit'] is not None and position['profit'] >= position['take_profit']:
                self.close_position(position)
        else:
            position['stop_loss'] = min(position['stop_loss'], position['entry_price'] + position['profit'])
            if position['take_profit'] is not None and position['profit'] <= -position['take_profit']:
                self.close_position(position)
        if position['stop_loss'] >= current_price and position['direction'] == 1:
            self.close_position(position)
        elif position['stop_loss'] <= current_price and position['direction'] == -1:
            self.close_position(position)



class PlaceCancelOrder:
    
    def __init__(self):
        self.orders = []
        self.units = None


    def place_order(self, signal, symbol, order_type):
        self.units = self.calculate_units()
        if signal == 'buy':
            self.orders.append({'side': 'buy',
                                'units': self.units,
                                'strategy': 'NNTS',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'sell':
            self.orders.append({'side': 'sell',
                                'units': self.units,
                                'strategy': 'NNTS',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})


    def cancel_order(self, order_id):
        request = orders.OrderCancel(self.account_id, orderID=order_id)
        self.client.request(request)
        for order in self.orders:
            if order['id'] == order_id:
                self.orders.remove(order)
                break
    




class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

class Bot:
    ib = None
    
    def __init__(self):
        self.ib = IBapi()
        self.ib.connect("192.168.56.1",7497,23467)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)
    
    def execute_trade(self, side, quantity, price):
        # create a new contract object
        print("execute")
        print("execute trade")
        contract = Contract()
        contract.symbol = 'EURUSD'
        contract.secType = "forex"  # change to the security type you are trading
        contract.currency = "EUR"  # change to the currency of the security
        contract.exchange = "MKT"  # change to the exchange you are trading on
        
        # create a new order object
        order = Order() 
        if side == 'BUY':
            order.action = 'BUY'
        if side == 'SELL':
            order.action = 'SELL'
        order.orderType = 'MKT'  # "LMT" for limit order, "MKT" for market order
        order.totalQuantity = quantity
        order.lmtPrice = price  # specify the price for limit orders
        
        # submit the order to the TWS
        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1
        
        # wait for the order to be filled
        time.sleep(1)
        print("sleep")
        # cancel the unfilled portion of the order
        remaining_quantity = order.totalQuantity - order.filledQuantity
        if remaining_quantity > 0:
            cancel_order = Order()
            cancel_order.action = "CANCEL"
            cancel_order.totalQuantity = remaining_quantity
            self.placeOrder(self.nextOrderId, contract, cancel_order)
            self.nextOrderId += 1

        today = datetime.datetime.today()
        if today.weekday() < 5:  # Check if today is a weekday (0 = Monday, ..., 4 = Friday)
            # Execute the trade
            signal = generate_signals()
            if signal == 'BUY':
                print("Going long...")
            elif signal == 'SELL':
                print("Going short...")
            else:
                print("No signal to trade with mate.")
        else:
            print("Today is a weekend, there is no trading. So chill-out dude...")
    def run_loop(self):
        self.ib.run()



# Run the loop
print("test 1")
bot = Bot()
print("test 2")    

# Call Market class

market = Market(symbol='EURUSD', yahoo_ticker='MSFT')
market.fx_price()
market.stock_price()

# Call the Balance class

balance = Balance()
balance.calculate_units()
balance.get_balance()
balance.accountSummary()

# Call the NNTS class

nnts = NNTS()
nnts._prepare_data
nnts._build_model
nnts.generate_signals

# Call the DataProcessor class

datapp = DataProcessor()
datapp.preprocess_data

# Call the RiskManager class

riskmg = RiskManager()
riskmg.update_equity
riskmg.calculate_risk
riskmg.can_open_position
riskmg.can_afford_position
riskmg.open_position
riskmg.close_position
riskmg.update_position

# Call PlaceCancelOrder class

pcorder = PlaceCancelOrder()
pcorder.place_order
pcorder.cancel_order

# Call the Bot class

bot = Bot()
bot.execute_trade
bot.run_loop
app.run
