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
from ibapi.common import *
from ibapi.ticktype import *
from ibapi.order import *



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
        


class IBapi(EWrapper, EClient):
    def __init__(self, bot):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.bot = bot



class Balance:
    def __init__(self):
        self.app = IBapi(bot)
        self.app.connect("127.0.0.1", 7497, 1)
        self.account_balance = None

    def calculate_units(self, portfolio_value, price):
        balance = self.get_balance()
        if balance is None:
            # Gérer le cas où la balance n'est pas disponible
            return None  # Ou une autre valeur appropriée

        max_units = balance / price
        return int(min(max_units, portfolio_value / price))

    def get_balance(self):
        self.app.reqAccountSummary(1, "All", "$LEDGER:EUR")
        self.app.run()

        return self.account_balance

    def accountSummary(self, reqId, account, tag, value, currency):
        if tag == "TotalCashValue":
            self.account_balance = value
            self.app.disconnect()
                    







class RiskManager:
    
    def __init__(self, balance, stop_loss_pct, take_profit_pct):
        self.balance = balance
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def calculate_order_size(self, current_price):
        max_loss_pct = 0.5  # maximum percentage of account balance that can be lost on a single trade
        risk_amount = self.balance * max_loss_pct
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        take_profit_price = current_price * (1 + self.take_profit_pct)
        
        # calculate number of shares to buy based on risk amount and stop loss price
        order_size = risk_amount / (current_price - stop_loss_price)
        
        # calculate potential profit based on take profit price
        potential_profit = order_size * (take_profit_price - current_price)
        
        # if potential profit is less than risk amount, reduce order size to minimize risk
        if potential_profit < risk_amount:
            order_size = risk_amount / (take_profit_price - current_price)
            
        return int(order_size)
        
    def calculate_risk(self, price, stop_loss):
        risk = (self.risk_percentage / 100) * self.balance
        max_loss = price - stop_loss
        position_size = risk / max_loss

        return position_size




class NNTS:
    
    def __init__(self, lookback, units, dropout, epochs, batch_size):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.risk_manager = RiskManager()

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

    def generate_signals(self, data, strategy, max_trades=2000):
        avg_trades_per_day = int(len(data) / self.lookback)
        if avg_trades_per_day < 500:
            self.units *= 2
        elif avg_trades_per_day > 2000:
            self.lookback = int(len(data) / 2000)
        batch_size = max(int(avg_trades_per_day / self.epochs), 1)
        X, y = self._prepare_data(data)
        model = self._build_model(X)
        model.fit(X, y, epochs=self.epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X)
        signals = np.zeros(len(data))
        signals[self.lookback:] = np.where(y_pred > y, 1, -1)
        signals = self.risk_manager.filter_signals(signals, data)
        
        if strategy == 'buy':
            signals[signals != 1] = 0
        elif strategy == 'sell':
            signals[signals != -1] = 0
        else:
            signals = np.zeros(len(data))
            
        if np.count_nonzero(signals) > max_trades:
            excess_trades = np.count_nonzero(signals) - max_trades
            if excess_trades < np.count_nonzero(signals == 1):
                signals[signals == 1][:excess_trades] = 0
            else:
                signals[signals == -1][:excess_trades] = 0
                
        return signals





class TradingProcess:
    def __init__(self, balance, risk_percentage, transaction_fee):
        self.balance = balance
        self.risk_percentage = risk_percentage
        self.transaction_fee = transaction_fee
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)

        self.positions = []
        self.profits = []

    def update_equity(self):
        equity = self.balance
        for position in self.positions:
            equity += position['profit']
        return equity


    def can_open_position(self, price, stop_loss):
        position_size = self.calculate_risk(price, stop_loss)
        return self.balance >= position_size * price

    def can_afford_position(self, price, stop_loss, size):
        position_cost = size * price
        return self.balance >= position_cost

    def open_position(self, price, stop_loss, size):
        self.balance -= size * price
        self.positions.append({
            'price': price,
            'stop_loss': stop_loss,
            'size': size,
            'profit': 0.0
        })

    def close_position(self, index, price):
        position = self.positions.pop(index)
        profit = position['size'] * (price - position['price']) - 2 * self.transaction_fee
        self.balance += profit
        self.profits.append(profit)
        return profit

    def update_position(self, index, price):
        position = self.positions[index]
        if price <= position['stop_loss']:
            return self.close_position(index, position['stop_loss'])
        else:
            position['profit'] = position['size'] * (price - position['price']) - 2 * self.transaction_fee
            return position['profit']

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)






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
    






class Bot:
    ib = None
    
    def __init__(self):
        self.ib = IBapi(self)
        self.ib.connect("127.0.0.1", 7497, 1)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)

    def connectAck(self):
        print("Connected to TWS")

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
        self.ib.placeOrder(self.ib.nextOrderId, contract, order)
        self.ib.nextOrderId += 1
    
        # wait for the order to be filled
        time.sleep(1)
        print("sleep")
    
        # cancel the unfilled portion of the order
        remaining_quantity = order.totalQuantity - order.filledQuantity
        if remaining_quantity > 0:
            cancel_order = Order()
            cancel_order.action = "CANCEL"
            cancel_order.totalQuantity = remaining_quantity
            self.ib.placeOrder(self.ib.nextOrderId, contract, cancel_order)
            self.ib.nextOrderId += 1

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
portfolio_value = 150
fx_price = market.fx_price()  # Récupérer la valeur du prix depuis le marché
units = balance.calculate_units(portfolio_value, fx_price)
balance.get_balance()
balance.accountSummary(portfolio_value, fx_price, currency = "EUR", tag = "TotalCashValue", value = balance)

# Call the RiskManager class

riskmg = RiskManager(balance)
riskmg.update_equity()
riskmg.calculate_risk()
riskmg.can_open_position()
riskmg.can_afford_position()
riskmg.open_position()
riskmg.close_position()
riskmg.update_position()

# Call the NNTS class

nnts = NNTS()
nnts._prepare_data()
nnts._build_model()
nnts.generate_signals()

# Call the TradingProcess class

tp = TradingProcess()
tp.update_equity()
tp.can_open_position()
tp.can_afford_position()
tp.open_position()
tp.close_position()
tp.update_position()
tp.fit()
tp.predict()

# Call the DataProcessor class

datapp = DataProcessor()
datapp.preprocess_data()


# Call PlaceCancelOrder class

pcorder = PlaceCancelOrder()
pcorder.place_order()
pcorder.cancel_order()

# Call Bot function

bot = Bot()
bot.execute_trade()
bot.run_loop()
