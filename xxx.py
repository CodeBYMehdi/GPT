

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
from alpha_vantage.techindicators import TechIndicators


# Initialize Alpha Vantage API key

alphavantage_api_key = 'PUZNEBWHAHJE76R2'



class Market:
    def __init__(self, symbol, yahoo_ticker, currency='EUR', hist_window=365):
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
        price = float(price)
        print(f"EURUSD Current Spot: {price}")

        return price

    def stock_price(self):

        api_key = 'QUJ00N0C3VLU7NKC'
        symbol = 'MSFT'

        ts = TimeSeries(key=api_key, output_format='pandas')
        historical_data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        hist_data = historical_data[['4. close']]
        hist_data = hist_data.rename(columns={'4. close': 'Close'})
        hist_data.index = pd.to_datetime(hist_data.index)

        print(f"MSFT Historical Price: {hist_data['Close'].values[-1]}")

        ts = TimeSeries(key=api_key, output_format='pandas')
        real_time_data, meta_data = ts.get_quote_endpoint(symbol=symbol)
        current_price = real_time_data['05. price'].values[0]
        print(f"MSFT Current Price: {current_price}")

        return hist_data, current_price


    def market_to_dataframe(self):

        market = Market('symbol', 'yahoo_ticker', 'EURUSD', 365)

        fx_price = market.fx_price(real_time=True)
        hist_data, stock_price = market.stock_price()

        data = {
            'Symbol': ['EURUSD'],
            'Yahoo Ticker': ['MSFT'],
            'Currency': ['EUR'],
            'Historical Window': [market.hist_window],
            'FX Price': [fx_price],
            'MSFT Historical Price': hist_data['Close'].values[-1],
            'MSFT Current Price': [stock_price]
        }
        df = pd.DataFrame(data)

        return df








        


class IBapi(EWrapper, EClient):
    def __init__(self, bot):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.bot = bot


class BalanceApp(EWrapper, EClient):
    def __init__(self, ip_address, port_id, client_id):
        EClient.__init__(self, self)
        self.ip_address = ip_address
        self.port_id = port_id
        self.client_id = client_id
        self.account_balance = None

    def start(self):
        self.connect(self.ip_address, self.port_id, self.client_id)
        self.run()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        if tag == 'TotalCashValue':
            self.account_balance = float(value)


    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} - {errorCode} - {errorString}")
        if errorCode == 2104:  # Market data farm connection is OK
            return  # Ignore this error





class RiskManager:
    
    def __init__(self, balance, stop_loss_pct):
        self.balance = balance
        self.stop_loss_pct = stop_loss_pct
        self.max_loss_pct = 0.05  # maximum percentage of account balance that can be lost on a single trade
        self.take_profit_pct = self.calculate_max_take_profit_pct()
        
    def calculate_max_take_profit_pct(self):
        return self.max_loss_pct / (1 - self.stop_loss_pct)
        
    def calculate_order_size(self, current_price):
        risk_amount = self.balance * self.max_loss_pct
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        take_profit_price = current_price * (1 + self.take_profit_pct)
        
        order_size = risk_amount / (current_price - stop_loss_price)
        
        potential_profit = order_size * (take_profit_price - current_price)
        
        if potential_profit < risk_amount:
            order_size = risk_amount / (take_profit_price - current_price)
            
        return int(order_size)
        
    def calculate_risk(self, price, stop_loss):
        risk = self.balance * self.max_loss_pct
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
        self.risk_manager = RiskManager(balance=150, stop_loss_pct=0.05)

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

    def generate_signals(self, data, strategy, max_trades=300):
        avg_trades_per_day = int(len(data) / self.lookback)
        if avg_trades_per_day < 250:
            self.units *= 2
        elif avg_trades_per_day > 300:
            self.lookback = int(len(data) / 300)
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

        if np.count_nonzero(signals) > max_trades:
            excess_trades = np.count_nonzero(signals) - max_trades
            if excess_trades < np.count_nonzero(signals == 1):
                signals[signals == 1][:excess_trades] = 0
            else:
                signals[signals == -1][:excess_trades] = 0

        return signals





class TradingProcess:
    def __init__(self, balance, risk_percentage):
        self.balance = balance
        self.risk_percentage = risk_percentage
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

    def can_afford_position(self, price, size):
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
        profit = position['size'] * (price - position['price'])
        self.balance += profit
        self.profits.append(profit)
        return profit

    def update_position(self, index, price):
        position = self.positions[index]
        if price <= position['stop_loss']:
            return self.close_position(index, position['stop_loss'])
        else:
            position['profit'] = position['size'] * (price - position['price'])
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
        self.ib.connect("127.0.0.1", 7495, 1)
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
        contract.secType = "forex"  
        contract.currency = "EUR"  
        contract.exchange = "MKT"  

        order = Order() 
        if side == 'BUY':
            order.action = 'BUY'
        if side == 'SELL':
            order.action = 'SELL'
        order.orderType = 'MKT' 
        order.totalQuantity = quantity
        order.lmtPrice = price 
    
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
        if today.weekday() < 5: 
            # Execute the trade
            signal = generate_signals()
            if signal == 'BUY':
                print("Mechi position long...")
            elif signal == 'SELL':
                print("Mechi position short...")
            else:
                print("Ma fomech signal...")
        else:
            print("Lioum Weekend ma fomech des signaux...")
        
    def run_loop(self):
        self.ib.run()




#run loop

print("test 1")
bot = Bot()
print("test 2")    

# Call Market class

market = Market(symbol='EURUSD', yahoo_ticker='MSFT', currency='EUR', hist_window=365)
market.fx_price()
market.stock_price()
data=market.market_to_dataframe()

# Call the Balance class
ip_address = "127.0.0.1" 
port_id = 7495 
client_id = 1  
current_price=market.fx_price(real_time= True)
price=market.fx_price()


bot.nextorderId = None
bot.run_loop();
print("wa7el Houni");
balance = BalanceApp(ip_address,port_id,client_id)
balance.start()
balance.accountSummary(reqId=123, account="DU11643091", tag="TotalCashValue", value="12345", currency="EUR")
balance.error(reqId=123, errorCode=456, errorString="Some error message")

# Call the RiskManager class

riskmg = RiskManager(balance, stop_loss_pct=0.05)
max_take_profit_pct = riskmg.calculate_max_take_profit_pct()
print("Maximum take profit pct: ", max_take_profit_pct)
order_size=riskmg.calculate_order_size(current_price)
print("Order size:", order_size)
riskmg.calculate_risk(price, stop_loss=7.5)

# Call the NNTS class

nnts = NNTS(lookback=50, units=128, dropout=0.5, epochs=200, batch_size=64)
X, y=nnts._prepare_data(data)
model=nnts._build_model(X)
buy_signals=nnts.generate_signals(data, strategy='buy')
sell_signals=nnts.generate_signals(data, strategy='sell')

# Call the TradingProcess class

tp = TradingProcess(balance, risk_percentage=0.05)
tp.update_equity()
tp.can_open_position(price, stop_loss=0.05)
tp.can_afford_position(price)
tp.open_position(price, stop_loss=0.05)
tp.close_position(price)
tp.update_position(price)
tp.fit(X, y)
tp.predict(X)

# Call the DataProcessor class

datapp = DataProcessor(feature_collumns=["open","high", "low", "close", "volume"])
datapp.preprocess_data(data)


# Call PlaceCancelOrder class

pcorder = PlaceCancelOrder()
pcorder.place_order(symbol='EURUSD', order_type='MKT')
pcorder.cancel_order(order_id=1)

# Call Bot function
bot.execute_trade(price)
