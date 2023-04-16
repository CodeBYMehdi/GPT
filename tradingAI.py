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
import ib_insync

# Import classic modules

import openai
import requests
import exchange
import json
import time
import datetime
import numpy as np
import ta
import asyncio
import pandas as pd
import pandas_datareader.data as web
import ccxt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import the machine learning

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

    def __init__(self, symbol):
        self.symbol = symbol

    def fx_price(self, real_time=True):
        if real_time:
            fx = ForeignExchange(key=alphavantage_api_key, output_format='pandas')
            data, meta_data = fx.get_currency_exchange_rate(from_currency='EUR', to_currency='USD')
            price = data['4. To_Currency Name'][0]
        else:
            fx = ForeignExchange(key=alphavantage_api_key, output_format='pandas')
            data, meta_data = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='compact')
            price = data['4. close'].iloc[-1]
        return float(price)

    def stock_price(self):
    # Get historical data for MSFT stock
        msft = yf.Ticker("MSFT")
        hist_data = msft.history(period="max")
        hist_data = hist_data[['Close']]  # Select only the closing price column

    # Get real-time data for MSFT stock
        msft_info = msft.info
        current_price = msft_info["regularMarketPrice"]
    
        return hist_data, current_price

class Backtester():

    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run_backtest(self, X_train, y_train, X_test):
        signals = self.strategy.generate_signals(self.data)
        positions = self.strategy.generate_positions(signals)
        portfolio = self.strategy.calculate_portfolio(positions, self.data)
        returns = self.strategy.calculate_returns(portfolio)
        
        # Define the deep learning model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Use the model to generate trading signals
        predicted_prices = model.predict(X_test)
        
        return returns


    def preprocess_data(data):
    # Drop any rows with NaN values
        data.dropna(inplace=True)
    
    # Add technical indicators
        data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume")
    
    # Define the features to use
        feature_columns = [
            "volume", "volume_adi", "volume_obv", "volume_vpt", 
            "volatility_atr", "trend_macd_signal", "trend_macd_diff", "trend_ema_fast",
            "momentum_rsi", "momentum_kama", "momentum_stoch_signal"
        ]
    
    # Create X and y
        X = data[feature_columns].values
        y = np.where(data["close"].shift(-1) > data["close"], 1, -1)
        y = y[:-1]
    
    # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
        return X, y




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


class MyTradingStrategy:
    def __init__(self):
        self.data = None
        self.macd = None
        self.bbands = None
        
    def load_data(self):

        
        # initialize MACD and Bollinger Bands
        self.macd = MACD(self.data['Close'])
        self.bbands = BollingerBands(self.data['Close'])
        
    def run_strategy(self):
        # use self.macd and self.bbands to make trading decisions
        pass

    def place_order(self, signal):
        units = self.calculate_units()
        if signal == 'buy':
            self.orders.append({'side': 'buy',
                                'units': units,
                                'strategy': 'supply_demand'})
        elif signal == 'sell':
            self.orders.append({'side': 'sell',
                                'units': units,
                                'strategy': 'supply_demand'})
        elif signal == 'buy_macd':
            self.orders.append({'side': 'buy',
                                'units': units,
                                'strategy': 'macd_bbands'})
        elif signal == 'sell_macd':
            self.orders.append({'side': 'sell',
                                'units': units,
                                'strategy': 'macd_bbands'})
        elif signal == 'buy_algo':
            self.orders.append({'side': 'buy',
                                'units': units,
                                'strategy': 'algo_trading'})
        elif signal == 'sell_algo':
            self.orders.append({'side': 'sell',
                                'units': units,
                                'strategy': 'algo_trading'})

    def cancel_order(self, order_id):
        order_found = False
        for order in self.orders:
            if order.get('id') == order_id:
                self.orders.remove(order)
                print(f"Order {order_id} has been cancelled.")
                order_found = True
                break
        if not order_found:
            print(f"No order found with id {order_id}.")




class TechnicalIndicators:
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = yf.download(self.symbol, period=self.timeframe)
        
    def volumes(self):
        return self.data['Volume']
    
    def macd(self):
        return ta.trend.macd_diff(self.data['Close'])
    
    def bollinger_bands(self):
        return ta.volatility.bollinger_hband(self.data['Close']), ta.volatility.bollinger_lband(self.data['Close'])


class AlgoTrading():
    
    def __init__(self, starting_funds):
        self.funds = starting_funds
        self.trade_history = []
    
    def algorithmic_strategy(self, prices):
        ma50 = prices.rolling(window=50).mean()
        ma200 = prices.rolling(window=200).mean()
        signals = np.zeros(prices.shape)
        signals[ma50 > ma200] = 1
        signals[ma50 < ma200] = -1
        return signals




class SupplyAndDemandTrader:
    
    def __init__(self, market, fx, stock):
        self.market = market
        self.fx = fx
        self.stock = stock
        
    def get_market_demand(self):
        if self.market == "forex":
            demand_data = self.fx
            return demand_data
        elif self.market == "stocks":
            demand_data = self.stock['Close']
            return demand_data
        else:
            return {}

    def get_market_supply(self):
        if self.market == "forex":
            supply_data = self.fx
            volume_data = self.fx
            return supply_data
        elif self.market == "stocks":
            supply_data = self.stock['Close']
            volume_data = self.stock['Volume']
            return supply_data * volume_data
        else:
            return {}

        


    def generate_signals(self, prices, supply, demand, trend, stop_loss, take_profit):
        avg_supply = np.convolve(supply, np.ones(10) / 10, mode='valid')
        avg_demand = np.convolve(demand, np.ones(10) / 10, mode='valid')

        # Calculate the ratio of average demand to average supply
        demand_supply_ratio = avg_demand / avg_supply

        # Generate trading signals based on the ratio of demand to supply
        signals = np.zeros(prices.shape)
        signals[demand_supply_ratio > 1.05] = 1
        signals[demand_supply_ratio < 0.95] = -1

        # Apply stop loss and take profit
        max_price = np.max(prices)
        min_price = np.min(prices)
        for i in range(1, len(prices)):
            if signals[i] == 1:
                if prices[i] < (1 - stop_loss) * max_price:
                    signals[i] = -1
                elif prices[i] > (1 + take_profit) * max_price:
                    signals[i] = 0
            elif signals[i] == -1:
                if prices[i] > (1 + stop_loss) * min_price:
                    signals[i] = 1
                elif prices[i] < (1 - take_profit) * min_price:
                    signals[i] = 0

        if trend == 'range':
        # Apply the supply and demand strategy
            avg_supply = np.convolve(supply, np.ones(10) / 10, mode='valid')
            avg_demand = np.convolve(demand, np.ones(10) / 10, mode='valid')
            demand_supply_ratio = avg_demand / avg_supply
            signals = np.zeros(prices.shape)
            signals[demand_supply_ratio > 1.05] = 1
            signals[demand_supply_ratio < 0.95] = -1
        else:
        # Apply the algorithmic trading strategy
            signals = np.zeros(prices.shape)

        return signals



    
    def generate_positions(self, signals):
        # implement your position sizing logic here
        positions = signals.diff()
        positions.iloc[0] = signals.iloc[0]
        return positions
    
    def calculate_portfolio(self, positions, data):
        # implement your portfolio calculation logic here
        portfolio = pd.DataFrame(index=data.index)
        portfolio['position'] = positions
        portfolio['price'] = data['close']
        portfolio['value'] = portfolio['position'] * portfolio['price']
        portfolio['returns'] = portfolio['value'].pct_change()
        return portfolio
    
    def calculate_returns(self, portfolio):
        # implement your returns calculation logic here
        returns = portfolio['returns']
        returns = returns[~pd.isnull(returns)]
        return returns




    def generate_prompt(market):
    
    # Define prompt prefix based on market
        if market == "forex":
            prompt_prefix = "Trade forex based on supply and demand. The current forex market is "
        elif market == "stocks":
            prompt_prefix = "Trade stocks based on supply and demand. The current stock market is "
        else:
            raise ValueError("Market must be 'forex' or 'stocks'.")

    # Define prompt suffix based on supply and demand in the market
        
        demand = get_market_demand(market)
        supply = get_market_supply(market)
        
        if demand > supply:
            prompt_suffix = "The market is currently in high demand, so look to buy or go long."
        elif demand < supply:
            prompt_suffix = "The market is currently oversupplied, so look to sell or go short."
        else:
            prompt_suffix = "The market is currently balanced, so look for other opportunities."

    # Combine prompt prefix and suffix
        prompt = prompt_prefix + prompt_suffix

    # Use GPT-3 API to generate trade prompt
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=60,
            n=1,
            stop=None,
            temperature=0.5,
            )

        return response.choices[0].text.strip()

    # Define function to make a decision based on input data
    def make_decision(data):
    # Define MLPClassifier model with 2 hidden layers of size 10 each
        clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
    
    # Split data into features (X) and labels (y)
        X = data[:, :-1]
        y = data[:, -1]

    
    # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Train the model on the data
        clf.fit(X, y)
    
    # Make a prediction on the last row of data
        prediction = clf.predict(X[-1].reshape(1, -1))[0]
    
    # Return the decision (0 or 1)
        return prediction

    def calculate_units(self, portfolio_value, price):
        # calculate the maximum number of units that can be bought with the portfolio value and the price
        balance = self.get_balance()
        max_units = balance / price
        return int(min(max_units, portfolio_value / price))



    def get_balance(self):
        # implement a function to get the account balance from the IB API
        # here is an example implementation that assumes that the account currency is USD
        class MyEWrapper(EWrapper):
            def __init__(self):
                self.account_balance = None

            def accountSummary(self, reqId, account, tag, value, currency):
                if tag == "TotalCashValue":
                    self.account_balance = value



        
class RiskManager:
    def __init__(self, max_risk_per_trade=0.02, max_open_positions=5):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_positions = max_open_positions
        self.open_positions = []
        self.current_equity = 0
        self.total_risk = 0
    
    def update_equity(self, equity):
        self.current_equity = equity
    
    def calculate_risk(self, position_size, stop_loss):
        return position_size * stop_loss
    
    def can_open_position(self):
        return len(self.open_positions) < self.max_open_positions
    
    def can_afford_position(self, position_size):
        return self.current_equity * self.max_risk_per_trade >= self.calculate_risk(position_size)
    
    def open_position(self, position):
        if not self.can_open_position():
            raise Exception("Cannot open position: max open positions reached.")
        if not self.can_afford_position(position['size']):
            raise Exception("Cannot open position: insufficient funds or excessive risk.")
        
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
        else:
            position['stop_loss'] = min(position['stop_loss'], position['entry_price'] + position['profit'])
        if position['stop_loss'] >= current_price and position['direction'] == 1:
            self.close_position(position)
        elif position['stop_loss'] <= current_price and position['direction'] == -1:
            self.close_position(position)


    




class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextOrderId = 0
        self.data = []

    def execute_trade(self, side, quantity, price):
        # create a new contract object
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
        time.sleep(5)
        
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
                print("No signal to trade mate.")
        else:
            print("Today is a weekend, no trades will be executed.")



app = IBapi()
app.connect('192.168.56.1', 7497, 23467)
app.run()

