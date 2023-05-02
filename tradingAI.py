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
        



class MACDBbands:
    def __init__(self):
        self.data = None
        self.macd = None
        self.bbands = None
        
    def load_data(self):

        
        # initialize MACD and Bollinger Bands
        self.macd = MACD(self.data['Close'])
        self.bbands = BollingerBands(self.data['Close'])

        # Error handlin
        try:
            self.macd = MACD(self.data['Close'])
            self.bbands = BollingerBands(self.data['Close'])
        except ValueError:
            logging("Error initializing technical indicators. Check your data.")
            return

    def generate_macd_bbands_signal(self, curr_price):
        # Get MACD signal
        macd_signal = self.macd.macd_signal()[-1]
        macd = self.macd.macd()[-1]

        # Get Bollinger Bands signal
        upper_band = self.bbands.bollinger_hband()[-1]
        lower_band = self.bbands.bollinger_lband()[-1]

        if macd_signal > macd and curr_price > upper_band:
            signal = 'buy_macd'
        elif macd_signal < macd and curr_price < lower_band:
            signal = 'sell_macd'
        else:
            signal = None

        return signal






        
    def run_strategy(self):
        # use self.macd and self.bbands to make trading decisions
        pass



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
        ticker = Ticker(self.symbol)
        self.data = ticker.history(period=self.timeframe)
        
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
        self.orders={}
    
    def algorithmic_strategy(self, prices):
        ma50 = prices.rolling(window=50).mean()
        ma200 = prices.rolling(window=200).mean()
        signals = np.zeros(prices.shape)
        signals[ma50 > ma200] = 1
        signals[ma50 < ma200] = -1
        return signals

    def place_order(self, order_type, units, instrument):
        data = {
            "order": {
                "units": units,
                "instrument": instrument,
             "type": order_type,
                "stopLossOnFill": {
                    "timeInForce": "GTC",
                    "price": str(self.stop_loss)
                },
             "takeProfitOnFill": {
                    "timeInForce": "GTC",
                    "price": str(self.take_profit)
                 }
            }
        }
        r = self.api.request("POST", "/v3/accounts/" + self.account_id + "/orders", data=data)
        print(r.status_code, r.reason, r.headers)
        print(r.data)

    def buy_algo(prices, short_window=10, long_window=30):
   # Generate a buy signal using a simple moving average crossover strategy
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        crossover = (short_ma > long_ma) & (short_ma.shift(1) < long_ma.shift(1))
        return crossover.astype(int)

    def sell_algo(prices, short_window=10, long_window=30):
        #Generate a sell signal using a simple moving average crossover strategy
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        crossover = (short_ma < long_ma) & (short_ma.shift(1) > long_ma.shift(1))
        return crossover.astype(int)


    



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

        


    def generate_signals(self, prices, supply, demand, trend, stop_loss, take_profit, threshold=0.05):
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
            assert len(supply) == len(demand), "Supply and demand arrays must have the same length"
            buy_signals = np.zeros(len(supply), dtype=bool)
            sell_signals = np.zeros(len(supply), dtype=bool)

            for i in range(1, len(supply)):
                supply_ratio = supply[i] / supply[i-1]
                demand_ratio = demand[i] / demand[i-1]
                if supply_ratio > (1 + threshold) and demand_ratio < (1 - threshold):
                # Strong supply increase and demand decrease, signal a sell
                    sell_signals[i] = True
                elif supply_ratio < (1 - threshold) and demand_ratio > (1 + threshold):
                # Strong supply decrease and demand increase, signal a buy
                    buy_signals[i] = True
            return buy_signals, sell_signals






    
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





class Backtester:
    def __init__(self, strategy, data, stock, fx, trend, stop_loss, take_profit):
        self.strategy = strategy
        self.data = data
        self.stock = stock
        self.fx = fx
        self.trend = trend
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def load_data(self):
        if isinstance(self.data, str):
            if self.data == "msft":
                msft = yf.Ticker("MSFT")
                self.data = msft.history(period="max")
            elif self.data == "fx":
                fx = ForeignExchange(key=alphavantage_api_key, output_format='pandas')
                data, meta_data = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='compact')
                self.data = data

    def run_backtest(self):
        self.load_data()
        sdt = SupplyDemandTrader(self.stock, self.fx)
        supply_data = sdt.get_supply_data()
        demand_data = sdt.get_demand_data()
        signals = self.strategy.generate_signals(self.data, supply_data, demand_data, self.trend, self.stop_loss, self.take_profit)
        positions = self.strategy.generate_positions(signals)
        portfolio = self.strategy.calculate_portfolio(positions, self.data)
        returns = self.strategy.calculate_returns(portfolio)

        return returns









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
                                'strategy': 'supply_demand',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'sell':
            self.orders.append({'side': 'sell',
                                'units': self.units,
                                'strategy': 'supply_demand',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'buy_macd':
            self.orders.append({'side': 'buy',
                                'units': self.units,
                                'strategy': 'macd_bbands',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'sell_macd':
            self.orders.append({'side': 'sell',
                                'units': self.units,
                                'strategy': 'macd_bbands',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'buy_algo':
            self.orders.append({'side': 'buy',
                                'units': self.units,
                                'strategy': 'algo_trading',
                                'symbol': symbol,
                                'type': order_type,
                                'stop_loss': self.stop_loss,
                                'take_profit': self.take_profit})
        elif signal == 'sell_algo':
            self.orders.append({'side': 'sell',
                                'units': self.units,
                                'strategy': 'algo_trading',
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



# Run the IBapi class
print("test 1")
bot = Bot()
print("test 2")    
fx_price = "fx"
stock_price = "stock"
market = Market(symbol='EURUSD', yahoo_ticker='MSFT')
sdt = SupplyAndDemandTrader(market, fx_price, stock_price)
supply_data = sdt.get_market_supply()
demand_data = sdt.get_market_demand()
    # Run the Market class
market.fx_price()
market.stock_price()

    # Run the Backtester class
backtest = Backtester(sdt, "fx", "stock", supply_data, demand_data, stop_loss, take_profit)

backtest.run_backtest()
backtest.preprocess__data()

    # Run the Data processor class
dataprocess = DataProcessor()
dataprocess.preprocess_data()

    # Run the MACDBbands class
macdbb = MACDBbands()
macdbb.load_data()
macdbb.run_strategy()


    # Run the Technical Indicators class
indicators = TechnicalIndicators()
indicators.volumes()
indicators.macd()
indicators.bollinger_bands()

    # Run the Algo trading class
algotrading = AlgoTrading()
algotrading.algorithmic_strategy()


    # Run the Supply and Demand class
supplydemand = SupplyAndDemandTrader()
supplydemand.get_market_demand()
supplydemand.get_market_supply()
supplydemand.generate_signals()
supplydemand.generate_positions()
supplydemand.calculate_portfolio()
supplydemand.calculate_returns()
supplydemand.generate_prompt()
supplydemand.make_decision()
supplydemand.get_balance()


    # Run the Risk manager class
riskmanagement = RiskManager()
riskmanagement.update_equity()
riskmanagement.calculate_risk()
riskmanagement.can_open_position()
riskmanagement.can_afford_position()
riskmanagement.open_position()
riskmanagement.close_position()
riskmanagement.update_position()
app.run()
