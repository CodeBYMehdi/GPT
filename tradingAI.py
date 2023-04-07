"""

                              Made by @mehdibj

"""

# Import modules

import openai
import requests
import exchange
import json
import time
import datetime
import numpy as np
import ta
import pandas as pd
import pandas_datareader.data as web
import ccxt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import machine learning modules

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Import IB modules

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import ib_insync

# Deep learning modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Finance Module
import blpapi

# Initialize OpenAI API key

openai.api_key = "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"

# Initialize Quandl API KEY

nasdaq_api_key = "L_iAPLZ7AkwV9NpH5L44"

# Import Dataset
import nasdaqdatalink

from nasdaqdatalink import Dataset, get



from fredapi import Fred

# Fred API key

fred = Fred (api_key = '3c42f5fbde4207ebc90bbbf7c2d47beb')

# Fetch data from Nasdaq database

forex_data = fred.get_series('DEXUSEU').tail(1)[0]
print(type(forex_data))

stocks_data = get("NASDAQOMX/COMP-INDEX")
print(type(stocks_data))







# Define function to generate trade prompt based on supply and demand using the GPT-3 API

def generate_prompt(market):
    
    # Define prompt prefix based on market
    
    if market == "forex":
        prompt_prefix = "Trade forex based on supply and demand. The current forex market is "
    elif market == "stocks":
        prompt_prefix = "Trade stocks based on supply and demand. The current stock market is "

        
    # Define prompt suffix based on supply and demand in the market
    if get_market_demand(market) > get_market_supply(market):
        prompt_suffix = "The market is currently in high demand, so look to buy or go long."
    elif get_market_demand(market) < get_market_supply(market):
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

markets = []

# Forex symbols
forex_symbols = ["EUR/USD", "USD/JPY", "GBP/USD"]
markets.extend(forex_symbols)

# Stock symbols
stock_symbols = ["AAPL", "AMZN", "GOOGL"]
markets.extend(stock_symbols)



# Print all symbols in markets
for symbol in markets:
    print(symbol)


# Print the list of available markets


for symbol in markets:
    print(symbol)



# Define function to get market demand based on recent price movements and all trades in the past

# Define function to get market demand based on recent price movements and all trades in the past




# Define function to get market demand based on the bid size of the latest quote
def get_market_demand(market, stocks_data, last_demand):
    if market == "forex":
        demand_data = forex_data
        return demand_data
    elif market == "stocks":
        demand_data = stocks_data
        return demand_data

    else:
        return {}

# Define function to get market supply based on recent volume movements and all trades in the past
def get_market_supply(market, stocks_data, last_demand):
    if market == "forex":
        supply_data = forex_data
        return supply_data
    elif market == "stocks":
        supply_data = stocks_data
        volume_data = stocks_data
        return supply_data * volume_data
    else:
        return {}




# Define function to execute trades based on generated prompts
def execute_trade(exchange, market, side, type, take_profit, price, symbol, amount):
    # Get market demand and supply data
    demand = get_market_demand(market, forex_data, stocks_data)
    supply = get_market_supply(market, forex_data, stocks_data)
    
    # Process data and make a decision
    data = np.array([demand, supply, side, type, take_profit, price, amount])
    decision = make_decision(data)
    
    # If decision is to execute trade, generate prompt and execute trade
    if decision == 1:
        # Generate trade prompt based on supply and demand using the GPT-3 API
        prompt = generate_prompt(market)

        # Print trade prompt
        print("Trade prompt:", prompt)

        # Execute trade based on prompt using the market API
        try:
            order = exchange.create_order(symbol, type, side, amount, price)
            print(f"Order {order['id']} successfully created")
            trade_placed = True
            return trade_placed
        except (ccxt.ExchangeError, ccxt.NetworkError) as error:
            print(f"Failed to create order: {error}")
            return False
    
    # If decision is not to execute trade, return False
    else:
        return False

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





    # Define function to execute trades based on generated prompts

# Define function to execute trades based on generated prompts

def execute_trade(exchange, market, side, type, take_profit, price, symbol, amount):

    # Get market demand and supply data
    demand = get_market_demand(market, forex_data, stocks_data)
    supply = get_market_supply(market, forex_data, stocks_data)
    
    # Process data and make a decision
    data = np.array([demand, supply, side, type, take_profit, price, amount])
    decision = make_decision(data)
    
    # If decision is to execute trade, generate prompt and execute trade
    if decision == 1:
        # Generate trade prompt based on supply and demand using the GPT-3 API
        prompt = generate_prompt(market)

        # Print trade prompt
        print("Trade prompt:", prompt)

        # Execute trade based on prompt using the market API
        if market == "forex":
            market_data = forex_data
        elif market == "stocks":
            market_data = stocks_data

            
        trade_placed = True
        return trade_placed
    
    # If decision is not to execute trade, return False
    else:
        return False
        
# Error handling
        try:
            order = exchange.create_order(symbol, trade['type'], trade['side'], trade['amount'], trade['price'])
            print(f"Order {order['id']} successfully created")
        except (ccxt.ExchangeError, ccxt.NetworkError) as error:
            print(f"Failed to create order: {error}")





# Begin main program loop
# Main loop

while True:
    # Check each market and execute a trade if conditions are met
    for market in ["forex", "stocks"]:
        demand = get_market_demand(market, forex_data, stocks_data)
        supply = get_market_supply(market, forex_data, stocks_data)

        # Set threshold values for high demand and oversupply
        demand_threshold = 0.7
        supply_threshold = 0.6

        # Execute a trade if the demand is high and the supply is low
        # Execute a trade if the demand is high and the supply is low
if (demand > demand_threshold).any() and (supply < supply_threshold).any():
    take_profit = 0.05  # Set take profit value as 5% of the account balance
    execute_trade(market, take_profit)

# Execute a trade if the demand is low and the supply is high
elif (demand < 1 - demand_threshold).any() and (supply > 1 - supply_threshold).any():
    take_profit = 0.05  # Set take profit value as 5% of the account balance
    execute_trade(market, take_profit)


# Define CHECK_INTERVAL
CHECK_INTERVAL = 60
    # Wait for a certain amount of time before checking the markets again
time.sleep(CHECK_INTERVAL)

# Initialize a dictionary to keep track of the last trade prices for each market
last_trade_prices = {}




# Define function to train neural network on market data
def train_model(market):
    # Get historical market data
    data = get_market_data(market)
    # Preprocess data
    X = np.array(data["features"])
    y = np.array(data["labels"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train neural network
    clf = MLPClassifier(random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler


# Define function to predict market conditions using trained neural network
def predict_market_conditions(clf, scaler, data):
    X = np.array(data)
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    return y_pred[0]

# Define function to generate trade prompt based on supply and demand using machine learning
def generate_prompt(market):
    # Train neural network on market data
    clf, scaler = train_model(market)
    # Get current market data
    data = get_current_market_data(market)
    # Predict market conditions
    market_condition = predict_market_conditions(clf, scaler, data)
    # Define prompt prefix based on market and predicted market condition
    if market == "forex":
        prompt_prefix = "Trade forex based on supply and demand. The current forex market is "
    elif market == "stocks":
        prompt_prefix = "Trade stocks based on supply and demand. The current stock market is "

    if market_condition == 0:
        prompt_suffix = "The market is currently in high demand, so look to buy or go long."
    elif market_condition == 1:
        prompt_suffix = "The market is currently oversupplied, so look to sell or go short."
    else:
        prompt_suffix = "The market is currently balanced, so look for other opportunities."
    prompt = prompt_prefix + prompt_suffix
    return prompt

# Define function to get current market data
def get_current_market_data(market):
    if market == "forex":
        # TODO: Get current forex market data
        pass
    elif market == "stocks":
        # TODO: Get current stock market data
        pass

    return []

# Define function to predict market movements using a Multi-Layer Perceptron Classifier


def predict_market_movements(market, historical_data):
    """
    Predicts the future market movement based on historical data using a Multi-Layer Perceptron Classifier.
    
    :param market: (str) The name of the market (e.g. "forex", "stocks").
    :param historical_data: (array) The historical data to be used for predicting the market movement.
    
    :return: (int) The predicted market movement (1 for a price increase, 0 for a price decrease), the latest price, and the z-score.
    """
    # Initialize the scaler and MLPClassifier
    scaler = StandardScaler()
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

    # Preprocess the data
    X = scaler.fit_transform(historical_data[:-1])
    y = historical_data[-1]

    # Train the model
    clf.fit(X, y)

    # Get the most recent price
    if market == "forex":
        response = requests.get()
        price_data = json.loads(response.content)
        latest_price = price_data['quoteResponse']['result'][0]['ask']
    elif market == "stocks":
        response = requests.get()
        price_data = json.loads(response.content)
        latest_price = price_data['quoteResponse']['result'][0]['ask']


    # Calculate the standard deviation and z-score of the price movements
    price_movements = np.diff(historical_data[:-1])
    std_dev = np.std(price_movements)
    z_score = (latest_price - historical_data[-2]) / std_dev

    # Predict the market movement
    if z_score > norm.ppf(0.8):
        predicted_movement = 1
    elif z_score < norm.ppf(0.2):
        predicted_movement = 0
    else:
        predicted_movement = clf.predict(scaler.transform([historical_data[:-1][-1]]))[0]

    return predicted_movement, latest_price, z_score

# Define function to get market data
def get_market_data(market):
    # TODO: Get historical market data
    
    return {"features": [], "labels": []}

# Define function to predict market conditions using trained neural network
def predict_market_conditions(clf, scaler, data):
    X = np.array(data)
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    return y_pred[0]

# Get market data
market_data = get_market_data("forex")

# Predict market movements
predicted_movement, latest_price, z_score = predict_market_movements("forex", market_data["features"])

# Plot predicted market movement
time_steps = range(len(market_data["labels"]))
plt.plot(time_steps[:-1], market_data["features"][:-1], label="Market Price")
plt.plot(time_steps[-1], latest_price, "ro", label="Latest Price")
plt.plot(time_steps[-1], predicted_movement, "go", label="Predicted Movement")
plt.legend()
plt.show()

# Get market data
market_data = get_market_data("forex")

# Predict market movements
predicted_movement, latest_price, z_score = predict_market_movements("forex", market_data["features"])

# Plot predicted market movement
time_steps = range(len(market_data["labels"]))

plt.plot(time_steps[:-1], market_data["features"][:-1], label="Market Price")
plt.plot(time_steps[-1], latest_price, "ro", label="Latest Price")

# Add predicted movement arrow
if predicted_movement == 1:
    arrow_color = "green"
    arrow_direction = "up"
else:
    arrow_color = "red"
    arrow_direction = "down"

plt.annotate(
    "",
    xy=(time_steps[-1], latest_price),
    xytext=(time_steps[-1], latest_price * 0.95),
    arrowprops=dict(arrowstyle=f"->, head_length=0.5, head_width=0.3", color=arrow_color),
)

plt.text(
    time_steps[-1] + 0.1,
    latest_price * 0.95,
    f"{arrow_direction}",
    fontsize=12,
    color=arrow_color,
)

plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()


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
    
    # Set the take profit to a value that maximizes potential profit
 
    take_profit = 0.05 * self.current_equity 
    
    # Set the stop loss to a value that minimizes potential loss
    # For example, you can set it to 1% of the account balance
    stop_loss = 0.01 * self.current_equity
    
    # Update the position dictionary to include the take profit and stop loss values
    position.update({'take_profit': take_profit, 'stop_loss': stop_loss})
    
    self.open_positions.append(position)
    self.total_risk += self.calculate_risk(position['size'], stop_loss)
    self.current_equity -= self.calculate_risk(position['size'], stop_loss)

    
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

class Backtester():
    
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        
    def run_backtest(self):
        signals = self.strategy.generate_signals(self.data)
        positions = self.strategy.generate_positions(signals)
        portfolio = self.strategy.calculate_portfolio(positions, self.data)
        returns = self.strategy.calculate_returns(portfolio)
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

        
class SecondStrategy():
    
    def generate_signals(self, data):
        # implement your signal generation logic here
        signals = pd.Series(index=data.index)
        signals[data['close'] > data['sma50']] = 1
        signals[data['close'] < data['sma50']] = -1
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

# Execute trades based on the trading signals generated by the model using the IB API
# Define your deep learning model here
def my_model(signal_data):
    # Implementation of your deep learning model
    # Returns a BUY, SELL or HOLD signal
    return signal

# Define a function to calculate the biggest variation for a market
def get_biggest_variation(market_history):
    market_history['variation'] = market_history['Close'].pct_change()
    return market_history['variation'].abs().nlargest(1).index[0]


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextOrderId = 0

    def placeOrder(self, orderId, contract, order):
        self.placeOrder(orderId, contract, order)

    def place_order(contract_symbol, contract_secType, contract_exchange, contract_currency, order_type, order_action, order_quantity, order_price, order_id):
        app = IBapi()
        app.connect("192.168.56.1", 7497, clientId=23467)

        # Calculate order quantity

        quantity = calculate_units(contract_symbol, signal)

        # Create order object

        order = Order()
        if signal == 'BUY':
            order.action = 'BUY'
        elif signal == 'SELL':
            order.action = 'SELL'
        order.totalQuantity = quantity
        order.orderType = 'MKT'
        order_id = 1
        
        
        
        
        app.placeOrder(order_id, order, Contract)




    

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextOrderId = orderId

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print("OrderStatus. Id: ", orderId, ", Status: ", status, ", Filled: ", filled, ", Remaining: ", remaining, ", LastFillPrice: ", lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print("OpenOrder. ID:", orderId, contract.symbol, contract.secType, "@", contract.exchange, ":", order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print("ExecDetails. ", reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)

    def historicalData(self, reqId, bar):
        print("HistoricalData. ", reqId, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.count, bar.wap, bar.hasGaps)


def calculate_units(balance, symbol):
    # Get the current market price of the symbol
    market_price = blpapi.get_last_price(symbol)

    # Calculate the units to buy based on a percentage of the account balance
    units = int((balance * 0.05) / market_price)
    return units


def select_best_symbol():
    # Get the list of available symbols
    symbols = ["EURUSD", "AAPL", "MSFT", "EURKRW", "USDCHF"]

    # Calculate the supply and demand for each symbol
    supply_demand = {symbol: blpapi.get_supply_demand(symbol) for symbol in symbols}

    # Select the symbol with the highest demand and lowest supply
    selected_symbol = max(supply_demand, key=lambda x: supply_demand[x]["demand"] / supply_demand[x]["supply"])

    return selected_symbol


def main():
    # Connect to IB API
    ib = ib_insync.IB()
    ib.connect('192.168.56.1', 7497, clientId=23467)

    # Get the current account balance
    account = ib.accountSummary()
    balance = float(account[0].value)

    # Select the best symbol to trade based on supply and demand
    symbol = select_best_symbol()

    # Calculate the units to buy based on the account balance and the current market price of the selected symbol
    contract = Contract()
    contract.symbol = EURUSD
    contract.secType = "forex"
    contract.exchange = "MKT"
    contract.currency = "EUR"

    # Get the current market price for the selected symbol
    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(1)
    market_price = ticker.last

    # Calculate the number of units to buy based on the account balance and the market price of the selected symbol
    units = int(balance / market_price)

    # Create the order object
    order = ib_insync.LimitOrder('BUY', units, price=0.0)

    # Place the order
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)





    app.run()

if __name__ == "__main__":
    # Connect to IB API
    ib = ib_insync.ib.IB()
    ib.connect('192.168.56.1', 7497, clientId=23467)

    # Request market data for EURUSD
    contract = ib_insync.Forex('EURUSD', 'MKT', 'EUR')
    ticker = ib.reqMktData(contract)

    while True:
        now = datetime.now()
        if now.weekday() in range(0, 5) and datetime.strptime('09:30', '%H:%M').time() <= now.time() <= datetime.strptime('21:00', '%H:%M').time():
            print("It's time to trade!")
            # Wait for historical data to be received
            time.sleep(10)

            # Get the historical data as a pandas dataframe
            df = pd.DataFrame(app.data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)

            # Generate trading signals based on the historical data using your deep learning model
            signal_data = df[['open', 'high', 'low', 'close', 'volume']]
            signal = my_model(signal_data)

            # Execute trades based on the trading signals generated by your deep learning model using the IB API
            if signal == 'BUY':
                order = Order()
                order.action = "BUY"
                order.orderType = "MKT"
                app.placeOrder(app.nextOrderId, contract, order)
            elif signal == 'SELL':
                order = Order()
                order.action = "SELL"
                order.orderType = "MKT"
                app.placeOrder(app.nextOrderId, contract, order)
        else:
            print("It's not trading hours.")
        # Wait for the next minute to start
        time.sleep(60 - now.second)


        app.run()




# Connect to IB API
ib = ib_insync.ib.IB()
ib.connect('192.168.56.1', 7497, clientId=23467)

# Request market data for EURUSD
contract = ib_insync.Forex('EURUSD', 'MKT', 'EUR')
ticker = ib.reqMktData(contract)
