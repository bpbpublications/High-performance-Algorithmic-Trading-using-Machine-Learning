# -*- coding: utf-8 -*-


# ----- page 6
pip install -q backtrader[plotting]

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# ------
# ----- page 7
symbol = 'PG'
ticker = yf.Ticker(symbol)
df = ticker.history(period='10d', interval='1d')

# Simulate predictions, favoring a 70% chance of '1' ('buy') and 30% chance of '-1' ('sell')
pred_dumb = np.random.choice([-1 , 1], size = df.shape[0], p = [0.3 , 0.7])

# insert new column in the dataframe to keep track predictions
df['custom'] = pred_dumb
# ------
# ----- page 7
# Define Custom Strategy
class CustomSignalStrategy(bt.Strategy):

    def __init__(self):
        # 'custom' signal <- predictions
        self.custom_signal = self.datas[0].custom

    def log(self, txt, dt=None):
        ''' Logging function'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # logic : pyramid positions
        if self.custom_signal[0] == 1:
            self.buy()
        elif self.custom_signal[0] == -1:
            self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Size: %d' %
                         (order.executed.price, self.getposition().size))
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Size: %d' %
                         (order.executed.price, self.getposition().size))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
# ------
# ----- page 10
# Subclass the PandasData feed to include 'custom' as an additional line
class CustomPandasData(bt.feeds.PandasData):
    lines = ('custom',)
    params = (('custom', -1),)
# ------
# ----- page 10
cash = 100000

def init_cerebro_before_run():
    # Load data into BackTrader using the custom data feed
    data = CustomPandasData(dataname=df)

    # Create a new Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.cheat_on_close = False
    cerebro.adddata(data)
    cerebro.addstrategy(CustomSignalStrategy)
    cerebro.broker.set_cash(cash)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    return cerebro

cerebro = init_cerebro_before_run()
# ------
# ----- page 11
# Run the backtest
results = cerebro.run()

# Print out the final cash
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
# ------
# ----- page 13
class CustomSignalStrategy(bt.Strategy):
    # 5% below the buy price
    params = (('stop_loss_multiplier', 0.95),)

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.stop_loss_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.stop_loss_price = self.buyprice * self.params.stop_loss_multiplier

    def next(self):
        if self.position:
            if self.data.close[0] <= self.stop_loss_price:
                self.sell(exectype=bt.Order.Stop, price=self.stop_loss_price)
# ------
# ----- page 14
class CustomSignalStrategy(bt.Strategy):
    params = (
        ('take_profit_multiplier', 1.05),  # 5% above the buy price
    )

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.take_profit_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.take_profit_price = self.buyprice * self.params.take_profit_multiplier

    def next(self):
        if self.position:
            if self.data.close[0] >= self.take_profit_price:
                self.sell(exectype=bt.Order.Limit, price=self.take_profit_price)
# ------
# ----- page 15
# Set broker parameters for slippage and commission
slippage = 0.005
commission = 5 / 100 / 100

# Set up slippage
cerebro.broker.set_slippage_fixed(fixed = 0.01)
# or :
cerebro.broker.set_slippage_perc(perc = slippage)
cerebro.broker.setcommission(commission = commission)
# ------
# ----- page 16
# Import the analyzers from Backtrader
import backtrader.analyzers as btanalyzers

# Add analyzers
cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trade_analyzer")
cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')

# Run the backtest
results = cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Extract the first strategy from the list of results
strat = results[0]
print('Max Drawdown:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
# ------
# ----- page 17
try:
    trades = strat.analyzers.trade_analyzer.get_analysis()
    hit_rate = trades['won']['total'] / trades['total']['closed']
    avg_win = trades['won']['pnl']['average']
    avg_loss = trades['lost']['pnl']['average']
    long_win = trades['long']['won']
    long_lost = trades['long']['lost']
    short_win = trades['short']['won']
    short_lost = trades['short']['lost']

    print('Total trades :', trades['total']['closed'])
    print('Wins :', trades['won']['total'])
    print('Losses :', trades['lost']['total'])
    print(f'Hit Rate : {(100*hit_rate):.0f}%')
    print(f'Avg Win : {avg_win:.1f}')
    print(f'Avg Loss : {avg_loss : .1f}')
    if avg_loss != 0:
        print(f'Avg win on Lost : {(-avg_win / avg_loss):.1f}')

    print()
    print('LONG TRADES:')
    print('trades long (win)', trades['long']['won'])
    print('trades long (lost)', trades['long']['lost'])
    print(f'pct : {(100*long_win / (long_win + long_lost)):.0f}%')
    print()
    print('SHRT TRADES:')
    print('trades short (win)', trades['short']['won'])
    print('trades short (lost)', trades['short']['lost'])
    if short_win or short_lost != 0:
        print(f'pct : {(100*short_win / (short_win + short_lost)):.0f}%')

except Exception as e:
    print(e)
# ------
# ----- page 20
# Download data for two assets: PG (Procter & Gamble) and KO (Coca-Cola Cie)
data1 = yf.download('PG', period='10d', interval='1d')
data2 = yf.download('KO', period='10d', interval='1d')

# insert new column to keep track predictions
data1['custom'] = pred_dumb
data2['custom'] = pred_dumb
# ------
# ----- page 20
def init_cerebro_before_run(data1 , data2):
    data1 = CustomPandasData(dataname =  data1)
    data2 = CustomPandasData(dataname = data2)
    cerebro = bt.Cerebro()
    cerebro.adddata(data1, name='PG')
    cerebro.adddata(data2, name='KO')
    # rest of code unchanged
# ------
# ----- page 21
def __init__(self):
    self.custom_signal1 = self.datas[0].custom
    self.custom_signal2 = self.datas[1].custom
# ------
def next(self):
    # Trading logic for Asset1
    if self.custom_signal1[0] == 1:
        self.buy(data=self.datas[0])
    elif self.custom_signal1[0] == -1:
        self.sell(data=self.datas[0])

    # Trading logic for Asset2
    if self.custom_signal2[0] == 1:
        self.buy(data=self.datas[1])
    elif self.custom_signal2[0] == -1:
        self.sell(data=self.datas[1])
# ------
def next(self):
    # buy signal
    if self.custom_signal1[0] == 1:
        self.order_target_size(data=self.datas[0], target=1)
        self.order_target_size(data=self.datas[1], target=-1)

    # sell signal
    if self.custom_signal1[0] == -1:
        self.order_target_size(data=self.datas[0], target=-1)
        self.order_target_size(data=self.datas[1], target=1)
# ------
def notify_order(self, order):
    data_name = order.data._name
    # ... (existing logic, now aware of which asset the order is for)
# ------