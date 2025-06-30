# -*- coding: utf-8 -*-

import numpy as np

try :
import yfinance as yf
except ModuleNotfoundError as e:
!pip install -q yfinance
import yfinance as yf

--------------------------------------------------------------------------------

# Define the stock tickers
tickers = ['AAP', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'GS', 'HD', \
'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE'\
'PFE', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT', 'XOM', 'MMM']

# Download historical price data from Yahoo Finance
data = yf.download(tickers, start='1990-01-01', end='2023-07-07' , interval ='1d')
# Calculate the percentage change in price over a recent period (e.g., last two weeks)
ret = data['Adj Close'].pct_change(periods=10)
# Drop `Nan` values
ret.dropna(inplace=True)
# Rank the stocks based on their percentage change
ranked_ret = ret.rank(axis=1, ascending=False)
# Select the top-performing and bottom-performing deciles of stocks
# x% first
top_decile = ranked_ret[ranked_ret <= len(tickers) * 0.2]
# worst performers
bottom_decile = ranked_ret[ranked_ret > len(tickers) * 0.9]
bottom_decile.dropna()
# Open new positions based on the strategy
long_positions = top_decile.loc[date].dropna()

--------------------------------------------------------------------------------

short_positions = bottom_decile.loc[date].dropna()
if stock not in top_decile.loc[date].dropna().index and stock not in bottom_decile.loc[date].dropna().index:
# Initial portfolio value
portfolio_value = 10000
total_trades = 0
total_profit_loss = 0
lst_pnl = []
stp_loss = -2/100 * portfolio_value

# Define transaction cost per share, x$  per share ($0.005 : IB Broker)
transaction_cost = 0.01

--------------------------------------------------------------------------------

try :
import yfinance as yf
except ModuleNotfoundError as e:
!pip install -q yfinance

import yfinance as yf
companies = ['FORD','GM']
tickers = yf.Tickers(companies)
tickers_hist = tickers.history(period='900d',interval='1d',)
s1 = tickers_hist['Close']['FORD']
s2 = tickers_hist['Close']['GM']
import numpy as np
tmp = s2 - s1
spread = (tmp - np.mean(tmp)) / np.std(tmp)

--------------------------------------------------------------------------------
quantile = [0.05 , .25 , .50 , .75 , 0.95]
qtl = np.quantile(spread , quantile)
qtl

--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import yfinance as yf
yf.pdr_override()
symbol = 'GM'
ticker = yf.Ticker(symbol)
#Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
df = ticker.history(period='1000d', interval='1d')
# Compute moving averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()
df['MA_cross'] = df['SMA50'] - df['SMA200']
# Create target variable
df['target'] = pd.DataFrame(np.zeros(len(df)))
df['target'] = np.where(df['Close'].pct_change() > 0 , 1 , -1)
df['target'] = df['target'].shift(-1)

--------------------------------------------------------------------------------
# Remove NaNs
df.dropna(inplace=True)
# Create features
X = df[['Open','High','Low','Close','SMA50', 'SMA200' , 'MA_cross']]

--------------------------------------------------------------------------------
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df['target'] , shuffle=False)

# Create and train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

--------------------------------------------------------------------------------
from sklearn.metrics import classification_report
print (classification_report(clf.predict(X_test) , y_test))

--------------------------------------------------------------------------------

# cross-over rules : +1 MA(50 days) > MA(200) ; -1 : MA(50 days) < MA(200)
ma_cross = np.where(X_test['MA_cross']>0,1,-1)
print (classification_report(ma_cross , y_test))

--------------------------------------------------------------------------------

# agregated PnL for each specific period :
ar_pnl = pd.DataFrame(np.zeros(len(ranked_ret.index)) , index = ranked_ret.index , columns=["agg_pnl"] )

--------------------------------------------------------------------------------
# Calculate transaction cost
# transaction cost in basis point:
# transaction_cost_amount = (open_price + close_price) * quantity * transaction_cost
# transaction cost per share (broker fee : minimum 1$):

transaction_cost_amount = max(quantity * transaction_cost , 1)
profit_loss = (close_price - open_price) * quantity - transaction_cost_amount
# apply stop loss
profit_loss = np.max([profit_loss , stp_loss])
lst_pnl.append(profit_loss)
total_profit_loss = total_profit_loss + profit_loss
ar_pnl['agg_pnl'].loc[date] = ar_pnl['agg_pnl'].loc[date] + profit_loss

--------------------------------------------------------------------------------
# features : we merge the ranked returns and array of returns
X = pd.concat([ret , ranked_ret] , axis =1)
# target : agregated PnL for each specific period -> ar_pnl
y = np.sign(ar_pnl.shift(-1))
# binary sign is either 1 or -1 not 0
y[y == 0] = -1
# last period, zero to clean missing value
y.iloc[-1] = -1
y = np.ravel(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.025,shuffle = False)
clf = RandomForestClassifier()
clf.fit(X_train , y_train)

--------------------------------------------------------------------------------

# open a trade if the ML model allows it ....
if clf.predict_proba(pd.DataFrame(X.loc[date].values.reshape(1,-1) , columns=X.columns))[1] > 0.5:
print("ML OK for this trade")