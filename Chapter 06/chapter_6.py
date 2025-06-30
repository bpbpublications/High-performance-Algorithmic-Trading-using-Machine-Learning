# -*- coding: utf-8 -*-
""

# ----- page 6
from sklearn.preprocessing import KBinsDiscretizer

# transform the dataset with KBinsDiscretizer
enc = KBinsDiscretizer(n_bins=8, encode="onehot")
X = enc.fit_transform(df)
# ------
# ----- page 8
pip install -q pandas_ta

import pandas_ta as ta
# ------
# ----- page 9
# Moving Average (SMA) with a window of 5 periods
df['SMA_5'] = df.ta.sma(close='Close', length=5)

# Exponential Moving Average (EMA)
df['EMA_5'] = df.ta.ema(close='Close', length=5)

# Calculating MACD using pandas_ta
df['MACD'] = df.ta.macd(close='Close')['MACD_12_26_9']

# MACD signal line
df['MACD_Signal'] = df.ta.macd(close='Close')['MACDs_12_26_9']
# ------
# ----- page 10
# RSI
df['RSI'] = df.ta.rsi(close='Close', length=14)

# Stochastic indicator
stoch = df.ta.stoch(high='High', low='Low', close='Close')
df = pd.concat([df, stoch], axis=1)

# CCI
df['CCI'] = df.ta.cci(high='High', low='Low', close='Close')
# ------
# ----- page 11
# Calculating Bollinger Bands using pandas_ta
bollinger = df.ta.bbands(close='Close', length=20, std=2)
df = pd.concat([df, bollinger], axis=1)
# ------
# ----- page 12
# Dollar value of each trade
df['Dollar'] = df['Close'] * df['Volume']

# Set your dollar bar threshold
threshold = df['Dollar'].mean() + df['Dollar'].std()
print(threshold / 1e8)
# ------
# ----- page 13
for index, row in df.iterrows():
    if open_price is None:
        open_price = row['Open']
        index_bar = index

    high_price = max(high_price, row['High'])
    low_price = min(low_price, row['Low'])
    volume += row['Volume']
    cum_dollar += row['Dollar']
# ------
# ----- page 16
from statsmodels.tsa.stattools import adfuller

# Performing Augmented Dickey-Fuller test
for item in lst:
    print(item)
    adf_result = adfuller(df_lag[item].dropna())
    adf_statistic, p_value = adf_result[0], adf_result[1]
    print(np.round(p_value , 4))
    print(f'this series is stationary ? -> {p_value < 0.05}')
# ------
# ----- page 20
from hurst import compute_Hc

def rolling_hurst(series):
    try:
        H, c, data_reg = compute_Hc(series.dropna(), kind='price', simplified=True)
        return H
    except FloatingPointError:
        return np.nan

df['hurst'] = df['Close'].rolling(window=window_size).apply(rolling_hurst, raw=False)
# ------
# ----- page 22
from pyinform.dist import Dist
from pyinform.shannon import entropy

def compute_entropy(df1: pd.DataFrame):
    d = Dist(500)
    for x in df1:
        d.tick(int(x))
    return entropy(d)

window_size = 30

df_combined['Entropy'] = df_combined['Close'].rolling(window=window_size).apply(compute_entropy, raw=False)
# ------
# ----- page 24
n_components_pca = 2
from sklearn.decomposition import PCA

pca = PCA(n_components=n_components_pca)
pca.fit(X_train)
vct = pca.transform(X_train)

plt.scatter(vct[:, 0], vct[:, 1], c=y_test, s=50, cmap='viridis', alpha=0.4)
plt.show()
# ------
# ----- page 26
from umap import umap_ as UMAP

clusterable_embedding = UMAP.UMAP(n_neighbors=30, min_dist=0.0, n_components=8).fit_transform(vct_bin)

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=y_test, s=20, cmap='viridis', alpha=0.4)
plt.legend(['-1','1'])
# ------
# ----- page 27
# Get the leaf indices for each sample
leaf_indices_train = clf.apply(X_train)
leaf_indices_test = clf.apply(X_test)

# merge leaf indexes with other features
X_train = pd.DataFrame(np.hstack([X_train, pd.DataFrame(leaf_indices_train)]))
X_test = pd.DataFrame(np.hstack([X_test, pd.DataFrame(leaf_indices_test)]))

# Train a new model on the combined dataset
clf = RandomForestClassifier(max_depth=15)
clf.fit(X_train, y_train)
# ------
# ----- page 29
from sklearn.feature_selection import mutual_info_classif, RFE, SelectKBest

pct = 5
k_best = int(pct/100 * len(X_train.columns))

sel2 = SelectKBest(mutual_info_classif, k=k_best)
sel2.fit(X_train, y_train)

selected_features_mask = sel2.get_support()
X_train.columns[selected_features_mask]
# ------
# ----- page 30
from sklearn.feature_selection import RFE

sel3 = RFE(clf, n_features_to_select=k_best, step=0.3)
sel3.fit(X_train, y_train)
sel3.get_feature_names_out()[:100]
# ------
# ----- page 31
sel1 = SelectPercentile(mutual_info_classif, percentile=pct, n_neighbors=10)
sel2 = SelectKBest(mutual_info_class