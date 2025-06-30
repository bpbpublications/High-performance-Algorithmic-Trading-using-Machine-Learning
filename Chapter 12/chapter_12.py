# -*- coding: utf-8 -*-

# ----- page 5
# Calculate Z-scores for trading volume
window_size = 20
stock_data['Z-Score'] = (stock_data['volume'] - stock_data['volume'].rolling(window_size).mean()) / stock_data['volume'].rolling(window_size).std()
# ------
# ----- page 5
# Step 2: Use trading volume for anomaly detection
volume_data = stock_data['volume']

# Step 3: Calculate Q1, Q3, and IQR
Q1 = np.percentile(volume_data, 25)
Q3 = np.percentile(volume_data, 75)
IQR = Q3 - Q1

# Step 4: Define the lower and upper bounds for anomalies
lower_bound = Q1 - 0.3 * IQR
upper_bound = Q3 + 1.5 * IQR
# ------
# ----- page 5
# Step 5: Identify anomalies (values below lower_bound OR above upper_bound)
stock_data['Anomaly'] = (volume_data < lower_bound) | (volume_data > upper_bound)
# ------
# ----- page 6
# Set a Z-score threshold for anomalies
threshold = 1.96
stock_data['Anomaly'] = stock_data['Z-Score'].apply(lambda x: True if abs(x) > threshold else False)

# visualize Anomalies
stock_data[stock_data['Anomaly'] == True]
# ------
# ----- page 10
# Calculate rolling averages and volume ratios
stock_data['Rolling Volume Mean'] = stock_data['volume'].rolling(window=10).mean()
stock_data['Volume Ratio'] = stock_data['volume'] / stock_data['Rolling Volume Mean']

# Drop NaN values (caused by rolling calculations)
stock_data.dropna(inplace=True)

# Select features for Isolation Forest
features = stock_data[['volume', 'Rolling Volume Mean', 'Volume Ratio']].values
# ------
# ----- page 10
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
stock_data['Anomaly Score'] = isolation_forest.fit_predict(features)
# ------
# ----- page 12
!pip install -q hurst
from hurst import compute_Hc

window_size = 100
stock_data['hurst_price'] = stock_data['adjClose'].rolling(window=window_size).apply(
    lambda x: compute_Hc(x, simplified=True)[0] if len(x) >= 100 else np.nan, raw=False)
# ------
# ----- page 13
for i in range(1 , 4):
    stock_data[f'close_{i}'] = stock_data['adjClose'].shift(i)
    stock_data[f'close_pct{i}'] = np.round(stock_data['adjClose'].pct_change().shift(i) , 3)
    stock_data[f'zscore_{i}'] = np.round(stock_data['Z-Score'].shift(i) , 2)
    stock_data[f'volume_pct{i}'] = np.round(stock_data['volume'].pct_change().shift(i) , 3)
    stock_data[f'hurst_{i}'] = np.round(stock_data['hurst_price'].shift(i),2)
# ------
# ----- page 13
lst_col = ['open','high',  'low', 'close', 'adjClose', 'volume', 'Z-Score',
'hurst_price', 'close_1', 'close_pct1', 'zscore_1', 'volume_pct1', 'hurst_1',
'close_2', 'close_pct2', 'zscore_2','volume_pct2', 'hurst_2',
'close_3', 'close_pct3', 'zscore_3','volume_pct3', 'hurst_3']
# ------
# ----- page 14
from sklearn.cluster import KMeans

clst = KMeans(n_init='auto', n_clusters=8)
clst.fit(stock_data[lst_col])
np.unique(clst.labels_ , return_counts = True)
# ------
# ----- page 15
!pip install -q umap-learn
from umap import UMAP

trsf = UMAP(n_neighbors = 7, min_dist = 0.1, n_components = 2)
seqs_reduced = trsf.fit_transform(stock_data[lst_col])
centroids_reduced = trsf.transform(clst.cluster_centers_)
# ------
# ----- page 15
factor = 1.3
distances = np.array([np.linalg.norm(point - centroids_reduced[label])
                      for point, label in zip(seqs_reduced, clst.labels_)])
# ------
# ----- page 15
thresholds = {}
for label in np.unique(clst.labels_):
    cluster_distances = distances[clst.labels_ == label]
    thresholds[label] = np.mean(cluster_distances) + factor * np.std(cluster_distances)
# ------
# ----- page 15
anomalies = np.array([distance > thresholds[label] and (label in top_clusters)
                      for distance, label in zip(distances, clst.labels_)])
print(f"Anomalies: {100*np.sum(anomalies)/len(seqs_reduced):.2f}%")
# ------
# ----- page 18
def detect_trend(prices, window=20):
    ma = prices.rolling(window=window).mean()
    trend = np.where(ma.diff() > 0, 1, -1)
    return trend

trend = detect_trend(stock_data['adjClose'], window)
# ------
# ----- page 18
for i in range(window, len(stock_data)):
    if anomalies[i] and clst_labels[i] in top_clusters:
        positions[i] = -trend[i]
# ------
# ----- page 18
window = 120
stop_loss = -2/100

positions, returns = trading_strategy(stock_data, anomalies, clst.labels_,
                                      list(top_clusters), window)

returns = np.where(returns < stop_loss , stop_loss , returns)
# ------
# ----- page 19
cumulative_returns = np.cumsum(returns)

print(f"Total Return: {cumulative_returns[-1]:.2%}")
print(f"Annualized Return: {(1 + cumulative_returns[-1]) ** (252/len(returns)) - 1:.2%}")
print(f"Sharpe Ratio: {np.mean(returns) / np.std(returns) * np.sqrt(252):.2f}")
# ------
# ----- page 21
lag = 1
asset_ret = stock_data["close"].pct_change().shift(-lag)

asset1_bins_pos = asset_ret > 0
asset1_bins_neg = asset_ret < 0

is_cluster_A = pd.Series(clst.labels_ == top_clusters[0])
is_anomaly = pd.Series(anomalies)
is_hust_trend = stock_data['hurst_price'] > 0.5
is_hurst_notrend = stock_data['hurst_price'] < 0.5
# ------
# ----- page 22
dataset = pd.concat([asset1_bins_pos, asset1_bins_neg, is_cluster_A, is_cluster_B,
                     is_cluster_C, is_cluster_D, is_anomaly, is_hust_trend,
                     is_hurst_notrend], axis=1)
# ------
# ----- page 23
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(dataset, min_support = 0.05, use_colnames = True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
# ------
# ----- page 23
asset_ret_rules = rules[rules['consequents'].apply(lambda x: 'asset_ret_pos' in x
                                                   or 'asset_ret_neg' in x)]
# ------
# ----- page 24
asset_ret_rules = asset_ret_rules.sort_values(['lift'], ascending=False)
asset_ret_rules
# ------
# ----- page 25
for i in range(1, len(data)):
    if cluster_labels[i-1] == top_clusters[0] or\
       cluster_labels[i-1] == top_clusters[2] or\
       cluster_labels[i-1] == top_clusters[3]:
        positions[i] = 1
    else:
        positions[i] = 0

    portfolio_value[i] = portfolio_value[i-1] * (1 + positions[i] * returns[i])

sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
# ------