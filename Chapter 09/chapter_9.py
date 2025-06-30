# -*- coding: utf-8 -*-

# ----- page 5
n = 500
change_point_1 = 250
change_point_2 = 300
stdev = 1

mult = 3
data = np.concatenate([np.random.normal(0, stdev, change_point_1),
                       np.random.normal(0, mult * stdev, change_point_2 - change_point_1),
                       np.random.normal(0, 3 * mult * stdev, n - change_point_2)])
# ------
# ----- page 5
import changefinder

# Initiate changefinder function
cf = changefinder.ChangeFinder(r=0.2)

# compute scores to detect Change Points
scores = [cf.update(p) for p in data]
# ------
# ----- page 6
print('True Change Point at position :', change_point_1)
cpd = np.argsort(scores)[-1]
print('CPD detected at position:', cpd)
# ------
# ----- page 9
asset = 'HG=F'
start_date = '2000-01-01'
end_date = '2024-04-25'

# monthly frequency
df = yf.download(asset, start=start_date, end=end_date , interval='1mo')
# ------
# ----- page 9
import matplotlib.pyplot as plt

plt.plot(df['Close'])
plt.plot(df['Close'].diff())
# ------
# ----- page 10
df.tail(10)
# ------
# ----- page 11
data = df['Close'].diff().dropna().values
# ------
# ----- page 11
import ruptures as rpt

for i, kernel in enumerate(['linear', 'rbf', 'cosine']):
    algo = rpt.KernelCPD(kernel=kernel, min_size=10)
    algo.fit(data)
    result = algo.predict(pen=1e-3)
# ------
# ----- page 14
from statsmodels.stats.diagnostic import acorr_ljungbox

# Segmenting the signal based on detected breakpoints and analyzing residuals
for i, bkpt in enumerate(result[:-1]):
    segment = data[result[i-1]:bkpt]
    # Assuming the model under null hypothesis is a constant (mean of the segment)
    residuals = segment - np.mean(segment)
    # Ljung-Box test for each segment
    df_test_stat = acorr_ljungbox(residuals, lags=[max_lags], return_df=True)
    p_val = df_test_stat.loc[df_test_stat.index[-1], 'lb_pvalue']

    print(f"Segment {i+1} (index pos: {bkpt})| Ljung-Box test p-value: {p_val}")
    # Interpretation
    if p_val < 0.05:
        print("Significant autocorrelation detected in residuals -> poor CPD.\n")
# ------
# ----- page 16
from keras.preprocessing.sequence import pad_sequences

padded_seqs = pad_sequences(seqs, padding='post', dtype='float32')
# ------
# ----- page 16
from sklearn.preprocessing import StandardScaler

# padded_seqs :array of padded sequences
scaler = StandardScaler()
# Standardize the sequences
seqs_scaled = scaler.fit_transform(padded_seqs.reshape(len(padded_seqs), -1))
# ------
# ----- page 17
from umap import UMAP

trsf = UMAP(n_neighbors=2, min_dist=0.1, n_components=2)
seqs_reduced = trsf.fit_transform(seqs_scaled)
# ------
# ----- page 17
# CLUSTERING
n_clusters = 4
clst = KMeans(n_clusters=n_clusters, n_init='auto')
clusters = clst.fit_predict(seqs_reduced)
# ------