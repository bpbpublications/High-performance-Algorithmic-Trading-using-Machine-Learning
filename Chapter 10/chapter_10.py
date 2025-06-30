# -*- coding: utf-8 -*-

# ----- page 6
# Function to create a Distance Matrix and a Recurrence Plot
def dist_matrix_rec_plot(data, threshold=0.1):
    # Calculate the distance matrix
    dist_matrix = np.abs(data[:, None] - data[None, :])
    # Create the recurrence plot using the threshold
    rp = dist_matrix <= threshold
    return rp , dist_matrix
# ------
# ----- page 8
def piecewise_function(t):
    return np.where(t <= 3, t, -t + 6)
# ------
# ----- page 21
def create_distance_matrix(series, window_size):
    matrices = []
    for i in range(len(series) - window_size + 1):
        window = series[i:i+window_size]
        distance_matrix = np.abs(window[:, None] - window[None, :])
        matrices.append(distance_matrix)
    return matrices
# ------
# ----- page 22
# Create labels
labels = np.sign(np.diff(close_prices[window_size - 1:]))
labels[labels == 0] = -1
# ------
# ----- page 23
# Split into train and test sets
train_matrices, test_matrices, train_labels, test_labels = train_test_split(
    distance_matrices[:-1], labels, test_size=0.2, shuffle=False)

# Flatten matrices into vectors
train_matrices_flat = np.array([matrix.flatten() for matrix in train_matrices])
test_matrices_flat = np.array([matrix.flatten() for matrix in test_matrices])
# ------
# ----- page 24
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter = 1000, solver = 'liblinear')
clf.fit(train_matrices_flat , train_labels)
test_predictions = clf.predict(test_matrices_flat)
# ------
# ----- page 25
# Print detailed classification report
print(classification_report(test_labels, test_predictions))
# ------
# ----- page 27
!pip -q install fastdtw

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Compute DTW distance
distance, path = fastdtw(time_series_1, time_series_2, dist=lambda x, y: euclidean(x.flatten(), y.flatten()))

for (map1, map2) in path:
    plt.arrow(map1, time_series_1[map1], map2 - map1, time_series_2[map2] - time_series_1[map1],
              color='black', length_includes_head=True, head_width=0.05)
# ------
# ----- page 29
!pip -q install stumpy

import stumpy

# Compute matrix profile
m = 640
ts = df.iloc[-1500:]['Close']
mp = stumpy.stump(ts, m)
plt.plot(mp[:, 0])
# ------
# ----- page 30
motif_idx_1 = np.argsort(mp[:, 0])[-1]
print(f"The discord is located at index {motif_idx_1}")
discord_idx = mp[motif_idx_1, 1]

motif_idx_2 = np.argsort(mp[:, 0])[0]
nearest_idx = mp[motif_idx_2, 1]
print(f"The motif is located at index {nearest_idx}")
# ------
df_norm = stumpy.core.z_norm(df['Close'].values)
# ------
# ----- page 30 (continued)
# Z-score normalization
window = 20
df['rolling_mean'] = df['Close'].rolling(window=window).mean()
df['rolling_std'] = df['Close'].rolling(window=window).std()

# Compute z-score
df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
# ------
length_hint = 80
idx_begin_query = df.shape[0] - length_hint

T_df = df.iloc[:idx_begin_query]['z_score']
Q_df = df.iloc[idx_begin_query:]['z_score']

print(f'query to search begin at position : ', df.iloc[idx_begin_query].name)
# ------
distance_profile = stumpy.core.mass(Q_df, T_df)
# ------