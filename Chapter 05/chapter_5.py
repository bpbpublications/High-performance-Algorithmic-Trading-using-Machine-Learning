# -*- coding: utf-8 -*-

# ----- page 28
from sklearn.metrics import confusion_matrix , accuracy_score
import numpy as np

# Classification predictions and actual labels:
y_true = [1]*30 + [0]*10 + [1]*40 + [0]*25  # 30 TP, 40 FN, 10 FP, 25 TN
y_pred = [1]*30 + [1]*10 + [0]*40 + [0]*25  # Predicted labels

# Create the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", np.round(accuracy,2))
# ------
# ----- page 30
# Calculate Precision:
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print("Precision:", np.round(precision,2))
# ------
# ----- page 31
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print("Recall:", np.round(recall,2))
# ------
# ----- page 32
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print("F1-Score:", np.round(f1,2))
# ------
# ----- page 33
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Example: MSE can be calculated using
mean_squared_error(y_true, y_pred)
# ------
# ----- page 34
from sklearn.metrics import mean_squared_error

# Example: RMSE can be calculated using:
sqrt(mean_squared_error(y_true, y_pred))
# ------
# ----- page 35
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_true, y_pred)
# ------