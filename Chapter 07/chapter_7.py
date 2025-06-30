# -*- coding: utf-8 -*-

# ----- page 10
# Calculating Bollinger Bands using pandas_ta
bollinger = df.ta.bbands(close='Close', length=20, std=2)
df = pd.concat([df, bollinger], axis=1)

# ATR
df['ATR'] = df.ta.atr()
df['TR'] = df.ta.true_range()
# ------
# ----- page 10
mult = 2
df['target'] = np.where((np.abs(df['Close'] - df['BBM_20_2.0']) / df['ATR']) > mult , 1 , -1)
# ------
# ----- page 11
y = df.pop('target')
X = df.copy()

clf = RandomForestClassifier(max_depth = 4)
X_train , X_test , y_train , y_test = train_test_split(X , y)
clf.fit(X_train , y_train)

pred = clf.predict(X_test)
clf.score(X_test , y_test)
# ------
# ----- page 11
from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(y_test , pred))
print()
print(classification_report(y_test, pred))
# ------
# ----- page 14
!pip install -q scikit-optimize

from skopt import BayesSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
# ------
# ----- page 14
space = {
    'learning_rate': Real(0.01, 0.3, prior='uniform'),
    'n_estimators': Integer(100, 1000, prior='uniform'),
    'max_depth': Integer(3, 25, prior='uniform'),
    'gamma': Real(0, 1, prior='uniform'),
    'min_child_weight': Integer(1, 10, prior='uniform'),
    'subsample': Real(0.5, 1, prior='uniform'),
    'colsample_bytree': Real(0.5, 1, prior='uniform'),
}
# ------
# ----- page 14
opt = BayesSearchCV(
    xgb.XGBClassifier(booster='gbtree', device='cuda:0'),
    space,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
# ------
# ----- page 14
opt.fit(X_train, np.where(y_train == -1 , 0 , 1))
# ------
# ----- page 15
from pprint import pprint
print("best params: ")
pprint(opt.best_params_)

best_model = opt.best_estimator_
accuracy = best_model.score(X_test, np.where(y_test == -1 , 0 , 1))
print(f"Accuracy: {accuracy * 100.0}%")
# ------
# ----- page 15
print(confusion_matrix(y_test , pred))
print()
print(classification_report(y_test, pred))
# ------
# ----- page 19
from sklearn.preprocessing import StandardScaler

scl = StandardScaler().set_output(transform="pandas")
X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)
# ------
# ----- page 20
from sklearn.linear_model import Lasso , Ridge
# ------
# ----- page 20
space = {
    'alpha': Real(0.001, 10, prior='log-uniform'),
}

opt = BayesSearchCV(
    Lasso(),
    space,
    n_iter=30,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0
)

opt.fit(X_train, y_train)
print("best params: ")
pprint(opt.best_params_)

rgr = opt.best_estimator_
# ------
# ----- page 21
print(rgr.score(X_test , y_test))
# ------
# ----- page 22
lst_coef = np.argwhere(rgr.coef_ !=0).reshape(1 , -1)[0]
print("columns of interest :",list(X_test.iloc[: , lst_coef ].columns))
# ------
# ----- page 22
print("coef values :")
dict(zip(list(X_test.iloc[: , lst_coef ].columns) , rgr.coef_[lst_coef]))
# ------
# ----- page 22
from sklearn.linear_model import Ridge
# ------
# ----- page 23
from sklearn.kernel_ridge import KernelRidge

space = {
    'alpha': Real(0.001, 10, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf', 'polynomial', 'sigmoid']),
    'gamma': Real(0.001, 10, prior='log-uniform'),
    'degree': Integer(2, 10),
    'coef0': Real(0, 10, prior='uniform'),
}
# ------
# ----- page 24
print("best params: ")
pprint(opt.best_params_)

best_model = opt.best_estimator_
accuracy = best_model.score(X_test,y_test)
print(f"score:", accuracy * 100.0)
# ------
# ----- page 25
!pip install -q scikit-multiflow

from skmultiflow.data import DataStream
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.evaluation import EvaluatePrequential

regressor = AdaptiveRandomForestRegressor(n_estimators=20)
pred_stream = []
# ------
# ----- page 26
stream = DataStream(data=np.column_stack((X_test, y_test)))

while stream.has_more_samples():
    X_stream, y_stream = stream.next_sample()
    y_pred = regressor.predict(X_stream)
    pred_stream.append(y_pred[0])
    regressor = regressor.partial_fit(X_stream, y_stream)
# ------
# ----- page 26
from sklearn.metrics import mean_squared_error

mean_squared_error(pred_stream , y_test)
# ------