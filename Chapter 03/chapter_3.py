# -*- coding: utf-8 -*-

----- page 4
import seaborn as sns

# Calculate the correlation matrix for the features
correlation_matrix = df_lag.corr()

# Create a mask for the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0, annot=False, fmt=".2f")

plt.title('Correlation Matrix of Features')
plt.show()
--------
----- page 5
# Identify features that are highly correlated
to_drop = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            if colname not in to_drop and not colname.endswith('_ret') :
                to_drop.append(colname)

# Drop the highly correlated features
df_lag.drop(columns=to_drop , inplace =True)
--------
----- page 6
# predict 'lag_predict' ahead in advance
lag_predict = 1

# predict 'lag_predict' ahead in advance
df_lag.dropna(inplace = True)

# classify returns next day (shift = -1) : negative->-1 ; positive-> +1
df_lag['target'] = np.where(df_lag['Close_ret'].shift(-lag_predict) >= 0 , 1 , -1)
--------
----- page 7
from sklearn.model_selection import train_test_split

y = df_lag.pop('target')
X = df_lag.copy()

X_train , X_test , y_train , y_test = train_test_split(X , y , shuffle = False)
--------
from sklearn.preprocessing import StandardScaler

sclr = StandardScaler()
X_train_tr = sclr.fit_transform(X_train)
X_test_tr = sclr.transform(X_test)
--------
----- page 8
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,QuadraticDiscriminantAnalysis as QDA)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

clf1 = LDA()
clf2 = QDA()
clf3 = RandomForestClassifier()
clf4 = GaussianNB()
clf5 = LogisticRegression()
clf7 = DecisionTreeClassifier()

lst_clf= [('lda',clf1), ('qda',clf2),  ('nb',clf4),('lr',clf5) , ('rf',clf3) , ('dt',clf7)]
--------
for tpl in lst_clf:
  clf_base = tpl[1]
  clf_name = tpl[0]
  clf_base.fit(X_train , y_train)
  clf_sc = np.round(clf_base.score(X_test , y_test) , 2)
  print(f"classifier {clf_name} -> accuracy score = {clf_sc}")
--------
----- page 9
sorted = np.argsort(clf_base.feature_importances_)

k_besk = int(40/100 * len(sorted))

X_train.iloc[: , sorted[-k_besk:]].columns

columns_to_drop = X_train.columns[sorted[:-k_besk]]

# retain only k_best features
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)
--------
----- page 12
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
--------
def objective(params):
    classifier_type = params['classifier']
    if classifier_type == 'lr':
        clf = LogisticRegression(**params['lr'])
    elif classifier_type == 'dt':
        clf = DecisionTreeClassifier(**params['dt'])
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params['rf'])
    # Add more elif conditions for other classifiers

    score = -np.mean(cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1, scoring='recall'))
    return {'loss': score, 'status': STATUS_OK}
--------
space = {
    'classifier': hp.choice('classifier', ['lr', 'rf' , 'dt']),
    'lr': {
        'penalty': hp.choice('penalty',['l2']),
        'C': hp.loguniform('C', -5, 5),
    },
    'rf': {
        'n_estimators': hp.choice('n_estimators', range(100, 1000, 50))
    },
    'dt': {
        'max_depth' : hp.choice('max_depth', [5 , 10])
    }
}
--------
# Create a Trials object to keep track of the optimization process
trials = Trials()

# Run the optimization using fmin
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)
--------
----- page 13
from hyperopt import space_eval

print(best)
print(space_eval(space, best)['classifier'])
print(space_eval(space, best)['lr'])
print(space_eval(space, best)['lr']['C'])
print(space_eval(space, best)['rf'])
print(space_eval(space, best)['dt'])
--------
lst_clf_opt = []

for tpl in lst_clf:
  tpl0 = tpl[0]
  tpl1 = tpl[1]

  if tpl0 == 'rf':
    tpl1 = RandomForestClassifier(space_eval(space, best)['rf']['n_estimators'])

  elif tpl0 == 'lr':
    C = space_eval(space, best)['lr']['C']
    penalty = space_eval(space, best)['lr']['penalty']
    tpl1 = LogisticRegression(C = C , penalty = penalty)

  elif tpl0 == 'dt':
    m_depth = space_eval(space , best)['dt']['max_depth']
    tpl1 = DecisionTreeClassifier(max_depth = m_depth)

  lst_clf_opt.append((tpl0 , tpl1))
--------
----- page 14
# blended algorithm: voting classifier
clf_blend = VotingClassifier(voting='hard' , estimators= lst_clf_opt)
clf_blend.fit(X_train, y_train)
clf_blend.score(X_test , y_test)
--------
----- page 16
# broker fee in basis point
fee = 3 /100 /100
stop_loss = -4/100

target_ret = df_lag.loc[y_test.index , 'Close_ret'].shift(-lag_predict)

# Calculate cumulative returns for long trades (+1 predictions)
long_returns = np.where(pred_blend == 1, target_ret , 0)
long_returns = np.where(long_returns < stop_loss, stop_loss , long_returns)
cumulative_long_returns = np.cumsum(long_returns)

# Calculate cumulative returns for short trades (-1 predictions)
short_returns = np.where(pred_blend == -1, -target_ret, 0)
short_returns = np.where(short_returns < stop_loss, stop_loss , short_returns)
cumulative_short_returns = np.cumsum(short_returns)

brk_fee = np.where(np.diff(pred_blend) != 0 , fee , 0)
brk_fee = np.pad(brk_fee , pad_width=(1,0))
--------
----- page 22
!pip install -q ffn

import ffn

symbol = 'MSFT'
prices = ffn.get(symbol, start='2010-01-01')

stats = ffn.calc_stats(pd.Series(np.cumprod(1+long_returns - brk_fee) , index = X_test.index))
print(stats.display())
--------
----- page 24
ax=stats.prices.to_drawdown_series().plot(grid=True,title='maxDrawDown')
--------
----- page 25
!pip install -q empyrical

from empyrical import max_drawdown, annual_return, sharpe_ratio, sortino_ratio

max_drawdown(long_returns - brk_fee)

drawdown = max_drawdown(long_returns - brk_fee)
ann_return = annual_return(long_returns - brk_fee)
sharpe = sharpe_ratio(long_returns - brk_fee)
sortino = sortino_ratio(long_returns - brk_fee)

print(f"Maximum Drawdown: {drawdown:.2%}")
print(f"Annual Return: {ann_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
--------
----- page 26
!pip  install -q quantstats

import quantstats as qs

serie_long_trades = pd.Series(np.cumprod(1+long_returns - brk_fee) , index = X_test.index)
serie_long_trades.index = serie_long_trades.index.tz_convert('America/New_York')

print('sharpe : ' , qs.stats.sharpe(serie_long_trades))
print('sortino : ',qs.stats.sortino(serie_long_trades))
print('mdd : ',qs.stats.max_drawdown(serie_long_trades))
--------
----- page 27
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.getLogger('matplotlib.font_manager').disabled = True

qs.plots.snapshot(serie_long_trades , benchmark = symbol , title='Machine Learning Strategy' , show=True)
--------
----- page 28
qs.reports.html(serie_long_trades , output=symbol +'-rapport.html')
--------