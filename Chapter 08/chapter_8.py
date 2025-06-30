# -*- coding: utf-8 -*-

# ----- page 7
from sklearn.linear_model import LassoCV

y = df.pop('target')
X = df.copy()

X_train , X_test , y_train , y_test = train_test_split(X , y , shuffle = True)

rgr = LassoCV()
rgr.fit(X_train , y_train)
# ------
# ----- page 7
from sklearn.metrics import mean_squared_error

print('RMSE = ', np.sqrt(mean_squared_error(pred , y_test)))
print(np.round(rgr.score(X_test , y_test), 3))
# ------
# ----- page 9
!pip install -q feature-engine
# ------
# ----- page 9
from feature_engine.creation import MathFeatures, RelativeFeatures, CyclicalFeatures

mf = MathFeatures(variables=['Open_ret', 'High_ret', 'Low_ret', 'Close_ret'],
                  func=["sum", "prod", "mean"])
df_feat = mf.fit_transform(df)
df = pd.concat([df, df_feat.iloc[:,-3:]], axis=1)
# ------
# ----- page 10
df.info()
# ------
# ----- page 10
!pip3 install -q autofeat
# ------
# ----- page 10
from autofeat import AutoFeatRegressor

auto_feat = AutoFeatRegressor(verbose=1, feateng_steps=1)
X_train_feat = auto_feat.fit_transform(X_train[['Open_ret','High_ret','Low_ret','Close_ret']], y_train)
X_test_feat = auto_feat.transform(X_test[['Open_ret','High_ret','Low_ret','Close_ret']])
# ------
# ----- page 11
lst = list(set(X_train_feat.columns) - set(X_train.columns))
lst
# ------
# ----- page 12
!pip install -q flaml
# ------
# ----- page 12
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="regression", time_budget=60)
# ------
# ----- page 14
print('RMSE = ', np.sqrt(mean_squared_error(pred , y_test)))
print(automl.score(X_test , y_test))
# ------
# ----- page 15
!pip3 install -q h2o
# ------
# ----- page 15
import h2o
from h2o.automl import H2OAutoML

h2o.init()
# ------
# ----- page 16
df['target'] = df['Close'].shift(-ahead)
df.dropna(inplace=True)

h2o_df = h2o.H2OFrame(df)

train, test = h2o_df.split_frame(ratios=[.8], seed=42)

x = train.columns
y = "target"
x.remove(y)

aml = H2OAutoML(max_models=5, seed=1)
aml.train(x=x, y=y, training_frame=train)
# ------
# ----- page 18
# View the AutoML Leaderboard
aml.leaderboard
# ------
# ----- page 18
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
metalearner = se.metalearner()
# ------
# ----- page 18
print('RMSE = ', np.sqrt(mean_squared_error(pred_df.values, test_df['target'].values)))
# ------