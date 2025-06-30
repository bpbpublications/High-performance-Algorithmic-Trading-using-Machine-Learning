# -*- coding: utf-8 -*-
# ----- page 4
try:  
    import wbdata  
except ModuleNotFoundError as e:  
    print(e)  
    !pip install wbdata  

import wbdata

# Set the data parameters  
indicator = {'FP.CPI.TOTL.ZG': 'Inflation'}  
country = {'USA': 'United States'}  

# Retrieve the data  
data = wbdata.get_dataframe(indicator, country=country)  

# chronological order  
data = data.reindex(index=data.index[::-1])  
data.info()
# ------
# ----- page 4
import matplotlib.pyplot as plt  

data.plot(figsize=(5,3))  
plt.grid()
# ------
# ----- page 5
# Set the data parameters  
indicator = {'NY.GDP.MKTP.CD': 'GDP'}  
country = {'CHN': 'China'}  

# ...rest of code unchanged except:
data.plot(figsize=(5,3), logy=True)
# ------
# ----- page 6
!pip install fredapi

import pandas as pd  
import fredapi

api_key = 'YOUR_API_KEY'  
fred = fredapi.Fred(api_key=api_key)  

interest_rate_data = fred.get_series('GS10')  
print(interest_rate_data)
# ------
# ----- page 7
!pip install -q quandl  
import quandl  

quandl.get("FRED/GDP")
# ------
# ----- page 9
import requests  

headers = {  
    'Content-Type': 'application/json'  
}  

requestResponse = requests.get(
    "https://api.tiingo.com/tiingo/fundamentals/msft/statements?token=" + API_KEY,
    headers=headers
)  
requestResponse.json()
# ------
# ----- page 11
url = 'https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey='  
import requests  

headers = {  
    'Content-Type': 'application/json'  
}  

requestResponse = requests.get(url + API_KEY, headers=headers)  
requestResponse.json()
# ------
# ----- page 14
df = pd.read_csv('2014_Financial_Data.csv', index_col=0)  
df.info()  
df.dtypes
# ------
# ----- page 15
df['Sector'].value_counts()
# ------
# ----- page 16
top_q = df.quantile(numeric_only=True, q=0.95)  
outliers = df > top_q  
df[outliers] = np.nan  
df.fillna(top_q , inplace = True)

top_q = df.quantile(numeric_only=True, q=0.05)  
outliers = df < top_q  
df[outliers] = np.nan  
df.fillna(top_q , inplace = True)
# ------
# ----- page 16
df.fillna(df.groupby('Sector').transform('median') , inplace = True)
# ------
# ----- page 17
y_price_var = df.pop('2015 PRICE VAR [%]')

from sklearn.preprocessing import LabelEncoder  
lbl = LabelEncoder()  

lst_sector = list(np.unique(df['Sector']))  
lbl.fit(lst_sector)  
df['Sector'] = lbl.transform(df['Sector'])  
dict(zip(lst_sector , lbl.transform(lst_sector)))
# ------
# ----- page 18
y = df.pop('Class')

from sklearn.model_selection import train_test_split  
X_train , X_test , y_train , y_test = train_test_split(df , y , shuffle=False)
# ------
# ----- page 19
from sklearn.ensemble import RandomForestClassifier  

clf = RandomForestClassifier()  
clf.fit(X_train , y_train)
# ------
# ----- page 20
clf.score(X_test , y_test)

from sklearn.metrics import classification_report , confusion_matrix  
pred = clf.predict(X_test)  
print(classification_report(y_test , pred))
# ------
# ----- page 21
tn, fp, fn, tp = confusion_matrix(y_test , pred).ravel()  
print(f"FALSE PREDICTIONS : {fp} negative returns wrongly detected as positive return and {fn} positive returns not detected")  
print(f"CORRECT PREDICTIONS : {tp} positive returns correctly detected as positive return and {tn} negative returns detected")  
confusion_matrix(y_test , pred)
# ------
# ----- page 22
from sklearn.model_selection import GridSearchCV  

grid = {  
    'max_depth': [None, 5, 15, 30],  
    'min_samples_split': [5, 10, 15],  
    'min_samples_leaf': [1, 2, 4]  
}

clf = GridSearchCV(  
    estimator=RandomForestClassifier(),  
    cv=2,  
    param_grid=grid,  
    n_jobs=-1,  
    scoring='f1',  
    verbose=2  
)

clf.best_params_  
clf.best_estimator_
# ------
# ----- page 24
import yfinance as yf  
yf.pdr_override()  

symbol = 'NVDA'  
ticker = yf.Ticker(symbol)  

df = ticker.history(period='1000d', interval='1d')  
df.info()
# ------
