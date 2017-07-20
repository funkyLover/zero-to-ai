import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
predict_date = df.iloc[-forecast_out:]
new_df = df[:]
df['label'] = df[forecast_col]
X = np.array(df.drop(['label'], 1))
X_scale = preprocessing.scale(X)
X = X_scale[:]
X_lately = X_scale[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(accuracy)

df['Forecast'] = np.nan
print(forecast_set)
idx = 0

for val in forecast_set:
    date = new_df.iloc[-forecast_out + idx].name
    idx += 1
    df.loc[date] = [np.nan for _ in range(len(df.columns) - 1)] + [val]

print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
new_df['Adj. Close'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
