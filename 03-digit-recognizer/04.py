
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.svm import SVC
import pandas as pd

train_data = pd.read_csv('train.csv')
print('finish load data')

test_data = pd.read_csv('test.csv')

X = np.array(train_data.drop(['label'], 1)).astype(np.float)
X = preprocessing.scale(X)
y = np.array(train_data['label'])

X_predict = np.array(test_data).astype(np.float)
X_predict = preprocessing.scale(X_predict)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print('finish process data')

clf = SVC(gamma=0.001, C=12)
# clf.fit(X_train, y_train)
# print('finish train model')

# accuracy = clf.score(X_test, y_test)
# print('finish test model')
# print(accuracy)

clf.fit(X, y)
y_predict = clf.predict(X_predict)

test_predict = pd.DataFrame(y_predict, columns=['Label'])
test_predict.index = test_predict.index + 1
# test_predict.to_csv('01-svc-gamma=001-C=1.csv')
test_predict.to_csv('02-svc-gamma=001-C=12.csv')
