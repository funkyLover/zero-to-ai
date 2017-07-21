import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC() # support vector machine
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# save train result
# with open('classifier.pickle', 'wb') as f:
# 	pickle.dump(clf, f)
