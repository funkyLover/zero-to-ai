# from https://www.kaggle.com/c/digit-recognizer

import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors, svm, multiclass, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# 00: load train data
train_data = pd.read_csv('train.csv')

print('finish load data')

length = 3000

X = np.array(train_data.drop(['label'], 1)).astype(np.float)[:length]
X = preprocessing.scale(X)
y = np.array(train_data['label'])[:length]
# y = preprocessing.LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

print('finish preprocessing data')

# all image pixel data
images = train_data.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

# 01: visualize the data example randomly
def display_random(max):
	# default count to display
	count = 9
	idx = sample(range(1, max), count)
	display_images = images[idx]
	display_images = [img.reshape(28, 28) for img in display_images]

	for i, v in enumerate(display_images):
		plt.subplot(3, 3, i + 1)
		plt.axis('off')
		plt.imshow(v, cmap=plt.cm.gray_r, interpolation='nearest')

	plt.show()
# display_random(len(images))

C = 0.01
class_weight = 'balanced'
max_iter = 20
solver = 'liblinear'
tol = 1e-8
n_jobs=-1

# 02-1: classifier with SVM
# clf1 = svm.LinearSVC(C=C)
# clf1.fit(X_train, y_train)
# print('finish train model 1')

# accuracy1 = clf1.score(X_test, y_test)
# print('finish test model 1')
# print(accuracy1)

# 02-2: classifier with Logistic Regression
# clf2 = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
# clf2.fit(X_train, y_train)
# print('finish train model 2')

# accuracy2 = clf2.score(X_test, y_test)
# print('finish test model 2')
# print(accuracy2)

# 02-3: classifier with OneVsRestClassifier
clf3 = OneVsRestClassifier(svm.LinearSVC(random_state=0))
clf3.fit(X_train, y_train)
print('finish train model 3')

accuracy3 = clf3.score(X_test, y_test)
print('finish test model 3')
print(accuracy3)
