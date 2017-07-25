# from https://www.kaggle.com/c/digit-recognizer

import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors, svm, multiclass, cross_validation

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

# 02-1: classifier with svm
clf1 = svm.LinearSVC(C=0.01)
clf1.fit(X_train, y_train)
print('finish train model')

accuracy = clf1.score(X_test, y_test)
print('finish test model')
print(accuracy)
