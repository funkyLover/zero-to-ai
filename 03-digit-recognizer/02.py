
# plotting scores to evaluate models (svm.SVC)

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, learning_curve
import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv('train.csv')
print('finish load data')

X = np.array(train_data.drop(['label'], 1)).astype(np.float)
X = preprocessing.scale(X)
y = np.array(train_data['label'])

# 01: select gamma
# param_range = np.logspace(-6, -1, 5)

# train_scores, test_scores = validation_curve(
#     SVC(), X, y, param_name="gamma", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=-1)

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve with SVM")
# plt.xlabel("$\gamma$")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2, color="r")
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="g")
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2, color="g")
# plt.legend(loc="best")
# plt.show()
# end: gamma = 0.0003

# 02: select degree
# param_range = np.arange(1, 11, 1)
# print(param_range)
# train_scores, test_scores = validation_curve(
#     SVC(kernel='poly'), X, y, param_name="degree", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=-1, verbose=1)

# print(train_scores, test_scores)

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve with SVM")
# plt.xlabel("Degree")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2, color="r")
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="g")
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2, color="g")
# plt.legend(loc="best")
# plt.show()
# end: degree = 1

# 01: select C
# c_range = np.logspace(-4, 4, 9)

# train_scores, test_scores = validation_curve(
#     SVC(gamma=0.0003), X, y, param_name="C", param_range=c_range,
#     cv=10, scoring="accuracy", n_jobs=-1)

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve with SVM")
# plt.xlabel("C")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(c_range, train_scores_mean, label="Training score", color="r")
# plt.fill_between(c_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2, color="r")
# plt.semilogx(c_range, test_scores_mean, label="Cross-validation score",
#              color="g")
# plt.fill_between(c_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2, color="g")
# plt.legend(loc="best")
# plt.show()
# end: C = 10.9
