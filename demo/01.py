from sklearn import datasets, svm

iris = datasets.load_iris()
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])
predict = clf.predict(digits.data[-1:])

print(predict)
