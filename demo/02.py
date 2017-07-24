from sklearn import datasets, svm, multiclass, preprocessing

iris = datasets.load_iris()

clf = svm.SVC()

clf.fit(iris.data, iris.target)
print(clf.predict(iris.data[:3]))

clf.fit(iris.data, iris.target_names[iris.target])
print(clf.predict(iris.data[:3]))

clf.set_params(kernel='rbf').fit(iris.data, iris.target)
print(clf.predict(iris.data[:3]))

clf2 = multiclass.OneVsRestClassifier(estimator=svm.SVC(random_state=0))
clf2.fit(iris.data, iris.target)
print(clf2.predict(iris.data[:3]))

target = preprocessing.LabelBinarizer().fit_transform(iris.target)
clf2.fit(iris.data, target)
print(clf2.predict(iris.data[:3]))
