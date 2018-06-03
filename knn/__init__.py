import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from knn import KNNClassifier
from module_selection import train_test_split

iris = datasets.load_iris()
#print(iris.keys())

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)


my_knn_clf = KNNClassifier(k = 3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)


#accuracy
accuracy = sum(y_predict == y_test) / len(y_test)
print(accuracy)


# using sklean KNeighborsClassifier and model_selection
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
#print(iris.keys())

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)


my_knn_clf = KNeighborsClassifier(k = 3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)


#accuracy
accuracy = sum(y_predict == y_test) / len(y_test)
print(accuracy)

"""

