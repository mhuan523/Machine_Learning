from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

grid_param =[
    {
        'weights': ['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [p for p in range(1, 6)]

    }
]

if __name__ == '__main__':

    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, grid_param, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    score = grid_search.best_score_

    print(score)


