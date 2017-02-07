import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
#np.array([[2, 3], [5, 8], [3, 8], [6, 6], [5, 2]])
y = digits.target
#np.array([0, 0, 0, 1, 1])

#Splitting datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Initializing classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X_train, y_train)

predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)

print("Score: {}".format(score))
