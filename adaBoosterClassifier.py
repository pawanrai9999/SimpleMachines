import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#loading datasets
digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = AdaBoostClassifier()
clf = clf.fit(X_train, y_train)

predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)

print("Score: {}".format(score))
