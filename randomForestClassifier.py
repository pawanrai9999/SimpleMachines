import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

#Splitting datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4
)

#Initializing classifier
clf = RandomForestClassifier(n_estimators=20)
clf = clf.fit(X_train, y_train)

#Predicting data
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)

print("Score: {}".format(score))
