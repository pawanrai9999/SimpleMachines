import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#loading datasets
digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clfSVM = AdaBoostClassifier(base_estimator=SVC(C=1, kernel='linear'),
                            n_estimators=50,
                            learning_rate=1,
                            algorithm='SAMME')
clfSVM = clfSVM.fit(X_train, y_train)
predictSVM = clfSVM.predict(X_test)
scoreSVM = accuracy_score(y_test, predictSVM)

clfTree = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, learning_rate=1)
clfTree = clfTree.fit(X_train, y_train)
predictTree = clfTree.predict(X_test)
scoreTree = accuracy_score(y_test, predictTree)

clfGaussianNB = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=50, learning_rate=0.2)
clfGaussianNB = clfGaussianNB.fit(X_train, y_train)
predictNB = clfGaussianNB.predict(X_test)
scoreNB = accuracy_score(y_test, predictNB)

print("Score for SVM: {0}\nScore for DecisionTreeClassifier: {1}\nScore for GaussianNB: {2}".format(scoreSVM,
                                                                                                 scoreTree,
                                                                                                 scoreNB))
