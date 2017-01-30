# This program uses support vector machine of scikit-learn as classifier
# Data used here is provided by scikit-learn
# Features used sepal-width sepal-length petal-width petal-length

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#from sklearn import tree
from sklearn.metrics import accuracy_score

# Switch to your matplotlib backend to show output
plt.switch_backend('Qt5Agg');

# loading iris dataset
irisDataset = datasets.load_iris()
#Shaping data for use with numpy
irisDataset.data.shape, irisDataset.target.shape

print('X: {0} Y: {1}'.format(irisDataset.data[0], irisDataset.target[0]))
print('X: {0} Y: {1}'.format(irisDataset.data[50], irisDataset.target[50]))
print('X: {0} Y: {1}'.format(irisDataset.data[105], irisDataset.target[105]))
#Splliting data
X_train, X_test, Y_train, Y_test = train_test_split(irisDataset.data,
                                                    irisDataset.target,
                                                    test_size=0.3,
                                                    random_state=0)
#Shaping for numpy use
X_train.shape, Y_train.shape
X_test.shape, Y_test.shape

# Initializing classifier
#clf = tree.DecisionTreeClassifier()
clf = SVC(C=1, kernel='linear')
# Training classifier
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
score = accuracy_score(Y_test, prediction)
print('Score: {}'.format(score))
