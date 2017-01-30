# This program uses support vector machine of scikit-learn as classifier
# Data used here is provided by scikit-learn
# Features used sepal-width sepal-length petal-width petal-length

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Switch to your matplotlib backend to show output
plt.switch_backend('Qt5Agg');

# loading iris dataset
irisDataset = datasets.load_iris()
#Shaping data for use with numpy
irisDataset.data.shape, irisDataset.target.shape
#Splliting data
X_train, X_test, Y_train, Y_test = train_test_split(irisDataset.data,
                                                    irisDataset.target,
                                                    test_size=0.3,
                                                    random_state=0)
#Shaping for numpy use
X_train.shape, Y_train.shape
X_test.shape, Y_test.shape


# Initializing classifier
clf = SVC(kernel='linear')
# Training classifier
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print('Score: {}'.format(score))
