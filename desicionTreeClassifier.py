from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Switching the backend to render
plt.switch_backend('Qt5Agg')

#Datasets
X = np.array([[2, 3], [8, 9], [6, 6]])
Y = np.array([1, 3, 2])

#Training classifier
clf = DecisionTreeClassifier()
clf.fit(X, Y)

#Predicting result
prediction = clf.predict([[3, 3]])
score = accuracy_score([1], prediction)
print('Prediction: {0}\nScore: {1}'.format(prediction, score))

#plotting on graph
for arr in X:
    plt.plot(arr[0], arr[1], 'ro')

plt.plot(3, 3, 'bo')
plt.ylabel('Level of heat')
plt.xlabel('Speed of wind')
plt.show()
