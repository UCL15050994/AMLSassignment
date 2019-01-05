import os
import cv2
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import lab2_landmarks as l2

images = './new1'
labels_filename = './dataset/data.csv'

f_dataset, f_labels = l2.extract_features_labels()
X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


parameters = {'C':[0.01, 0.1, 1]}

logreg = linear_model.LogisticRegression(multi_class = 'ovr', solver = 'lbfgs', max_iter = 5000)

classifier = GridSearchCV(logreg, parameters, cv = 3, n_jobs = -1)

print("F1")
classifier.fit(X_train, y_train)

print("Grid scores on development set:")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']


for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))

expected = y_test

print(metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print(cm)

print(classifier.best_params_)
