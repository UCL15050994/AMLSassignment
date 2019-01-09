
import os
import cv2
import pandas as pd
import numpy as np
import dlib
from keras.preprocessing import image
import lab2_landmarks as l2
from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score




f_dataset, f_labels = l2.extract_features_labels('human')


X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))



parameters = {'C':[0.01]}

svr = svm.SVC(kernel = 'linear')

classifier = GridSearchCV(svr, parameters, cv = 3)

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
