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



hog = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
images = './new1'
labels_filename = './dataset/data.csv'

f_dataset, f_labels = l2.extract_features_labels()


X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


num_features = 256 * 256 * 3

gamma_val = 1 / num_features

parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.01, 0.1, 1, 10]}

svr = svm.SVC(gamma = gamma_val)

classifier = GridSearchCV(svr, parameters, cv = 5)

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
