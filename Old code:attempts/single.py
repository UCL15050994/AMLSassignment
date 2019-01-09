timport time
import os
import cv2

import datetime as dt
import pandas as pd
import numpy as np

import dlib

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from keras.preprocessing import image

import lab2_landmarks as l2


hog = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

images = './new1'
labels_filename = './dataset/data.csv'

f_dataset, f_labels = l2.extract_features_labels()
X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


classifier = svm.SVC(C=1, kernel='linear')

folds = 5

print('starting cross_validation')
scores = cross_val_score(classifier, X_validation, y_validation, cv=folds)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))

expected = y_test

print(metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print(cm)
