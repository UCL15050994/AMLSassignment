import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from joblib import dump
import lab2_landmarks as l2

images = './new1'
labels_filename = './dataset/data.csv'

f_dataset, f_labels = l2.extract_features_labels('smiling')


X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


classifier = svm.SVC(gamma = 'auto', kernel = 'linear', C = 0.01)
#classifier = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class = 'ovr', C = 0.01)

print('test')
classifier.fit(X_train, y_train)
dump(classifier, 'models/smiling.model')
