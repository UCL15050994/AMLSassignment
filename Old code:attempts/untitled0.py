#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 21:59:27 2019

@author: mohamed
"""
import time
import os
import cv2

import datetime as dt
import pandas as pd
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

images = '../dataset'
labels_file = 'attribute_list.csv'


# lists keep the order
f_dataset = []
f_labels = []

# collect labels
df = pd.read_csv(labels_file, skiprows=1, index_col='file_name')
newdf = df[df.columns[0]]

# collect images
for (root, directory, dataset) in os.walk('{0}'.format(images)):

    for file in dataset:
        # read image
        img = cv2.imread('{0}/{1}'.format(images, file), 1)

        # could rescale images here

        f_dataset.append(img)
        f_labels.append(int(file[:-4]))

# only select labels of interest
f_labels = newdf.loc[f_labels]
f_labels = f_labels.values.tolist()


# convert to numpy array to feed to SVM
f_dataset = np.array(f_dataset)
f_labels = np.array(f_labels)


# perform train / validation / test split for x-validation
X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_validation, (X_validation.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# chose number of folds for validation
folds = 5

print('starting cross_validation')
scores = cross_val_score(classifier, X_validation, y_validation, cv=folds)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# fit the best alg to the training data
clf.fit(X_train, y_train)


# predict using test set
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))

# Now predict the value of the test
exp = y_test

print(metrics.classification_report(exp, pred))

cm = metrics.confusion_matrix(exp, pred)
print(cm)
