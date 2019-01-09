"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994
This module was used to in order to train the best model for age identification and
save it into a separate file. The best model was found to be a logistic regression
classifier with C = 0.01 trained with raw RGB values.
"""

#Importing relevant packages/modules
import os
import cv2
import numpy as np
import dataset_loader as ds

from joblib import dump
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#RGB pixel values for age identification
form = 'rgb'
label = 'young'

#Ensuring the correct directory
ds.l2.images_dir = './images1'

#Extracting the facial landmarks and labels for each image
X, y = ds.get_data(form, label)

#Data splitting into training data and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Reshaping the arrays into rows before feeding them into an SVM
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#Specifying the classifier with the right hyperparameters
classifier = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class = 'ovr', C = 0.01)

#Training the model
classifier.fit(X_train, y_train)

#Saving the classifier in the models folder in the directory
dump(classifier, 'models/young.model')



