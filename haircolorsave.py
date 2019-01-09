import os
import cv2
import numpy as np
import dataset_loader as ds

from joblib import dump
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split


form = 'rgb'
label = 'young'
ds.l2.images_dir = './images1'

X, y = ds.get_data(form, label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#classifier = svm.SVC(gamma = 'auto', kernel = 'poly', C = 0.01)
classifier = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class = 'ovr', C = 0.01)

classifier.fit(X_train, y_train)
dump(classifier, 'models/young.model')



