import time
import os
import cv2

import datetime as dt
import pandas as pd
import numpy as np

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

images = './new1'
labels_filename = './dataset/data.csv'
f_dataset = []
f_labels = []
df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
newdf = df[df.columns[3]]

for (root, directory, dataset) in os.walk('{0}'.format(images)):

    for file in dataset:

        img = cv2.imread('{0}/{1}'.format(images, file), 1)
        
        resize_img = cv2.resize(img, (128,128))
        f_dataset.append(resize_img)
        f_labels.append(int(file[:-4]))

f_labels = newdf.loc[f_labels]
f_labels = f_labels.values.tolist()

f_dataset = np.array(f_dataset)
f_labels = np.array(f_labels)
X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


classifier = svm.SVC(C=1, kernel='poly', max_iter = 500)
folds = 3
scores = cross_val_score(classifier, X_train, y_train, cv=folds)

print("Cross Validation Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print("Accuracy Score: %0.2f" % (accuracy_score(y_test, predictions)))

expected = y_test

print(metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print(cm)