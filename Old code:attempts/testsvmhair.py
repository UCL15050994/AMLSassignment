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

from sklearn.model_selection import GridSearchCV


images = './new1'
labels_filename = './dataset/data.csv'


# lists keep the order
f_dataset = []
f_labels = []

# collect labels
df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
newdf = df[df.columns[0]]

# collect images
for (root, directory, dataset) in os.walk('{0}'.format(images)):

    for file in dataset:
        # read image
        img = cv2.imread('{0}/{1}'.format(images, file), 1)
        
        resize_img = cv2.resize(img, (128,128))

        # could rescale images here

        f_dataset.append(resize_img)
        f_labels.append(int(file[:-4]))

# only select labels of interest
f_labels = newdf.loc[f_labels]
f_labels = f_labels.values.tolist()


# convert to numpy array to feed to SVM
f_dataset = np.array(f_dataset)
f_labels = np.array(f_labels)


# perform train / validation / test split for x-validation
X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))




parameters = {'kernel':('linear','poly'), 'C':[0.01, 0.1]}

svr = svm.SVC(gamma = 'auto')

classifier = GridSearchCV(svr, parameters, cv = 3, n_jobs = -1)

print("F1")
classifier.fit(X_train, y_train)

print("Grid scores on development set:")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']


for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

# chose number of folds for validation
#folds = 5
#
#print('starting cross_validation')
#scores = cross_val_score(classifier, X_validation, y_validation, cv=folds)
#
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# fit the best alg to the training data



# predict using test set
predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))

# Now predict the value of the test
expected = y_test

print(metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print(cm)

print(classifier.best_params_)
