"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994

This module was used to find the optimal hyperparameters for the SVM 
classifiers used for all the tasks using GridSearchCV. 3-fold cross validation
is used and relevant metrics are obtained.
"""

#Importing relevant packages/modules
import numpy as np
import dataset_loader as ds

from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#Change the value of form to 'rgb' to obtain raw RGB values for the images
#label can be changed to 'hair_color', 'eyeglasses', 'smiling', or 'young'
#as appropriate
form = 'landmarks'
label = 'smiling'

#Obtaining X (data) and y (labels) using the get_data function from the
#dataset_loader.py module
ds.l2.images_dir = './images1'
X, y = ds.get_data(form, label)

#Performing train-test split using the train_test_split function from sklearn
#80% of the data is taken to be training data and 20% of the data is taken
#to be test data as indicated by test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Preprocessing the data by reshaping it as appropriate into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#Selecting the number of folds for cross validation
#3-fold cross validation was used for all the results
folds = 3

#Using an SVM from sklearn
#gamma is set to 'auto' as a fixed hyperparameter, which means that it is  
#always the inverse of the number of features
svr = svm.SVC(gamma = 'auto')

#Selecting the hyperparameters to be investigated
parameters = {'kernel' : ('linear', 'poly'), 'C':[0.01, 0.1]}

#Using GridSearchCV to find the optimal hyperparameters
#n_jobs is set to -1 in order to use all the processors and speed up
#search and cross validation
classifier = GridSearchCV(svr, parameters, cv = folds, n_jobs = -1)

#Fitting the data to the best model found using GridSearchCV
classifier.fit(X_train, y_train)

#Printing out the mean cross validation scores and standard deviations
#for all combinations of hyperparameters tested using GridSearchCV
print('Grid scores:')
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

#Printing out the optimal hyperparameters found using GridSearchCV
print('')
print('The best hyperparameters are:')
print(classifier.best_params_)

#Printing out the average inference accuracy score on the test set
predictions = classifier.predict(X_test)
print('')
print('Accuracy score on test set:')
print('%0.3f' % (accuracy_score(y_test, predictions)))

#Printing out the classification report (containing precision, recall, and
#f1 scores) and the confusion matrix
expected = y_test

print('')
print('Classification report:')
print(metrics.classification_report(expected, predictions))

print('')
print('The confusion matrix is:')
cm = metrics.confusion_matrix(expected, predictions)
print(cm)
