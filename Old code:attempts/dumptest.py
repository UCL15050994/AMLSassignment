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

f_dataset = []
f_labels = []

df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
newdf = df[df.columns[0]]

for (root, directory, dataset) in os.walk('{0}'.format(images)):

    for file in dataset:

        img = cv2.imread('{0}/{1}'.format(images, file), 1)
        resize_img = cv2.resize(img, (128,128))
        resize_img = resize_img[0:63,0:127]
        f_dataset.append(resize_img)
        f_labels.append(int(file[:-4]))


f_labels = newdf.loc[f_labels]
f_labels = f_labels.values.tolist()

f_labels1 = []
f_dataset1 = []

for i in range(len(f_labels)):
    if f_labels[i] != -1:
        f_labels1.append(f_labels[i])
        f_dataset1.append(f_dataset[i])
        


f_dataset = np.array(f_dataset1)
f_labels = np.array(f_labels1)


f_dataset = np.array(f_dataset)
f_labels = np.array(f_labels)


X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


classifier = svm.SVC(gamma = 'auto', kernel = 'poly', C = 0.01)
#classifier = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class = 'ovr', C = 0.01)

print('test')
classifier.fit(X_train, y_train)
dump(classifier, 'models/hair_color_final.model')






#print("Grid scores on development set:")
#means = classifier.cv_results_['mean_test_score']
#stds = classifier.cv_results_['std_test_score']
#
#
#for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean, std * 2, params))
#
#predictions = classifier.predict(X_test)
#print(accuracy_score(y_test, predictions))
#
#expected = y_test
#
#print(metrics.classification_report(expected, predictions))
#
#cm = metrics.confusion_matrix(expected, predictions)
#print(cm)
#
#print(classifier.best_params_)

