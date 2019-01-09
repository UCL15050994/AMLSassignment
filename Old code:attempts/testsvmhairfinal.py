import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import plot_learning_curve as learning_curve
import lab2_landmarks as l2

images = './new1'
labels_filename = './dataset/data.csv'

#f_dataset, f_labels = l2.extract_features_labels('eyeglasses')

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


f_dataset = np.array(f_dataset)
f_labels = np.array(f_labels)

X_train, X_test, y_train, y_test = train_test_split(f_dataset, f_labels, test_size=0.2, random_state=1)


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 0.01)
#classifier = linear_model.LogisticRegression(solver = 'lbfgs', multi_class = 'ovr', max_iter = 5000, C = 0.01)

learning_curve.plot_learning_curve(classifier, 'Hair Color Learning Curves (SVM)', X_train, y_train, ylim=(0.5, 1.01), cv = 3, n_jobs = -1, train_sizes=np.linspace(.1, 1.0, 5))

