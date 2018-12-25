#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 02:45:44 2018

@author: mohamed
"""

import pandas as pd
import lab2_landmarks as l2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import glob
#from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
#import facenet
import dlib


def get_data():
    X, y = l2.extract_features_labels()
    X_train = X[:int(0.8*len(X))]
    y_train = y[:int(0.8*len(X))]
    X_test = X[int(0.8*len(X)):]
    y_test = y[int(0.8*len(X)):]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    
    return X_train, y_train, X_test, y_test

def train_SVM_multi(training_images, training_labels, test_images, test_labels):
    training_images, training_labels = make_classification(n_features=136, random_state=0)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(training_images, training_labels)
    ans = clf.score(test_images, test_labels)
    return ans
    

def train_SVM(training_images, training_labels, test_images, test_labels):
    clf = svm.SVC(gamma='scale')
    clf.fit(training_images, training_labels)
    ans = clf.score(test_images, test_labels)
    return ans


def main():    
    (X_train, y_train, X_test, y_test) = get_data()
    print("Length of X_train", len(X_train))
    print("Length of y_train", len(y_train))
    print("Length of X_test", len(X_test))
    print("Length of y_test", len(y_test))
    #print(train_SVM_multi(X_train, y_train, X_test, y_test))

main()


#def load_dataset(dataset_path, labels_path, col_name):
#    
#    filenames_list = sorted(glob.glob(dataset_path + "/*"))
#    
#    print("Loading " + str(len(filenames_list)) + " samples..")
#      
#    dataset = []
#    for filename in filenames_list:
#        img = Image.open(filename).convert('L').resize((128,128))
#        img_array = np.array(img)
#        dataset.append(img_array)
#
#    labels = pd.read_csv(labels_path, header=1)[col_name]
#    
#    labels = np.array(labels)
#    
#    return dataset, labels    
#    
#def main():
#    
#    # Welcome
#    print("AMLS Assignment")
#    
#    # Load data set
#    X, y = load_dataset("dataset", "data.csv", "human")
#    
#    # Split the dataset
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#    print("Training set size =", len(X_train), "samples.")
#    print("Testing set size =", len(X_test), "samples.")
#
#    
#    #Reshaping the arrays
#    
#    X = np.array(X)
#    X = X.reshape(len(X), -1)
#    
#    X_train = np.array(X_train)
#    X_train = X_train.reshape(len(X_train), -1)
#    
#    X_test = np.array(X_test)
#    X_test = X_test.reshape(len(X_test), -1)
#    
#    #log_reg = LogisticRegression()
#    
#    
#    #log_reg.fit(X_train, y_train)
#    #preds = log_reg.predict(X_test)
#    
#    #print(metrics.confusion_matrix(y_test, preds))
#    
#    pca = PCA(n_components = 2)
#    pca.fit(X)
#    
#    kmeans = KMeans(n_clusters = 3)
#    kmeans.fit(X)
#          
#main()

