#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 02:45:44 2018

@author: mohamed
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import numpy as np

def load_dataset(dataset_path, labels_path):
    
    filenames_list = sorted(glob.glob(dataset_path + "/*"))
    
    print("Loading " + str(len(filenames_list)) + " samples..")
    
    dataset = np.array([np.ravel(np.array(Image.open(filename)), (256, 256)) for filename in filenames_list])
    
    labels = pd.read_csv(labels_path, header=1)
    
    return dataset, labels    
    
def main():
    
    # Welcome
    print("AMLS Assignment")
    
    # Load data set
    X, y = load_dataset("dataset", "data.csv")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("Training set size =", len(X_train), "samples.")
    print("Testing set size =", len(X_test), "samples.")
      
    
main()
