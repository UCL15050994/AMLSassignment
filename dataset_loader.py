"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994

This module serves the purpose of a dataset loader. The get_data function
implemented below allows the user to fetch the image data and any associated
labels, of which the former is obtained in one of two formats: raw RGB
values or facial landmarks (using the lab2_landmarks.py module).
"""

#Importing relevant packages/modules
import os
import cv2
import numpy as np
import pandas as pd
import lab2_landmarks as l2

#Path to the images as well as the labels
images = './images1'
labels = './dataset/data.csv'

def get_data(form, label):
    """
    The function takes two string parameters, form and label. The parameter
    form takes one of two values, 'rgb' or 'landmarks'. The parameter label 
    can take on any of the following values: 'hair_color', 'eyeglasses',
    'smiling', 'young', or 'human'. The function returns two numpy arrays,
    dataset_array and labels_array.
    
    Invoking this function with form = 'rgb' fetches the raw RGB values for
    each image with associated labels. Likewise, invoking this function with 
    form = 'landmarks' fetches 68 facial landmarks for each image with 
    associated labels. The latter is done using the modified lab2_landmarks.py
    module
    """
    
    #If the raw RGB values are required
    if form == 'rgb':
        
        #Creating lists for the raw RGB values and associated labels
        f_dataset = []
        f_labels = []
        
        #A basic dictionary for which the keys are strings corresponding to
        #the labels and the values are integers
        D = {'hair_color':1, 'eyeglasses':2, 'smiling':3, 'young':4, 'human':5}
        
        #Fetching the labels from the data.csv file and indexing using
        #keys for the dictionary D
        df = pd.read_csv(labels, skiprows=1, index_col='file_name')
        newdf = df[df.columns[D[label] - 1]]
        
        #Looping through the dataset and dynamically populating the dataset 
        #list with raw RGB values for each image
        for (root, directory, dataset) in os.walk('{0}'.format(images)):
            for file in dataset:

                #Using the imread function to read each image file 
                #The argument 1 means that the image is NOT grayscaled
                img = cv2.imread('{0}/{1}'.format(images, file), 1)
                
                #Downsampling the image from 256 x 256 to 128 x 128 so as
                #to reduce training times and speed up hyperparametrization
                resize_img = cv2.resize(img, (128, 128))
                
                #Update: Cropping the image to only include the top half for hair color classification
                if label == 'hair_color':
                     resize_img = resize_img[0:63,0:127]
                         
                #Appending the raw RGB values to the dataset list and 
                #updating the labels list
                f_dataset.append(resize_img)
                f_labels.append(int(file[:-4]))
        
        #Only choosing the relevant labels from the data.csv file
        f_labels = newdf.loc[f_labels]
        f_labels = f_labels.values.tolist()
        
        #Update: Filtering out mislabeled images for which hair color is labeled as -1
        if label == 'hair_color':
            
            #Creating new lists to be dynamically populated with nonnegative labels and corresponding images
            #that are correctly labeled
            f_labels_new = []
            f_dataset_new = []
            
            #Looping through the original f_labels list
            for i in range(len(f_labels)):
                
                #Only include images and corresponding labels if the images are labeled correctly
                if f_labels[i] != -1:
                    f_labels_new.append(f_labels[i])
                    f_dataset_new.append(f_dataset[i])
                    
            #Converting f_dataset_new and f_labels_new to numpy arrays       
            dataset_array = np.array(f_dataset_new)
            labels_array = np.array(f_labels_new)
            
            return dataset_array, labels_array
        
        #Proceed in the usual manner for other labels
        else:
            
            #Converting f_dataset and f_labels to numpy arrays
            dataset_array = np.array(f_dataset)
            labels_array = np.array(f_labels)
            
            return dataset_array, labels_array
        

    #If the facial landmarks are required
    elif form == 'landmarks':
        
        #Using the extract_features_labels function from lab2_landmarks.py
        #in order to obtain the facial landmarks for each image in the images1
        #folder as well as associated labels
        #Binary labels -1 and 1 are mapped to 0 and 1 respectively unlike
        #the case for which form == 'rgb'
        dataset_array, labels_array = l2.extract_features_labels(label)
     
        return dataset_array, labels_array
