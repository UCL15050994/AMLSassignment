"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994
This module was used to in order to create and write a .csv file for task 2 containing
all the predicitions of the chosen young model on a previously unseen dataset. Specifically,
the model used is a logistic regression classifier with an lbfgs solver and C = 0.01 previously
trained on raw RGB values from resized images.
"""

#Importing relevant packages/modules
import os
import cv2
import csv

from joblib import load

#Path to the folder containing saved models
models_dir = './models/'

#Path to the folder containing new test set
images_dir = './unseentestset'

#Sorting the filenames alphanumerically
filenames = sorted(os.listdir(images_dir))

#Creating a list for the raw RGB values
test_data = []

#Looping through the test set and dynamically populating the
#test_data array with raw RGB values for each image 
for file in filenames:
    
    #Using the imread function to read each image file 
    #The argument 1 means that the image is NOT grayscaled
    img = cv2.imread('{0}/{1}'.format(images_dir, file), 1)
    
    #Downsampling the image from 256 x 256 to 128 x 128
    resize_img = cv2.resize(img, (128, 128))
    
    #Appending the raw RGB values to the test data_list
    test_data.append(resize_img)
    
#Converting test_data to a numpy array
test_data = np.array(test_data)

#Reshaping the test data into a row in order to test the model on it
test_data = np.reshape(test_data, (test_data.shape[0], -1))

#No failed data in this case unlike landmarks
failed_data = []

#Name of the saved model associated with the task
model_name = 'young.model'

#Loading the model from the directory
model = load(models_dir + model_name)

#Using the predict method in order to obtain predictions on the new test set
predictions = model.predict(test_data)

#Test/inference score obtained earlier on the old test set
accuracy = '86.5%'

#No filtering is required in this case unlike landmarks
filenames_filtered = []

for filename in filenames:
    if filename not in failed_data:
        filenames_filtered.append(filename)

#Opening and writing a .csv file 
with open('task_2.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([accuracy])
    for index, filename in enumerate(filenames_filtered):
        if filename not in failed_data:
            writer.writerow([filename, predictions[index]])
        
