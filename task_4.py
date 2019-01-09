"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994
This module was used to in order to create and write a .csv file for task 4 containing
all the predicitions of the chosen human model on a previously unseen dataset. Specifically,
the model used is an SVM with a linear kernel and C = 0.01 previously trained on facial
landmarks.
"""

#Importing relevant packages/modules
import os
import csv
import lab2_landmarks as l2

from joblib import load

#Path to the folder containing saved models
models_dir = './models/'

#Path to the folder containing
images_dir = './unseentestset'

#Ensuring the correct directory
l2.images_dir = './unseentestset'

#Extracting the features (but not the labels) for the images as well
#as the images for which no landmarks were detected
test_data, failed_data = l2.extract_features('human')

#Reshaping the test data into a row in order to test the model on it
test_data = np.reshape(test_data, (test_data.shape[0], -1))

#Name of the saved model associated with the task
model_name = 'human.model'

#Loading the model from the directory
model = load(models_dir + model_name)

#Using the predict method in order to obtain predictions on the new test set
predictions = model.predict(test_data)

#The extract_features_labels and extract_features functions from lab2_landmarks
#transform the labels from -1 to 0. This is undone using this loop.
for i in range(len(predictions)):
    if predictions[i] == 0:
        predictions[i] = -1

#Test/inference score obtained earlier on the old test set
accuracy = '99.4%'

#Sorting the filenames alphanumerically
filenames = sorted(os.listdir(images_dir))

#Creating an empty list for files for which landmarks were detected that will be
#dynamically populated
filenames_filtered = []

#Only appending test files for which for which facial landmarks were detected
#Landmarks were detected in 96 files out of 100 
for filename in filenames:
    if filename not in failed_data:
        filenames_filtered.append(filename)

#Opening and writing a .csv file        
with open('task_4.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([accuracy])
    
    #Only including predictions on files for which facial landmarks were detected
    for index, filename in enumerate(filenames_filtered):
        if filename not in failed_data:
            writer.writerow([filename, predictions[index]])
