"""
ELEC0132: Applied Machine Learning Systems
Student Number: 15050994

This is the module used for removing the noise/outliers (mostly corresponding
to nature images) from the original celebsa dataset before any further
preprocessing/classification is carried out. In this module, the images
are grayscaled. Refer to preprocess_noise_color.py for outlier removal
with RGB images. 
"""

#Importing relevant packages/modules
import os
import cv2
import dlib
import shutil
import numpy as np

from keras.preprocessing import image

#Path to the original dataset as well as the labels
directory = './dataset/celebsa'
labels = './dataset/data.csv'

#Using the Histogram of Oriented Gradients (HOG) classifier from the dlib
#library as implemented in lab2_landmarks.py
#Allows face detection for every image through the use of 68 facial landmarks
#Images for which no facial landmarks are detected (e.g. nature images) will
#be removed
hog = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Creating a list for the outliers
outliers = []

#Looping through the dataset and dynamically populating the outliers list
#with images for which no faces were detected
for (root, dirs, files) in os.walk('{0}'.format(directory)):
    #Looping through the image files
    for image_file in files:
        
        #Using the imread function to read each image file
        #Each image object is then converted to an array using
        #the img_to_array function 
        #The argument 0 means that the image is grayscaled
        img = cv2.imread('{0}/{1}'.format(directory, image_file), 0)
        img_array = image.img_to_array(img)
        
        #Reshaping the image array as appropriate into an array of shape
        #(256, 256)
        resized_image = img_array.astype('uint8')
        resized_image_256 = resized_image.reshape(256,256)
        
        #Detecting the facial landmarks
        rects = hog(resized_image_256, 1)
        num_faces = len(rects)
        
        #If no faces are detected in the image print out the image filename
        #and append the image to the outliers list
        #The slicing removes ".png" from the filename string
        if num_faces == 0:
            print('Outlier detected: {0}'.format(image_file))
            outliers.append(int(image_file[:-4]))

#The number of outliers detected
print('There are', len(outliers), 'outliers.')

#Creating a clone list for the outliers
outliers_new = [] 

#For loop adds ".png" back to filenames
for image_file in outliers:
    file_str = '{0}.png'.format(image_file)
    outliers_new.append(file_str)

#Path to the new folder in which the images ready for further preprocessing
#and classification will be placed
new_dir = 'images2'

#Copying the images (non-outliers) from the old directory to the new one
for (root, directories, files) in os.walk('{0}'.format(directory)):
    for image_file in files:
        #Only copy image if it is not an outlier
        if image_file not in outliers_new:
            shutil.copy('./dataset/celebsa/{0}'.format(image_file), new_dir)
