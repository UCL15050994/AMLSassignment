import os
import cv2
import dlib
import shutil
import numpy as np
import datetime as dt
from keras.preprocessing import image


directory = './dataset/celebsa'
labels = './dataset/data.csv'

hog = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

outliers = []


for (root, dirs, files) in os.walk('{0}'.format(directory)):
    print(len(files))
    print('Start learning at {}'.format(str(start_time)))
    for image_file in files:

        img = cv2.imread('{0}/{1}'.format(directory, image_file), 1)
        img_array = image.img_to_array(img)

        resized_image = img_array.astype('uint8')
        resized_image_256 = resized_image.reshape(256,256,3)

        rects = hog(resized_image_256, 1)
        num_faces = len(rects)

        if num_faces == 0:
            print('outlier detected: {0}'.format(image_file))

            outliers.append(int(image_file[:-4]))

print(len(outliers))


outliers_new = [] 

for image_file in outliers:
    file_str = '{0}.png'.format(image_file)
    outliers_new.append(file_str)

new_dir = 'new1'

for (root, directories, files) in os.walk('{0}'.format(directory)):
    for image_file in files:
        if image_file not in outliers_new:
            shutil.copy('./dataset/celebsa/{0}'.format(image_file), new_dir)
