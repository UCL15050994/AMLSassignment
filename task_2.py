from joblib import load
import csv
import pandas as pd
import lab2_landmarks as l2
import os 
import cv2

models_dir = './models/'

images_dir = './testing_dataset'

filenames = sorted(os.listdir(images_dir))


test_data = []
for file in filenames:
    
    img = cv2.imread('{0}/{1}'.format(images_dir, file), 1)
    
    resize_img = cv2.resize(img, (128, 128))
    
    test_data.append(resize_img)
    


test_data = np.array(test_data)
test_data = np.reshape(test_data, (test_data.shape[0], -1))

failed_data = []

model_name = 'young.model'

model = load(models_dir + model_name)


accuracy = '86.5%'


filenames_filtered = []

for filename in filenames:
    if filename not in failed_data:
        filenames_filtered.append(filename)

with open('task_2.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([accuracy])
    for index, filename in enumerate(filenames_filtered):
        if filename not in failed_data:
            writer.writerow([filename, predictions[index]])
        
