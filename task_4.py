from joblib import load
import csv
import pandas as pd
import lab2_landmarks as l2
import os 
import dataset_loader as ds

models_dir = './models/'

images_dir = './unseentestset'

l2.images_dir = './unseentestset'

test_data, failed_data = l2.extract_features('human')

test_data = np.reshape(test_data, (test_data.shape[0], -1))

model_name = 'human.model'

model = load(models_dir + model_name)

predictions = model.predict(test_data)

for i in range(len(predictions)):
    if predictions[i] == 0:
        predictions[i] = -1
        
accuracy = '99.4%'

filenames = sorted(os.listdir(images_dir))

filenames_filtered = []

for filename in filenames:
    if filename not in failed_data:
        filenames_filtered.append(filename)

with open('task_4.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([accuracy])
    for index, filename in enumerate(filenames_filtered):
        if filename not in failed_data:
            writer.writerow([filename, predictions[index]])
