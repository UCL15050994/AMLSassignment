# AMLSassignment

## Preamble

This is the git repository for the ELEC0132: Applied Machine Learning Systems module, linked to the submitted paper "Supervised Learning Methods for Facial Recognition and Classification". 

The following is a link to the .zip file containing CSV prediction results as well as the code for training and testing all of the classification models discussed in the report:

<link to zip file>

## Python Version and Library Installations

Python 3.6.6 was used throughout this project. In order for the code to run, please ensure that the following libraries and packages are installed:

dlib  19.16.0

joblib  0.13.0

Keras  2.2.4

Keras-Applications  1.0.6

Keras-Preprocessing  1.0.5

numpy  1.15.4

opencv-python  3.4.4.19

pandas  0.23.4

scikit-learn  0.20.1

scipy  1.1.0

tensorflow  1.12.0

## Running the Code

In order to run the code, please extract all the files and folders in the .zip file into a folder. The code is modularized such that every task (e.g. finding the hyperparameters and metrics or removing outliers) is carried out in a separate module. 

## Unseen Test Data Prediction Results

The prediction results on unseen test data can be found in the .csv files. task_1.csv corresponds to emotion recognition, task_2.csv corresponds to age identification, task_3.csv corresponds to glasses detection, task_4.csv corresponds to human detection, and task_5.csv corresponds to hair color recognition.

## Preprocessing - Noise Removal

For noise removal with grayscaling run the module preprocess_noise_gray.py. For noise removal without grayscaling (which was the noise removal primarily used) run the module preprocess_noise_color.py. The folders included in the .zip file already contain the filtered images without outliers in folders images1 (color preprocessing) and images0 (gray preprocessing), but in order to test the code one can delete the images inside the folders (but not the folder themselves) and run the code. 

## Training, testing, validation, and tuning hyperparameters

This can be found in the modules gridsearch_svm.py and gridsearch_logreg.py for SVM and logistic regression implementations respectively. The modules output a number of metrics, many of which have been used in the report, and allows the user to optimize the hyperparameters for a given classifier. In order to change the hyperparameter search space, change the values in the parameters dictionary in the two modules.

## Saving the Models

For convenience in the .zip file (but not in the git repository due to limitations on upload sizes), all of the finalized models were saved in the models folder found in the .zip file. Those are the optimal classifiers for each task; for example, linear SVMs with a value of C = 0.01 trained on facial landmarks for the smiling task. 

In order to save those models, the modules smilingsave.py, youngsave.py, eyeglassessave.py, humansave.py, and haircolorsave.py can be used. 

## Generating the CSV files

Every task_x.csv file has an accompanying task_x.py file that was used to generate it. It is important that the models be present in the models folder before those modules can run.

## Key Results

The key results included in the report can be found in the Results folder. Confusion matrices were printed out in gridsearch_svm.py and gridsearch_logreg.py, and were generated in the form presented in the report through the use of the external Scikit-learn module plot_confusion_matrix.py found in the Auxiliary sklearn modules folder. The table of results was also generated with the help of gridsearch_svm.py and gridsearch_logreg.py. Finally, the learning curves were generated through the use of the external Scikit-learn module plot_learning_curve.py found in the Auxiliary sklearn modules folder. 




