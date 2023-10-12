#! /Users/asiu/opt/anaconda3/bin/python3
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting, woohoo!
import numpy as np
import scipy as sp
from scipy import signal
import random
import os
import math
import itertools

# We wrote this gesturerec package for the class
# It provides some useful data structures for the accelerometer signal
# and running experiments so you can focus on writing classification code,
# evaluating your solutions, and iterating
import inductancerec.utility as dfutils
import inductancerec.data as dfdata
import inductancerec.vis as dfvis

from inductancerec.data import SensorData
from inductancerec.data import DeformationSet
import Utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Scikit-learn stuff
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, StratifiedKFold

import pandas as pd

import seaborn as sns



# Load the data
root_deformation_log_path = '/Users/asiu/Desktop/FluxableCode/DataLogsForSpringSensing'

print("Found the following deformation log sub-directories")
print(dfutils.get_immediate_subdirectories(root_deformation_log_path))

deformation_log_paths = dfutils.get_immediate_subdirectories(root_deformation_log_path)
map_deformation_sets = dict()
selected_deformation_set = None

for deformation_log_path in deformation_log_paths:
    path_to_deformation_log = os.path.join(root_deformation_log_path, deformation_log_path)
    print("Creating a DeformationSet object for path '{}'".format(path_to_deformation_log))
    deformation_set = DeformationSet(path_to_deformation_log)
    deformation_set.load()
    map_deformation_sets[deformation_set.name] = deformation_set

if selected_deformation_set is None:
    # Since we load multiple deformation sets and often want to just visualize and explore
    # one set, in particular, we set a selected_deformation_set variable here
    # Feel free to change this
    #selected_deformation_set = get_random_deformation_set(map_deformation_sets)
    selected_deformation_set = dfdata.get_deformation_set_with_str(map_deformation_sets, "Feng")
    if selected_deformation_set is None:
        # if the selected gesture set is still None
        selected_deformation_set = dfdata.get_random_deformation_set(map_deformation_sets);

print("The selected deformation set:", selected_deformation_set)


# Preprocess the trial and data
for deformation_set in map_deformation_sets.values():
    for deformation_name, trials in deformation_set.map_deformations_to_trials.items():
        for trial in trials:
            Utility.preprocess_trial(trial)

# Plot the signal from time domain
# Utility.plot_signals(selected_deformation_set, ['ind_data','ind_data_p'])

# Plot the signal from frequency domain
# Utility.plot_fft_signals(selected_deformation_set,['ind_data','ind_data_p'])

# Explore features in time domain
    # Plot the Standard Deviation
# Utility.plot_feature_1d(selected_deformation_set,Utility.calculate_sd,title="Standard Deviation",use_random_y_jitter=True,xlim=None)

    # Plot the Root Mean Square
# Utility.plot_feature_1d(selected_deformation_set,Utility.calculate_rms,title="Root Mean Square",use_random_y_jitter=True,xlim=None)

    # Plot the Skewness
# Utility.plot_feature_1d(selected_deformation_set,Utility.calculate_sd,title="Standard Deviation",use_random_y_jitter=True,xlim=None)

    # Plot the Kurtosis
# Utility.plot_feature_1d(selected_deformation_set,Utility.calculate_kurtosis,title="Kurtosis",use_random_y_jitter=True,xlim=None)

# Explore features in frequency domain
    # Plot the Mean Spectral Energy
# Utility.plot_feature_1d(selected_deformation_set,Utility.calculate_mean_energy,title="Mean Spectral Energy",use_random_y_jitter=True,xlim=None)

    # Plot the Bandwidth
# Utility.plot_feature_1d(selected_deformation_set,Utility.bandwidth,title="Bandwidth",use_random_y_jitter=True,xlim=None)

    # Plot the Mean
# Utility.plot_feature_1d(selected_deformation_set,Utility.spectral_mean,title="Spectral Mean",use_random_y_jitter=True,xlim=None)

    # Plot the Standard Deviaton
# Utility.plot_feature_1d(selected_deformation_set,Utility.spectral_sd,title="Spectral Standard Deviation",use_random_y_jitter=True,xlim=None)

    # Plot the Minimum Index
# Utility.plot_feature_1d(selected_deformation_set,Utility.min_index,title="Spectral Minimum Index",use_random_y_jitter=True,xlim=None)

    # Center of Mass
# Utility.plot_feature_1d(selected_deformation_set,Utility.center_of_mass,title="Spectral Center of Mass",use_random_y_jitter=True,xlim=None)

# Plot the 3D images of feature combination []
# Utility.plot_feature_3d(selected_deformation_set, Utility.calculate_rms, Utility.calculate_mean_energy, Utility.spectral_sd, xlabel="Root Mean Square", ylabel="Mean Spectral Energy", zlabel="Spectral Standard Deviation", title=None,figsize=(12,12))



#generate 5-folds list
list_folds = Utility.generate_kfolds_scikit(5,selected_deformation_set,seed=5)
Utility.print_folds(list_folds)
#Create a dictionary that will have all the accuracy in the convenience of showing them out in a bar chart in the end.
result_accuracy_dict = dict()




# RF model using combination [RMS, Skewness, Kurtosis, Mean Spectral Energy]
# parameter space
if sys.argv[1] == "Random_Forest":

    total_confusion_matrix = np.empty((int(sys.argv[5]),int(sys.argv[5])))
    axis = list()
    overall_accuracy = list()

    fold_index = 0
    for fold in list_folds:
        # divide and get the training folds and testing fold
        # The current fold is treated as the testing data and the rest of list_folds are treated as the training data
        train_folds = [elem for elem in list_folds if elem != fold]

        # train_trial keeps all the trials in the train_folds
        train_trial = list()
        # x_train keeps the data that is going to be used as training data
        x_train = list()
        # y_train keeps labels corresponding to every trial and is going to be used as the targets(labels) of training data
        y_train = list()

        for elem in train_folds:
            for deformation_name, trial in elem.items():
                y_train.append(deformation_name)
                train_trial.append(trial)
        for trial in train_trial:
            result = Utility.extract_feature_from_trial4(trial, Utility.calculate_rms, Utility.calculate_skewness, Utility.calculate_kurtosis,
                                                 Utility.calculate_mean_energy)
            x_train.append(result)
        #     print(x_train)
        #     print(y_train)

        x_test = list()
        y_test = list()
        # Get Features Set of testing data (i.e. x_test) and Targets (labels) of testing data (i.e. y_test)
        for deformation_name, test_trial in fold.items():
            x_test.append(Utility.extract_feature_from_trial4(test_trial, Utility.calculate_rms, Utility.calculate_skewness, Utility.calculate_kurtosis,
                                                      Utility.calculate_mean_energy))
            y_test.append(deformation_name)
        #     print(x_test)
        #     print(y_test)
        axis = y_test

        # Now we have the training data and the testing data, it's time to do the random forest model work

        # The following codes are used to determine optimal hyper-parameter combination using Grid Search.
        # Grid search tries all possible combinations in the parameter space and gives the most effective combination.
        #     rf = RandomForestClassifier(random_state=42)
        #     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4)
        #     grid_search.fit(x_train, y_train)
        #     best_params = grid_search.best_params_
        #     print(best_params)

        # Initial a random forest model
        # From the codes above, get the optimal combination "{'n_estimators=100, max_depth=2, min_samples_split=10'}"
        n = int(sys.argv[2])
        max_depth = int(sys.argv[3])
        min_samples_split = int(sys.argv[4])
        random_forest = RandomForestClassifier(random_state=42, n_estimators=n, max_depth=max_depth, min_samples_split=min_samples_split)

        # Train this model
        random_forest.fit(x_train, y_train)

        # Evaluate the model from the perspective of Accuracy, Precision, Recall, F1 score
        random_forest_metrics, confusion_mat = Utility.evaluate_model(random_forest, x_test, y_test)
        title = "Confusion Matrix for Testing fold {}".format(fold_index)
        # Utility.plot_confusion_matrix(confusion_mat, y_test, title)
        metric_names = list(random_forest_metrics.keys())
        metric_values = list(random_forest_metrics.values())
        title1 = "Evaluation Metrics for Testing fold {}".format(fold_index)
        # Utility.plot_evaluation_metric(metric_names, metric_values, title1)

        total_confusion_matrix += confusion_mat
        overall_accuracy.append(random_forest_metrics['Accuracy'])

        fold_index += 1

    print("===============================================================================================================")
    RF_accuracy = np.mean(overall_accuracy)
    text = "The overall accuracy of Random forest model is: ", RF_accuracy
    Utility.plot_confusion_matrix(total_confusion_matrix, axis,
                          "Confusion Matrix for Random Forest model with features RMS & Skewness & Kurtosis & Mean Spectral Energy",
                          normalize=True, custom_text=text)
    print("The overall accuracy of Random forest model is: ", RF_accuracy)



# SVM model using the same feature combination
# parameter space
if sys.argv[1] == "SVM":

    total_confusion_matrix = np.empty((int(sys.argv[5]),int(sys.argv[5])))
    axis = list()
    overall_accuracy = list()

    fold_index = 0
    for fold in list_folds:
        # divide and get the training folds and testing fold
        # The current fold is treated as the testing data and the rest of list_folds are treated as the training data
        train_folds = [elem for elem in list_folds if elem != fold]

        # train_trial keeps all the trials in the train_folds
        train_trial = list()
        # x_train keeps the data that is going to be used as training data
        # It is a two-dimension matrix: the first dimension is the number of trials(which is 4 folds * 7 deformaitons = 28)
        # The second dimension is the features chosen (which are three features: rms, mean energy, and spectral standard deviation)
        x_train = list()
        # y_train keeps labels corresponding to every trial and is going to be used as the targets(labels) of training data
        y_train = list()

        for elem in train_folds:
            for deformation_name, trial in elem.items():
                y_train.append(deformation_name)
                train_trial.append(trial)
        for trial in train_trial:
            result = Utility.extract_feature_from_trial4(trial, Utility.calculate_rms, Utility.calculate_skewness, Utility.calculate_kurtosis,
                                                 Utility.calculate_mean_energy)
            x_train.append(result)

        x_test = list()
        y_test = list()
        # Get Features Set of testing data (i.e. x_test) and Targets (labels) of testing data (i.e. y_test)
        for deformation_name, test_trial in fold.items():
            x_test.append(Utility.extract_feature_from_trial4(test_trial, Utility.calculate_rms, Utility.calculate_skewness, Utility.calculate_kurtosis,
                                                      Utility.calculate_mean_energy))
            y_test.append(deformation_name)
        axis = y_test

        # Now we have the training data and the testing data, it's time to do the SVM model work

        # The following codes are used to determine optimal hyper-parameter combination using Grid Search.
        # Grid search tries all possible combinations in the parameter space and gives the most effective combination.
        #     svm_model = SVC(random_state=42)
        #     grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=4)
        #     grid_search.fit(x_train, y_train)
        #     best_params = grid_search.best_params_
        #     print(best_params)

        # Initial an SVM model
        # From the codes above, get the optimal combination "{'C': 0.1, 'gamma': 1, 'kernel': 'poly'}"
        c = float(sys.argv[2])
        gamma = float(sys.argv[4])
        svm_model = SVC(random_state=42, C=c, gamma=gamma, kernel=sys.argv[3])

        # Train this model
        svm_model.fit(x_train, y_train)

        # Evaluate the model from the perspective of Accuracy, Precision, Recall, F1 score
        svm_metrics, confusion_mat = Utility.evaluate_model(svm_model, x_test, y_test)
        title = "Confusion Matrix for Testing fold {}".format(fold_index)
        # Utility.plot_confusion_matrix(confusion_mat, y_test, title)
        metric_names = list(svm_metrics.keys())
        metric_values = list(svm_metrics.values())
        title1 = "Evaluation Metrics for Testing fold {}".format(fold_index)
        # Utility.plot_evaluation_metric(metric_names, metric_values, title1)

        total_confusion_matrix += confusion_mat
        overall_accuracy.append(svm_metrics['Accuracy'])

        fold_index += 1

    print("===============================================================================================================")
    svm_accuracy = np.mean(overall_accuracy)
    text = "The overall accuracy of SVM model with feature combination is: ", svm_accuracy
    Utility.plot_confusion_matrix(total_confusion_matrix, axis,
                          "Confusion Matrix for SVM model with features RMS & Skewness & Kurtosis & Mean Spectral Energy",
                          normalize=True, custom_text=text)

    print("The overall accuracy of SVM model with feature combination is: ", svm_accuracy)




if sys.argv[1] == "XGB":

    total_confusion_matrix = np.empty((int(sys.argv[5]),int(sys.argv[5])))
    axis = list()
    overall_accuracy = list()

    fold_index = 0
    for fold in list_folds:
        # divide and get the training folds and testing fold
        # The current fold is treated as the testing data and the rest of list_folds are treated as the training data
        train_folds = [elem for elem in list_folds if elem != fold]

        # train_trial keeps all the trials in the train_folds
        train_trial = list()
        # x_train keeps the data that is going to be used as training data
        # It is a two-dimension matrix: the first dimension is the number of trials(which is 4 folds * 7 deformaitons = 28)
        # The second dimension is the features chosen (which are three features: rms, mean energy, and spectral standard deviation)
        x_train = list()
        # y_train keeps labels corresponding to every trial and is going to be used as the targets(labels) of training data
        y_train = list()

        for elem in train_folds:
            for deformation_name, trial in elem.items():
                y_train.append(deformation_name)
                train_trial.append(trial)
        for trial in train_trial:
            result = Utility.extract_feature_from_trial4(trial, Utility.calculate_rms, Utility.calculate_skewness,
                                                         Utility.calculate_kurtosis,
                                                         Utility.calculate_mean_energy)
            x_train.append(result)

        # Because the XGBoost model needs to deal with Integer type labels rather than original groups of labels, so use
        # LabelEncoder to transfer the original labels to an integer and use them to do the training and testing works.
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)
        # print(y_encoded)

        x_test = list()
        y_test = list()
        # Get Features Set of testing data (i.e. x_test) and Targets (labels) of testing data (i.e. y_test)
        for deformation_name, test_trial in fold.items():
            x_test.append(Utility.extract_feature_from_trial4(test_trial, Utility.calculate_rms, Utility.calculate_skewness, Utility.calculate_kurtosis,
                                                      Utility.calculate_mean_energy))
            y_test.append(deformation_name)
        y_test_encoded = label_encoder.fit_transform(y_test)
        axis = y_test

        # Now we have the training data and the testing data, it's time to do XGB model work

        # The following codes are used to determine optimal hyper-parameter combination using Grid Search.
        # Grid search tries all possible combinations in the parameter space and gives the most effective combination.
        #     xgb = XGBClassifier(random_state=42)
        #     grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=4)
        #     grid_search.fit(x_train, y_encoded)
        #     best_params = grid_search.best_params_
        #     print(best_params)

        # Initial an XGB model
        # From the codes above, get the optimal combination "{'learning_rate': 0.4, 'max_depth': 3, 'n_estimators': 100}"
        n = int(sys.argv[2])
        max_depth = int(sys.argv[3])
        learning_rate = float(sys.argv[4])
        xgb = XGBClassifier(random_state=42, objective='multi:softmax', num_class=7, learning_rate=learning_rate, max_depth=max_depth,
                            n_estimators=n)

        # Train this model
        xgb.fit(x_train, y_encoded)

        # Evaluate the model from the perspective of Accuracy, Precision, Recal, F1 score
        xgb_metrics, confusion_mat = Utility.evaluate_model(xgb, x_test, y_test_encoded)
        title = "Confusion Matrix for Testing fold {}".format(fold_index)
        metric_names = list(xgb_metrics.keys())
        metric_values = list(xgb_metrics.values())
        title1 = "Evaluation Metrics for Testing fold {}".format(fold_index)


        total_confusion_matrix += confusion_mat
        overall_accuracy.append(xgb_metrics['Accuracy'])

        fold_index += 1

    print("===============================================================================================================")
    xgb_accuracy = np.mean(overall_accuracy)
    text = "The overall accuracy of XGBoost model is: ", xgb_accuracy
    Utility.plot_confusion_matrix(total_confusion_matrix, axis,
                          "Confusion Matrix for XGB model with features RMS & Spectral Mean Energy & Spectral Standard Deviation",
                          normalize=True, custom_text=text)
    print("The overall accuracy of XGBoost model is: ", xgb_accuracy)