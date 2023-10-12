#! /Users/asiu/opt/anaconda3/bin/python3


code_content = """
import sys
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate Macro precision rate
    precision_macro = precision_score(y_test, y_pred, average='macro')

    # Calculate Macro Recall
    recall_macro = recall_score(y_test, y_pred, average='macro')

    # Calculate Macro F1
    f1_macro = f1_score(y_test, y_pred, average='macro')

    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1 Score (Macro)": f1_macro,
    }

    return metrics, confusion_mat



def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues, custom_text=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")

    # print(cm)
    plt.figure(figsize=(12, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # imshow displays data on a 2D raster
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if custom_text:
        plt.text(0.3 * plt.gca().get_xlim()[0], 0.05 * plt.gca().get_ylim()[1], custom_text, fontsize=10, ha='right',
                 va='bottom', color='red')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def run_rf(path_to_training, path_to_testing, n_estimators, max_depth, min_samples_split):
    #path_to_training: the path to your training data
    #path_to_testing: the path to your testing data
    #n_estimators, max_depth, min_samples_split: the parameter needed for a random forest model, all of them are int data

    #designate your training data and corresponding label of your training dataset as x_train and y_train, respectively
    #designate your testing data and corresponding label of your testing dataset as x_test and y_test, respectively
    x_train = 
    y_train = 
    x_test = 
    y_test = 

    #The label of testing data is also the axis for confusion matrix in the result. 
    axis = y_test

    #You can choose your own random_state, this is for your convenience to reproduce your result. 
    random_forest = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

    # Train this model using x_train and y_train
    random_forest.fit(x_train, y_train)


    #Use the provided function to evaluate your model
        #random_forest_metrics contains: Accuracy, Precision(Macro), Recall(Macro), F1 Score(Macro), you can refer to their value by random_forest_metrics["name"]
        #In confusion_mat, you can see how your testing data is recognized by the classifier. 
    random_forest_metrics, confusion_mat = evaluate_model(random_forest, x_test, y_test)



    print("===============================================================================================================")
    text = "The accuracy of random forest model with feature combination is: ", random_forest_metrics['Accuracy']
    plot_confusion_matrix(confusion_mat, axis,
                          "Confusion Matrix for random forest model with features RMS & Skewness & Kurtosis & Mean Spectral Energy",
                          normalize=True, custom_text=text)
    print("The accuracy of random forest model with feature combination is: ", random_forest_metrics['Accuracy'])

    """

print(code_content)