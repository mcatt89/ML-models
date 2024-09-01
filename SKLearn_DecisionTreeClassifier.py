# Matt Herman
# 2/12/2024
# All code within is my own work.

#Tasks:
# 1. Identify two datasets from OpenML within the parameters
#       See Lines 39 - 55
# 2. Compare decision trees using entropy & Gini Index with a ROC Curve and AOC for each dataset
#       See output when executing file
# 3. Trees should use 10-fold cross validation
#       See Lines 116, 131, 147, 148
# 4. Include parameter search on min_samples_leaf using GridsearchCV with at least 5 values
#       See lines 41, 50 for where arrays are created for the parmeters
#       See lines 97, 98, 116, 131 for where the arrays are called and used with the GridSearchCV function
# 5. Program should outut the ROC Curve, AOC and other relevant statistics on the models
#       pyplot used for ROC Curves displaying Entropy and Gini Index results for each dataset
#       Text output from main function for positive baseline, best leaf size, comparison of AOC for both trees, and processing time

#Import Libraries
import numpy as np
import pandas
import sklearn as skl
import matplotlib 

from datetime import datetime
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import model_selection as ms
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
from sklearn.utils import Bunch, shuffle


# NASA Metrics Data Program https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfFeatures=between_10_100&id=1056 
nasa_dataset = datasets.fetch_openml(data_id=1056)
nasa_leaf_array = np.array([300,330,340,350,360,370,400])
nasa_title = "MC1 Software Defect Prediction"
nasa_pos_label = "TRUE"
nasa_true_pos = "last" # "first" or "last"
# nasa_dataset.data
# nasa_dataset.target

# Magic Telescope https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfFeatures=between_10_100&id=1120
magic_dataset = datasets.fetch_openml(data_id=1120) 
magic_leaf_array = np.array([45,46,48,50,52,53])
magic_title = "Magic Telescope"
magic_pos_label = "g"
magic_true_pos = "first" # "first" or "last"
#magic_dataset.data
#magic_dataset.target

#setup color variables for output
blue = "\033[34m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m" 

# Main function that will process datasets using 10-Fold Cross Validation with Entropy and Gini Index for the following output:
# 1. Positive Baseline
# 2. The minimum leaf size selected from an input array
# 3. ROC Curve for Entropy & Gini Index plotted together
# 4. AUC Scores for Entropy & Gini Index
# 5. Total processing time
def process_dataset(dataset_name,dataset_leaf_array,dataset_title,dataset_true_pos,dataset_pos_label):
    
    #dataset_name = nasa_dataset # For testing in the function, comment out when running the function
    #dataset_leaf_array = nasa_leaf_array# For testing in the function, comment out when running the function
    #dataset_title = nasa_title # For testing in the function, comment out when running the function
    #dataset_pos_label = "TRUE"
    
    #dataset_name = datasets.fetch_openml(data_id=1120) # For testing in the function, comment out when running the function
    #dataset_leaf_array = np.array([30,40,80,90,100]) # For testing in the function, comment out when running the function
    #dataset_title = "Magic Telescope" # For testing in the function, comment out when running the function
    #dataset_pos_label = "g" # For testing in the function, comment out when running the function
    #dataset_true_pos = "first" # For testing in the function, comment out when running the function
    #print(type(dataset_name)) # For testing in the function, comment out when running the function

    start_time = datetime.now()
    print(blue + 'Starting to process', dataset_title, 'dataset at:', start_time, ' ' + reset)

    #Shuffle Data
    #Commenting out for Assignment 1
    #x = dataset_name.data.to_numpy()
    #y = dataset_name.target.to_numpy()
    #x_shuffled, y_shuffled = shuffle(x,y,random_state=7)
    #dataset_name.data = x_shuffled
    #dataset_name.target = y_shuffled    


    
    #Manage data types to tuning evaluation & GridSearchCV
    leaf_list = dataset_leaf_array.tolist()
    dataset_parameters = [{"min_samples_leaf": leaf_list  }]
    
    #Create baseline for dataset
    target_df = pandas.DataFrame(dataset_name.target)
    column_label = target_df.columns[0]
    target_instances = target_df[column_label].count()
    positive_count = target_df[column_label].value_counts()[dataset_pos_label]
    positive_baseline = positive_count / target_instances
    print(blue + 'Positive baseline of the data is:', positive_baseline)

    if positive_baseline == 0.5:
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing Baseline (AUC = 0.5)')
   
    #Create Entropy & Gini Decision Tree Classifiers
    entropy_tree = dtc(criterion="entropy")
    gini_tree = dtc(criterion="gini")

    #Entropy Tree Tuning
    et_tuned_dtc = ms.GridSearchCV(entropy_tree, dataset_parameters, scoring="roc_auc", cv=10)
    et_tuned_dtc.fit(dataset_name.data, dataset_name.target)
    et_best_leaf_size = et_tuned_dtc.best_params_['min_samples_leaf']

    #Check where parameters fall in range
    if et_best_leaf_size == np.max(dataset_leaf_array):
        print(yellow + "WARNING: Best leaf size for", dataset_title, "is the maximum value in the array.")
        print("RECOMENDATION: Add additional higher values to the array." + reset)
    elif et_best_leaf_size == np.min(dataset_leaf_array):
        print(yellow + "WARNING: Best leaf size for", dataset_title, "is the minimum value in the array.")
        print("RECOMENDATION: Add additional lower values to the array." + reset)
    else:
        print(green + "The best leaf size for", dataset_title, "Entropy Tree is", et_best_leaf_size, " and within the minimum and maximum of the array." + reset)
        
    #Gini Tree Tuning
    gn_tuned_dtc = ms.GridSearchCV(gini_tree, dataset_parameters, scoring="roc_auc", cv=10)
    gn_tuned_dtc.fit(dataset_name.data, dataset_name.target)
    gn_best_leaf_size = gn_tuned_dtc.best_params_['min_samples_leaf']

    #Check where parameters fall in range
    if gn_best_leaf_size == np.max(dataset_leaf_array):
        print(yellow + "WARNING: Best leaf size for Gini Index Tree", dataset_title, "is the maximum value in the array.")
        print("RECOMENDATION: Add additional higher values to the array." + reset)
    elif gn_best_leaf_size == np.min(dataset_leaf_array):
        print(yellow + "WARNING: Best leaf size for Gini Index Tree", dataset_title, "is the minimum value in the array.")
        print("RECOMENDATION: Add additional lower values to the array." + reset)
    else:
        print(green + "The best leaf size for", dataset_title, "Gini Index Tree is", gn_best_leaf_size, "and within the minimum and maximum of the array." + reset)
        

    #Y_Scores
    et_y_scores = ms.cross_val_predict(et_tuned_dtc, dataset_name.data, dataset_name.target, method="predict_proba", cv=10)
    gn_y_scores = ms.cross_val_predict(gn_tuned_dtc, dataset_name.data, dataset_name.target, method="predict_proba", cv=10)

    #Create ROC Curve based on true position from function parameter
    if dataset_true_pos == "first":
        et_fpr, et_tpr, et_th = roc_curve(dataset_name.target,et_y_scores[:,0],pos_label=dataset_pos_label)
        gn_fpr, gn_tpr, gn_th = roc_curve(dataset_name.target,gn_y_scores[:,0],pos_label=dataset_pos_label)
    else:
        et_fpr, et_tpr, et_th = roc_curve(dataset_name.target,et_y_scores[:,1],pos_label=dataset_pos_label)
        gn_fpr, gn_tpr, gn_th = roc_curve(dataset_name.target,gn_y_scores[:,1],pos_label=dataset_pos_label)
    
    #Plot both ROC Curves
    plt.xlabel("1 - Specificty")
    plt.ylabel("Sensitivity")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(dataset_title + " ROC Curve")
    plt.plot(et_fpr,et_tpr,label="Entropy Tree")
    plt.plot(gn_fpr,gn_tpr,label="Gini Tree")
    plt.legend()
    plt.show()

    #Evaluate AUC
    et_auc = roc_auc_score(dataset_name.target, et_y_scores[:,1])
    gn_auc = roc_auc_score(dataset_name.target, gn_y_scores[:,1])

    #Print output to screen
    print(green +'For decision trees created from', dataset_title, ':')
    if et_auc > gn_auc:
        auc_diff = et_auc - gn_auc
        print('Entropy performed', auc_diff, 'better with a', et_auc, 'AUC')
        print('While Gini Index produced a', gn_auc, 'AUC' + reset)
    elif gn_auc > et_auc:
        auc_diff = gn_auc - et_auc
        print('Gini Index performed', auc_diff, 'better with a', gn_auc, 'AUC')
        print('While Entropy Index produced a', et_auc, 'AUC' + reset)
    else:
        print('Both Entropy and Gini Index produced the same AUC')
        print('Entropy AUC:', et_auc)
        print('Gini Index AUC:', gn_auc)
        print(yellow + 'RECOMMENDATION: Check varialbes to confirm result' + reset)

    end_time = datetime.now()
    print(blue + 'Processing of', dataset_title, 'complete at:', end_time, ' ' + reset)

    processing_time = end_time - start_time
    print(blue + 'Total processing time:', processing_time, ' ' + reset )
    print()




#Execute function on datasets
process_dataset(nasa_dataset,nasa_leaf_array,nasa_title,nasa_true_pos,nasa_pos_label)

process_dataset(magic_dataset,magic_leaf_array,magic_title,magic_true_pos,magic_pos_label)

