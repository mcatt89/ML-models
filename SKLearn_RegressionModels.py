# Matt Herman
# 2/26/2024
# All code within is my own work.

#Dataset
# Diamonds https://openml.org/search?type=data&sort=qualities.NumberOfFeatures&status=active&qualities.NumberOfClasses=lte_1&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfFeatures=between_10_100&order=asc&id=42225 

#Import Libraries
import numpy as np
import pandas
import sklearn as skl
import matplotlib 

from datetime import datetime
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.linear_model import LinearRegression as lr 
from sklearn import model_selection as ms
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay
#from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
#from sklearn.metrics import roc_auc_score
#from sklearn.datasets import load_iris
from sklearn.utils import Bunch, shuffle
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer 
from sklearn.neighbors import KNeighborsRegressor as knr

#setup color variables for output
blue = "\033[34m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m" 


diamond_dataset = datasets.fetch_openml(data_id=42225)
diamond_dtr_parameters = [{"min_samples_leaf":[5,10,15,20,35]}] 
diamond__knn_parameters = [{"n_neighbors":[3,7,11,19,21]}] 
#diamond_leaf_array = np.array([300,330,340,350,360,370,400, 425, 450])
diamond_title = "Diamonds"
#diamond_pos_label = "TRUE"
#diamond_true_pos = "last" # "first" or "last"
#check dataset load
#diamond_dataset.feature_names
#diamond_dataset.data
#diamond_dataset.target
#type(diamond_dataset.feature_names)
#diamond_dataset.data.info()
#diamond_dataset.data["clarity"].unique()

#Transform nominal features with one hot encoder
ohe = OneHotEncoder(sparse_output=False)
diamond_ct = ColumnTransformer([('encoder', ohe, [1,2,3])], remainder="passthrough")
diamond_new_data = diamond_ct.fit_transform(diamond_dataset.data)
#diamond_ct.get_feature_names_out()

#Scale features for KNN to values 0 - 1
scaler = MinMaxScaler()
scaler.fit(diamond_new_data)
scaler.data_max_
diamond_scaled_data = scaler.transform(diamond_new_data)

#K3 Scoring
k3_examples, k3_training_scores, k3_test_scores = learning_curve(knr(n_neighbors=3), diamond_scaled_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
k3_test_rmse =  0 - k3_test_scores
k3_test_means = np.mean(k3_test_rmse, axis=1)
print(blue + "K=3 Scoring Complete" + reset)

#K9 Scoring
k9_examples, k9_training_scores, k9_test_scores = learning_curve(knr(n_neighbors=9), diamond_scaled_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
k9_test_rmse =  0 - k9_test_scores
k9_test_means = np.mean(k9_test_rmse, axis=1)
print(blue + "K=9 Scoring Complete" + reset)

#K21 scoring
k21_examples, k21_training_scores, k21_test_scores = learning_curve(knr(n_neighbors=21), diamond_scaled_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
k21_test_rmse =  0 - k21_test_scores
k21_test_means = np.mean(k21_test_rmse, axis=1)
print(blue + "K=11 Scoring Complete" + reset)

#Tuned KNN
tuned_knn =  ms.GridSearchCV(knr(), diamond__knn_parameters, scoring="neg_root_mean_squared_error", cv=10)
knn_examples, knn_training_scores, knn_test_scores = learning_curve(tuned_knn, diamond_scaled_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
knn_test_rmse =  0 - knn_test_scores
knn_test_means = np.mean(knn_test_rmse, axis=1)
print(blue + "Tuned KNN Scoring Complete" + reset)


#Decision Tree Regressor
diamond_dtr = dtr(criterion="squared_error")
tuned_dtr = ms.GridSearchCV(diamond_dtr, diamond_dtr_parameters, scoring="neg_root_mean_squared_error", cv=10)
#tuned_dtr.fit(diamond_new_data, diamond_dataset.target)
dtr_examples, dtr_training_scores, dtr_test_scores = learning_curve(tuned_dtr, diamond_new_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
dtr_test_rmse =  0 - dtr_test_scores
dtr_test_means = np.mean(dtr_test_rmse, axis=1)
print(blue + "Tuned Decision Tree Regressor Scoring Complete" + reset)

#Liner Regression
diamond_lr = lr()
lr_examples, lr_training_scores, lr_test_scores = learning_curve(diamond_lr, diamond_new_data, diamond_dataset.target, train_sizes=[0.01,0.06,0.1,0.3,0.4,0.6,0.8,1.0], cv=10, scoring="neg_root_mean_squared_error")
lr_test_rmse =  0 - lr_test_scores
lr_test_means = np.mean(lr_test_rmse, axis=1)
print(blue + "Linear Regression Scoring Complete" + reset)

#Plot Nearest Neighbor Results
plt.plot(k3_examples, k3_test_means, "mo-", label= 'K = 3')
plt.plot(k9_examples, k9_test_means, "bo-", label= 'K = 9')
plt.plot(k21_examples, k21_test_means, "co-", label= 'K = 21')


plt.title("KNN Models Learning Curves")
plt.xlabel("Examples")
plt.ylabel("Root Mean Squared Errors")
plt.legend()
plt.show()

#KNN, DTR, LR Plot
plt.plot(knn_examples, knn_test_means, "bo-", label= 'K Nearest Neighbor')
plt.plot(dtr_examples, dtr_test_means, "mo-", label= 'Decision Tree Regressor')
plt.plot(lr_examples, lr_test_means, "co-", label= 'Linear Regression')

plt.title("Learning Curves for Tuned Models")
plt.xlabel("Examples")
plt.ylabel("Root Mean Squared Errors")
plt.legend()
plt.show()

#Create Data Table for KNN Models
type(k9_examples)
e1 = str(k9_examples[0]) + ' Examples'
e2 = str(k9_examples[1]) + ' Examples'
e3 = str(k9_examples[2]) + ' Examples'
e4 = str(k9_examples[3]) + ' Examples'
e5 = str(k9_examples[4]) + ' Examples'
e6 = str(k9_examples[5]) + ' Examples'
e7 = str(k9_examples[6]) + ' Examples'
e8 = str(k9_examples[7]) + ' Examples'
learning_curve_df = pandas.DataFrame(columns = ['K Value', e1, e2, e3, e4, e5, e6, e7, e8] )

row_2 = ['K=3',k3_test_means[0],k3_test_means[1],k3_test_means[2],k3_test_means[3],k3_test_means[4],k3_test_means[5],k3_test_means[6],k3_test_means[7]]
row_0 = ['K=9',k9_test_means[0],k9_test_means[1],k9_test_means[2],k9_test_means[3],k9_test_means[4],k9_test_means[5],k9_test_means[6],k9_test_means[7]]
row_1 = ['K=21',k21_test_means[0],k21_test_means[1],k21_test_means[2],k21_test_means[3],k21_test_means[4],k21_test_means[5],k21_test_means[6],k21_test_means[7]]

learning_curve_df.loc[len(learning_curve_df)] = row_2
learning_curve_df.loc[len(learning_curve_df)] = row_0
learning_curve_df.loc[len(learning_curve_df)] = row_1

learning_curve_df = learning_curve_df.round(2)
#learning_curve_df.to_csv(path_or_buf=r'C:\temp\KNN.csv', mode='a', header=True, index=False)
print(learning_curve_df)

#Print Tuned Learning Curve Data
tuned_learning_curve_df = pandas.DataFrame(columns = ['K Value', e1, e2, e3, e4, e5, e6, e7, e8] )

tuned_row_0 = ['DTR',dtr_test_means[0],dtr_test_means[1],dtr_test_means[2],dtr_test_means[3],dtr_test_means[4],dtr_test_means[5],dtr_test_means[6],dtr_test_means[7]]
tuned_row_1 = ['LR',lr_test_means[0],lr_test_means[1],lr_test_means[2],lr_test_means[3],lr_test_means[4],lr_test_means[5],lr_test_means[6],lr_test_means[7]]
tuned_row_2 = ['KNN',knn_test_means[0],knn_test_means[1],knn_test_means[2],knn_test_means[3],knn_test_means[4],knn_test_means[5],knn_test_means[6],knn_test_means[7]]

tuned_learning_curve_df.loc[len(tuned_learning_curve_df)] = tuned_row_0
tuned_learning_curve_df.loc[len(tuned_learning_curve_df)] = tuned_row_1
tuned_learning_curve_df.loc[len(tuned_learning_curve_df)] = tuned_row_2

tuned_learning_curve_df = tuned_learning_curve_df.round(2)
#tuned_learning_curve_df.to_csv(path_or_buf=r'C:\temp\Tuned.csv', mode='a', header=True, index=False)
print(tuned_learning_curve_df)