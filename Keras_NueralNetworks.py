# Matt Herman
# 4/13/2024
# All code within is my own work.

#Tasks
#
# 1. (Regression) Build 4 different neural network models using Keras for regression.  
#       Only use input and dense layers
#       Other parameters can be varied in the models
#       For each model, plot the training and validation errors.  If training has not platued, increase epochs
#       In a table show the minimum error for each model
# 2. (Classification) Build 4 different neural network models using Keras for classification.  
#       Only use input and dense layers
#       Other parameters can be varied in the models
#       For each model, plot the training and validation accuracy.  If training has not platued, increase epochs
#       In a table show the minimum error for each model



#Datasets
# Diamonds https://openml.org/search?type=data&sort=qualities.NumberOfFeatures&status=active&qualities.NumberOfClasses=lte_1&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfFeatures=between_10_100&order=asc&id=42225 
# Magic Telescope https://www.openml.org/search?type=data&sort=qualities.NumberOfNumericFeatures&status=active&qualities.NumberOfClasses=%3D_2&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfFeatures=between_10_100&id=1120 

#Import Libraries
import numpy as np
import pandas
import sklearn as skl
import matplotlib 

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras import callbacks
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer 


#Regression Function
def regression_network():
    #Import Dataset
    diamond_dataset = fetch_openml(data_id=42225)

    #Transform nominal features with one hot encoder
    ohe = OneHotEncoder(sparse_output=False)
    diamond_ct = ColumnTransformer([('encoder', ohe, [1,2,3])], remainder="passthrough")
    diamond_new_data = diamond_ct.fit_transform(diamond_dataset.data)
    #diamond_ct.get_feature_names_out()
    #diamond_new_data.shape

    #Scale features to values 0 - 1
    scaler = MinMaxScaler()
    scaler.fit(diamond_new_data)
    scaler.data_max_
    diamond_scaled_data = scaler.transform(diamond_new_data)

    #Split into train and validation data with shuffle
    x_train, x_val, y_train, y_val = train_test_split(diamond_scaled_data, diamond_dataset.target, test_size=0.25, random_state=0)

    #Format Callback to prevent overfitting
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)
  

    #Neural Network Structure
    #Model 1, Sigmoid, Layer 1 = 2xInput, Layer 2 = 1/3 Layer1, Output 1 Node for Regression
    nn1 = Sequential()
    nn1.add(Input((26,)))
    nn1.add(Dense(52,activation="sigmoid"))
    nn1.add(Dense(20,activation="sigmoid"))
    nn1.add(Dense(1))

    nn1.compile(optimizer="adam", loss="mse", metrics=["mse"])
   
    history1 = nn1.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    history1.history["mse"] 
    history1.history["val_mse"] 
    history1_epoch = len(history1.epoch)

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)

    #Model 2, Sigmoid, Layer 1 = 3xInput, Layer 2 = 1/4 Layer1, Output 1 Node for Regression
    nn2 = Sequential()
    nn2.add(Input((26,)))
    nn2.add(Dense(78,activation="sigmoid"))
    nn2.add(Dense(20,activation="sigmoid"))
    nn2.add(Dense(1))

    nn2.compile(optimizer="adam", loss="mse", metrics=["mse"])
   
    history2 = nn2.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history2.history["mse"] 
    #history2.history["val_mse"] 
    history2_epoch = len(history2.epoch)

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)

    #Model 3, Leaky Relu, Layer 1 = 2xInput, Layer 2 = 1/3 Layer1, Output 1 Node for Regression
    nn3 = Sequential()
    nn3.add(Input((26,)))
    nn3.add(Dense(52,activation="leaky_relu"))
    nn3.add(Dense(20,activation="leaky_relu"))
    nn3.add(Dense(1))

    nn3.compile(optimizer="adam", loss="mse", metrics=["mse"])
   
    history3 = nn3.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history3.history["mse"] 
    #history3.history["val_mse"] 
    history3_epoch = len(history3.epoch)

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)

    #Model 4, Leaky Relu, Layer 1 = 3xInput, Layer 2 = 1/4 Layer1, Output 1 Node for Regression
    nn4 = Sequential()
    nn4.add(Input((26,)))
    nn4.add(Dense(78,activation="leaky_relu"))
    nn4.add(Dense(20,activation="leaky_relu"))
    nn4.add(Dense(1))

    nn4.compile(optimizer="adam", loss="mse", metrics=["mse"])
   
    history4 = nn4.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history4.history["mse"] 
    #history4.history["val_mse"] 
    history4_epoch = len(history4.epoch)

    
    
    #Plot Results
    #Model 1
    plt.title("Diamond Price Dataset Model 1, Sigmoid, 2x Input")
    plt.plot(history1.history["mse"], label="Training")
    plt.plot(history1.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 2
    plt.title("Diamond Price Dataset Model 2, Sigmoid, 3x Input")
    plt.plot(history2.history["mse"], label="Training")
    plt.plot(history2.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 3
    plt.title("Diamond Price Dataset Model 3, Leaky Relu, 2x Input")
    plt.plot(history3.history["mse"], label="Training")
    plt.plot(history3.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 4
    plt.title("Diamond Price Dataset Model 4, Leaky Relu, 3x Input")
    plt.plot(history3.history["mse"], label="Training")
    plt.plot(history3.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Report Minimum MSE
    mse_df = pandas.DataFrame(columns=["Model","Trainging Epochs","Minimum Mean Squared Error"])
    mse1 = ["Model 1, Sigmoid, 2x Input", history1_epoch , min(history1.history["val_mse"])]
    mse2 = ["Model 2, Sigmoid, 3x Input", history2_epoch, min(history2.history["val_mse"])]
    mse3 = ["Model 3, Leaky Relu, 2x Input", history3_epoch, min(history3.history["val_mse"])]
    mse4 = ["Model 4, Leaky Relu, 3x Input", history4_epoch, min(history4.history["val_mse"])]

    mse_df.loc[len(mse_df)] = mse1
    mse_df.loc[len(mse_df)] = mse2
    mse_df.loc[len(mse_df)] = mse3
    mse_df.loc[len(mse_df)] = mse4

    mse_df = mse_df.round(2)

    print(mse_df)

   



def classification_network():
    #Import Dataset
    magic_dataset = fetch_openml(data_id=1120)

    #Scale Data
    magic_data = magic_dataset.data
    scaler = MinMaxScaler()
    scaler.fit(magic_data)
    scaler.data_max_
    magic_scaled_data = scaler.transform(magic_data)

    #OneHotEncoding for Nominal Target
    ohe = OneHotEncoder(sparse_output=False)
    tmp = [[x] for x in magic_dataset.target] 
    ohe_target = ohe.fit_transform(tmp)

    x_train, x_val, y_train, y_val = train_test_split(magic_scaled_data, ohe_target, test_size=0.25, random_state=0)

    #Format Callback to prevent overfitting
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)
    
    #Neural Network Structure
    #Model 1, Sigmoid, Layer 1 = 2xInput, Layer 2 = 1/3 Layer1, Output 2 Nodes for Classification
    nn1 = Sequential()
    nn1.add(Input((10,)))
    nn1.add(Dense(20,activation="sigmoid"))
    nn1.add(Dense(7,activation="sigmoid"))
    nn1.add(Dense(2,activation="softmax"))

    nn1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history1 = nn1.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history.history["accuracy"] 
    #history.history["val_accuracy"]
    history1_epoch = len(history1.epoch) 

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=10)

    #Model 2, Sigmoid, Layer 1 = 3xInput, Layer 2 = 1/4 Layer1, Output 2 Nodes for Classification
    nn2 = Sequential()
    nn2.add(Input((10,)))
    nn2.add(Dense(30,activation="sigmoid"))
    nn2.add(Dense(8,activation="sigmoid"))
    nn2.add(Dense(2,activation="softmax"))

    nn2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history2 = nn2.fit(x_train, y_train, epochs=1000,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history2.history["accuracy"] 
    #history2.history["val_accuracy"]
    history2_epoch = len(history2.epoch) 

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=20)

    #Model 3, Leaky Relu, Layer 1 = 2xInput, Layer 2 = 1/3 Layer1, Output 2 Nodes for Classification
    nn3 = Sequential()
    nn3.add(Input((10,)))
    nn3.add(Dense(20,activation="leaky_relu"))
    nn3.add(Dense(7,activation="leaky_relu"))
    nn3.add(Dense(2,activation="softmax"))

    nn3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history3 = nn3.fit(x_train, y_train, epochs=600,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history3.history["accuracy"] 
    #history3.history["val_accuracy"]
    history3_epoch = len(history3.epoch) 

    #Reset Patience
    earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=20)
    
    #Model 4, Relu, Layer 1 = 3xInput, Layer 2 = 1/4 Layer1, Output 2 Nodes for Classification
    nn4 = Sequential()
    nn4.add(Input((10,)))
    nn4.add(Dense(30,activation="leaky_relu"))
    nn4.add(Dense(8,activation="leaky_relu"))
    nn4.add(Dense(2,activation="softmax"))

    nn4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history4 = nn4.fit(x_train, y_train, epochs=600,validation_data=(x_val,y_val),callbacks=[earlystop],verbose=1)
    #history3.history["accuracy"] 
    #history3.history["val_accuracy"]
    history4_epoch = len(history4.epoch) 

    #Plot Results
    #Model 1
    plt.title("Magic Telescope Dataset Model 1, Sigmoid, 2x Input")
    plt.plot(history1.history["accuracy"], label="Training")
    plt.plot(history1.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 2
    plt.title("Magic Telescope Dataset Model 2, Sigmoid, 3x Input")
    plt.plot(history2.history["accuracy"], label="Training")
    plt.plot(history2.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 3
    plt.title("Magic Telescope Dataset Model 3, Leaky Relu, 2x Input")
    plt.plot(history3.history["accuracy"], label="Training")
    plt.plot(history3.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Model 4
    plt.title("Magic Telescope Dataset Model 4, Leaky Relu, 3x Input")
    plt.plot(history3.history["accuracy"], label="Training")
    plt.plot(history3.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim(0,100)
    plt.legend()
    plt.show()

    #Report Maximum Accuaracy
    accuracy_df = pandas.DataFrame(columns=["Model","Trainging Epochs","Maximum Accuracy"])
    accuracy1 = ["Model 1, Sigmoid, 2x Input", history1_epoch , max(history1.history["val_accuracy"])]
    accuracy2 = ["Model 2, Sigmoid, 3x Input", history2_epoch, max(history2.history["val_accuracy"])]
    accuracy3 = ["Model 3, Leaky Relu, 2x Input", history3_epoch, max(history3.history["val_accuracy"])]
    accuracy4 = ["Model 4, Leaky Relu, 3x Input", history4_epoch, max(history4.history["val_accuracy"])]

    accuracy_df.loc[len(accuracy_df)] = accuracy1
    accuracy_df.loc[len(accuracy_df)] = accuracy2
    accuracy_df.loc[len(accuracy_df)] = accuracy3
    accuracy_df.loc[len(accuracy_df)] = accuracy4

    accuracy_df = accuracy_df.round(4)

    print(accuracy_df)



regression_network()

classification_network()