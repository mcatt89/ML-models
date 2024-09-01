# Matt Herman
# 4/22/2024
# All code within is my own work.


import numpy as np
import pandas
import sklearn as skl
import matplotlib 
import keras
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras import callbacks
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer 
from keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
from keras.applications import EfficientNetB6
from tensorflow import data


training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))
test_set = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)


cnn1 = Sequential()
#Input layer for 100 x 100 pixel rgb color images
cnn1.add(Input((100,100,3)))
#rescale colors to 0 to 1
cnn1.add(Rescaling(1/255))
#Convolution and Pooling Layers
#cnn1.add(keras.layers.Conv2D(64, (3, 3), activation="leaky_relu"))
cnn1.add(Conv2D(32, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(64, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
#Flatten to 1D
cnn1.add(Flatten())
#Dense Layer
cnn1.add(Dense(512, activation="relu"))
#Dropout for overfitting
cnn1.add(Dropout(0.1))
#Output with Softmax activation for classification
cnn1.add(Dense(10, activation="softmax"))

#Compile network
cnn1.compile(loss="categorical_crossentropy", metrics=["accuracy"])

cnn1.summary()

history1 = cnn1.fit(training_set, epochs=20)
history1.history


cnn2 = Sequential()
#Input layer for 100 x 100 pixel rgb color images
cnn2.add(Input((100,100,3)))
#rescale colors to 0 to 1
cnn2.add(Rescaling(1/255))
#Convolution and Pooling Layers
#cnn2.add(keras.layers.Conv2D(64, (3, 3), activation="leaky_relu"))
cnn2.add(Conv2D(32, (3, 3), activation="relu"))
cnn2.add(AveragePooling2D(pool_size=(2, 2)))
cnn2.add(Conv2D(64, (3, 3), activation="relu"))
cnn2.add(AveragePooling2D(pool_size=(2, 2)))
#Dropout for overfitting
cnn2.add(Conv2D(128, (3, 3), activation="relu"))
cnn2.add(AveragePooling2D(pool_size=(2, 2)))
#Flatten to 1D
cnn2.add(Flatten())
#Dense Layer
cnn2.add(Dense(1024, activation="relu"))
#Dropout for overfitting
cnn2.add(Dropout(0.1))
#Dense Layer
cnn2.add(Dense(512, activation="relu"))
#Dropout for overfitting
cnn2.add(Dropout(0.1))
#Output with Softmax activation for classification
cnn2.add(Dense(10, activation="softmax"))

#Compile network
cnn2.compile(loss="categorical_crossentropy", metrics=["accuracy"])

cnn2.summary()

history2 = cnn2.fit(training_set, epochs=20)
history2.history

#Score Models
score1 = cnn1.evaluate(test_set)
print("CNN 1 Test Accuracy:", score1[1])

score2 = cnn2.evaluate(test_set)
print("CNN 2 Test Accuracy:", score2[1])

if score1[1] > score2[1]:
    print("CNN1 has a higher accuracy")
    final_cnn = cnn1
    final_history = history1
    final_score = score1
else:
    print("CNN2 has a higher accuracy")
    final_cnn = cnn2
    final_history = history2
    final_score = score2

final_cnn.summary()
final_history.history
final_score[1]

corr = []
for e in test_set :
    corr = corr + list(np.argmax(e[1],axis=1))

p = final_cnn.predict(test_set)
pr = np.argmax(p, axis=1)
confusion_matrix(corr, pr)

final_cnn.save("final_model.keras")

incorrect_set = np.where(pr != corr)[0]

output_csv = 'Final_CNN_Errors.csv'
output_path = os.path.join(os.getcwd(), output_csv)

for i in incorrect_set:
    print(f"Example {i}: Predicted = {pr[i]}, True = {corr[i]}")
    print(f"Example {i} File Pats: {test_set.file_paths[i]}")
    #export_row = [i,pr[i],corr[i],test_set.file_paths[i]]
    #export_df = pandas.DataFrame([export_row], columns=['Index','Predicted','True','File Path'])
    #export_df.to_csv(output_path, mode='a', header=False, index=False)


pre_trained = EfficientNetB6(include_top=False)
ft = pre_trained.output
ft = GlobalAveragePooling2D()(ft)
ft = Dense(1024, activation="relu")(ft)
ft = Dropout(0.2)(ft)
ft_output = Dense(10, activation="softmax")(ft)
ft_model = Model(inputs=pre_trained.input, outputs=ft_output)

for layer in pre_trained.layers:
    layer.trainable = False

ft_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

ft_model.summary()

ft_history = ft_model.fit(training_set, epochs=35)
ft_history.history

ft_score = ft_model.evaluate(test_set)
print("Fine Tuned Model Test Accuracy:", ft_score[1])


corr = []
for e in test_set :
    corr = corr + list(np.argmax(e[1],axis=1))


ft_p = ft_model.predict(test_set)
ft_pr = np.argmax(ft_p, axis=1)
ft_cm = confusion_matrix(corr, ft_pr)
ft_cm

ft_model.save("fine_tuned.keras")

ft_p
ft_pr
incorrect_set = np.where(ft_pr != corr)[0]

print("Incorrect examples:", incorrect_set)

output_csv = 'FT_Model_Errors.csv'
output_path = os.path.join(os.getcwd(), output_csv)

for i in incorrect_set:
    print(f"Example {i}: Predicted = {ft_pr[i]}, True = {corr[i]}")
    print(f"Example {i} File Pats: {test_set.file_paths[i]}")
    #export_row = [i,ft_pr[i],corr[i],test_set.file_paths[i]]
    #export_df = pandas.DataFrame([export_row], columns=['Index','Predicted','True','File Path'])
    #export_df.to_csv(output_path, mode='a', header=False, index=False)



#CNN1 Learning Curve
plt.title("CNN1 Learning Curve")
plt.plot(history1.history["accuracy"], label="Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.ylim(0,100)
plt.legend()
plt.show()

#CNN2 Learning Curve
plt.title("CNN2 Learning Curve")
plt.plot(history2.history["accuracy"], label="Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.ylim(0,100)
plt.legend()
plt.show()

#FTM Learning Curve
plt.title("Fine Tuned Model Learning Curve")
plt.plot(ft_history.history["accuracy"], label="Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.ylim(0,100)
plt.legend()
plt.show()

