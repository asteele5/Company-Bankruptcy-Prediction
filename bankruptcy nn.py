# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 23:51:07 2021

@author: mrswa
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import math
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler



# Train Data
fileTrainName = 'company bankruptcy prediction shortened.csv'
print("fileTrainName: ", fileTrainName)
print()
raw_train_data = open(fileTrainName, 'rt')
trainData = np.loadtxt(raw_train_data, usecols = (range(51)), skiprows = 1, delimiter=",")


numEpochs = 100

x = trainData[:,0:50]
y = trainData[:,50]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.75,test_size=0.25, random_state=101)

sc = StandardScaler()
# Standardize the training dataset
#(and calculate the mean and standard deviation)
x_train = sc.fit_transform(x_train)
#Use this mean and standard deviation
#calculated in the training dataset to
#standardize the test dataset
x_test = sc.transform (x_test)


model = keras.Sequential()
model.add(layers.Dense(16, activation='relu', 
 input_shape = (x_train.shape[1],) ))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
model.summary()
history = model.fit(x_train, y_train,
                        epochs=numEpochs, batch_size=80, validation_data=(x_test, y_test))

mae_history = history.history['mae']

test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

predicted_bankruptcy = model.predict(x_test)


print('test mse: ',test_mse_score)
print('test mae: ',test_mae_score)

print('train average mae: ',sum(mae_history)/len(mae_history))

#2 Verify accuracy of test_mae_score, returned by model.evaluate()

predicted_bankruptcy = predicted_bankruptcy.flatten()
predicted_bankruptcy[:][predicted_bankruptcy[:] >= 0.5]=1
predicted_bankruptcy[:][predicted_bankruptcy[:] < 0.5]=0
absoluteValue = abs(predicted_bankruptcy-y_test)

test_mae_score_calculated = sum(absoluteValue)/len(absoluteValue)

print('test mae calculated: ',test_mae_score_calculated)


val_loss = history.history['val_loss']
#plt.plot(range(1,numEpochs+1),mae_history, 'ro')
#plt.plot(range(1,numEpochs+1), val_loss, 'b', label='Validation loss')


fig, ax = plt.subplots()
ax.plot(np.arange(len(mae_history)), mae_history, 'bo', label='Training loss')
# b is for "solid blue line"
ax.plot(np.arange(len(mae_history)), val_loss, 'red', label='Validation loss')


ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation loss');
ax.legend()

plt.title('MAE Graph')
plt.xlabel('Epochs')
plt.ylabel('MAE')
def calcConfusionMatrix(P,Y):
  totalN = np.sum(P==0)
  totalP = np.sum(P==1)
  TP = P[Y==1]==1
  TPC = np.sum(TP)
  
  FP = P[Y==0]==1
  FPC = np.sum(FP)
  FN = P[Y==1]==0
  FNC = np.sum(FN)
  TN = P[Y==0]==0
  TNC = np.sum(TN)
  precision = TPC/(TPC + FPC)
  recall = TPC/(TPC + FNC)
  f1 = (2*precision * recall)/(precision+recall)
#TPC is the variable for the true positive count
  print ("\nTrue Positives: \t", TPC, "\nPercent Positives correct:", f"{(TPC/totalP):.0%}")
  print ("False Positives: \t", FPC, "\nPercent Positives Incorrect:", f"{(FPC/totalP):.0%}")
  print ("True Negatives: \t", TNC, "\nPercent Negatives correct:", f"{(TNC/totalN):.0%}")
  print ("False Negatives: \t", FNC, "\nPercent Negatives incorrect:", f"{(FNC/totalN):.0%}")

  print ("\nPrecision: \t\t",f"{precision:.0%}")
  print ("Recall:\t\t\t",f"{recall:.0%}")
  print ("\nF1: \t\t\t",f"{f1:.0%}")

  return TPC,TNC,FPC,FNC,precision,recall,f1

TPC,TNC,FPC,FNC,precision,recall,f1 = calcConfusionMatrix(predicted_bankruptcy,y_test)
print('accuracy: ', (TPC+TNC)/(TPC+TNC+FNC+FPC))
