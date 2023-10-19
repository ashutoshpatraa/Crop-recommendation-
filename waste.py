#project using tensorflow library to make ai model that recommend which crop tp grow depending on soil nutrition and weather condition(file for data is Crop_recommendation.csv)
 #importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#loading data
data=pd.read_csv('Crop_recommendation.csv')
print(data.head())

#splitting data into train and test to 75% and 25% respectively
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.25,random_state=0)

print("train")
print(train.head())
print(test.head())



#from sklearn import preprocessing
#example code-
# le = preprocessing.LabelEncoder()
# le.fit(df.fruit)
# df['categorical_label'] = le.transform(df.fruit)

#pandas library to convert string to number
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le.fit(train['label'])
train['label']=le.transform(train['label'])

le1=LabelEncoder()
le1.fit(test['label'])
le1.fit(test['label'])
test['label']=le1.transform(test['label'])



#making data ready for training
train_x=train.drop(columns=['label'])
train_y=train['label']
test_x=test.drop(columns=['label'])
test_y=test['label']

#make model
model=keras.Sequential([
    keras.layers.Dense(11,input_shape=(7,),activation='relu'),
    keras.layers.Dense(8,activation='relu'),
    keras.layers.Dense(22,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#train model
model.fit(train_x,train_y,epochs=50)

#save model
model.save('crop.h5')
print("Saved model to disk")

