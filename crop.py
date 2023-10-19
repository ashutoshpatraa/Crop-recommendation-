#project using tensorflow library to make ai model that recommend which crop tp grow depending on soil nutrition and weather condition(file for data is Crop_recommendation.csv)
 #importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#loading data
data=pd.read_csv('Crop_recommendation.csv')
data.head()

#checking for null values
data.isnull().sum()

#data visualization
data['label'].value_counts().plot(kind='bar')

#plotting nutrition in soil
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
plt.title('Nitrogen')
plt.hist(data['N'],color='green')
plt.subplot(2,3,2)
plt.title('Phosphorus')
plt.hist(data['P'],color='red')
plt.subplot(2,3,3)
plt.title('Potassium')
plt.hist(data['K'],color='blue')
plt.subplot(2,3,4)


#splitting data into train and test to 75% and 25% respectively
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.25,random_state=0)

#pandas library to convert string to number
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['label']=le.fit_transform(train['label'])
test['label']=le.fit_transform(test['label'])

#making data ready for training
#we are droping
train_x=train.drop(columns=['label'])
train_y=train['label']
test_x=test.drop(columns=['label'])
test_y=test['label']

#making model
model=keras.Sequential([
    keras.layers.Dense(11,input_shape=(7,),activation='relu'),
    keras.layers.Dense(8,activation='relu'),
    keras.layers.Dense(8,activation='softmax'),
    keras.layers.Dense(8,activation='relu'),
    keras.layers.Dense(22,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=16,epochs=250)

#testing model
test_loss,test_acc=model.evaluate(test_x,test_y)
print("Tested Acc:",test_acc)



#predicting

n=int(input("Enter nitrogen:"))
p=int(input("Enter phosphorus:"))
k=int(input("Enter potassium:"))
t=int(input("Enter temperature:"))
ph=int(input("Enter ph:"))
h=int(input("Enter humidity:"))
r=int(input("Enter rainfall:"))

prediction=model.predict([[n,p,k,t,ph,h,r]])
prediction=np.argmax(prediction)
print(le.inverse_transform([prediction]))

#accuracy of prediction made by model
from sklearn.metrics import accuracy_score
prediction=model.predict(test_x)
prediction=np.argmax(prediction,axis=1)
accuracy=accuracy_score(test_y,prediction)
print(accuracy)


#saving model
model.save('crop.h5')
print("Saved model to disk")

==================================================================================================================================================================