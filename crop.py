import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Loading data
data = pd.read_csv('Crop_recommendation.csv')
print(data.columns)  # Print the columns to check the names
data.head()

# Checking for null values
data.isnull().sum()

# Data visualization
data['label'].value_counts().plot(kind='bar')

# Plotting nutrition in soil
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
plt.title('Nitrogen')
plt.hist(data['Nitrogen'], color='green')
plt.subplot(2,3,2)
plt.title('Phosphorus')
plt.hist(data['phosphorus'], color='red')
plt.subplot(2,3,3)
plt.title('Potassium')
plt.hist(data['potassium'], color='blue')
plt.subplot(2,3,4)

# Splitting data into train and test sets
train, test = train_test_split(data, test_size=0.25, random_state=0)

# Encoding labels
le = LabelEncoder()
train['label'] = le.fit_transform(train['label'])
test['label'] = le.fit_transform(test['label'])

# Preparing data for training
train_x = train.drop(columns=['label'])
train_y = train['label']
test_x = test.drop(columns=['label'])
test_y = test['label']

# Check if the model file exists
if os.path.exists('crop.h5'):
    # Load the model
    model = keras.models.load_model('crop.h5')
    print("Model loaded from disk")
else:
    # Building model
    model = keras.Sequential([
        keras.layers.Dense(11, input_shape=(7,), activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='softmax'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(22, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=16, epochs=250)
    
    # Save the model
    model.save('crop.h5')
    print("Model saved to disk")

# Testing model
test_loss, test_acc = model.evaluate(test_x, test_y)
print("Tested Acc:", test_acc)

# Predicting
n = int(input("Enter nitrogen:"))
p = int(input("Enter phosphorus:"))
k = int(input("Enter potassium:"))
t = int(input("Enter temperature:"))
ph = int(input("Enter ph:"))
h = int(input("Enter humidity:"))
r = int(input("Enter rainfall:"))

# Convert the input list to a NumPy array
input_data = np.array([[n, p, k, t, ph, h, r]])

# Make prediction
prediction = model.predict(input_data)
prediction = np.argmax(prediction)
print(le.inverse_transform([prediction]))

# Accuracy of prediction made by model
prediction = model.predict(test_x)
prediction = np.argmax(prediction, axis=1)
accuracy = accuracy_score(test_y, prediction)
print(accuracy)

# Check if the model file does not exist before saving
if not os.path.exists('crop.h5'):
    # Saving model
    model.save('crop.h5')
    print("Saved model to disk")