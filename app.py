from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Path to the CSV file
csv_file_path = 'Crop_recommendation.csv'

def train_model():
    # Read the entire dataset
    data = pd.read_csv(csv_file_path)

    # Print the columns to check the names
    print("Columns in the dataset:", data.columns)

    # Features and target
    X = data[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and label encoder
    joblib.dump(model, 'crop_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("Model and label encoder trained and saved.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Load the trained model and label encoder
    model = joblib.load('crop_model.pkl')
    le = joblib.load('label_encoder.pkl')

    # Make prediction with the latest input data
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    predicted_label = le.inverse_transform(prediction)

    # Display the prediction result on the webpage
    return render_template('index.html', prediction=predicted_label[0])

if __name__ == '__main__':
    train_model()  # Train the model when the application starts
    app.run(debug=True)