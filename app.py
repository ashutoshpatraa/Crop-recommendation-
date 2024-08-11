from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'crop_recommendation_model.pkl')

# Train the model when the application starts
train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Load the trained model
    model = joblib.load('crop_recommendation_model.pkl')

    # Get form data
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                              columns=['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Print the prediction to the console
    print(f"Recommended Crop: {prediction}")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)