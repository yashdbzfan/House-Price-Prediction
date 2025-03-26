import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    with open('best_model.pkl', 'rb') as file:
        lr_pipeline = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("model.pkl not found. Please ensure the pre-trained model is in the project directory.")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'mainroad': request.form['mainroad'],
            'guestroom': request.form['guestroom'],
            'basement': request.form['basement'],
            'hotwaterheating': request.form['hotwaterheating'],
            'airconditioning': request.form['airconditioning'],
            'parking': int(request.form['parking']),
            'prefarea': request.form['prefarea'],
            'furnishingstatus': request.form['furnishingstatus']
        }

        # Calculate derived features
        data['area_per_bedroom'] = data['area'] / max(data['bedrooms'], 1)
        data['total_rooms'] = data['bedrooms'] + data['bathrooms']

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Predict (in log scale)
        log_pred = lr_pipeline.predict(input_df)

        # Inverse transform to get actual price
        price = np.expm1(log_pred[0])
        price = round(price, 2)

        return render_template('result.html', price=f"{price:,.2f}")
    except Exception as e:
        return render_template('result.html', price=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)