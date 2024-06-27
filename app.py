import pmdarima as pm
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

# Load your training data (replace with your actual data loading logic)
data_path = "Gretel.csv"  # Replace with your data path
x = pd.read_csv(data_path)
df = pd.DataFrame(x)

# Split data into training and testing sets
train, test = df[:160], df[160:]

# Create the PMDARIMA model
m = pm.auto_arima(
    train, error_action="ignore", seasonal=True, m=12, D=1
)  # Enable trace for model diagnostics

# Function to generate predictions for new data
def make_predictions(new_data):
    try:
        # Ensure new_data is a pandas DataFrame
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        # Check if new data has the same columns as the training data
        if not set(new_data.columns) == set(train.columns):
            raise ValueError("Input data must have the same columns as the training data")

        # Make predictions using the trained model
        predictions = m.predict(n_periods=len(new_data))
        return predictions.tolist()  # Convert to list for API response

    except Exception as e:
        return {"error": str(e)}

# Create a Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get new data from the request
    try:
        new_data = request.get_json()
        predictions = make_predictions(pd.DataFrame(new_data))

        # Return the predictions
        response = {"predictions": predictions}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return 400 for bad requests

# Host the Flask app on a cloud platform (instructions vary by platform)
# Here's an example using Heroku (replace with your preferred platform):

# 1. Create a Procfile (optional, for automatic deployment):
#    web: gunicorn app:app

# 2. Deploy your app to Heroku using Git:
#    - Create a Git repository for your code
#    - Push your code to Heroku
#    - Follow Heroku's deployment instructions

# Once deployed, you can access your API endpoint at the provided URL.

# Example usage (assuming your API is deployed on Heroku):
# import requests

# url = "https://your-app-name.herokuapp.com/predict"  # Replace with your app's URL
# new_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}  # Example new data
# response = requests.post(url, json=new_data)

# if response.status_code == 200:
#     predictions = response.json()["predictions"]
#     print("Predictions:", predictions)
#     # Access MAE and MAPE (if calculated) from the response if needed
# else:
#     print("Error:", response.json()["error"])
