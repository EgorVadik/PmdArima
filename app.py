import pandas as pd
import joblib
from flask import Flask, request, jsonify

model = joblib.load('arima_model.pkl')


def make_predictions(input_data):
    try:
        new_data = pd.DataFrame(input_data)
        if new_data.shape[1] != 1:
            raise ValueError("Input data must have exactly one column")

        predictions = model.predict(n_periods=len(new_data))
        return predictions.tolist()

    except Exception as e:
        return {"error": str(e)}


app = Flask(__name__)


@app.route("/")
def index():
    return "ARIMA Model API"


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
