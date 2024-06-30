import pandas as pd
import joblib
import gradio as gr
from flask import Flask, request, jsonify

# Load the saved ARIMA model
model = joblib.load('arima_model.pkl')

# Function to generate predictions for new data


def make_predictions(input_data):
    try:
        # Convert input data to DataFrame
        new_data = pd.DataFrame(input_data)

        # Check if the new data has only one column
        if new_data.shape[1] != 1:
            raise ValueError("Input data must have exactly one column")

        # Make predictions using the trained model
        predictions = model.predict(n_periods=len(new_data))
        return predictions.tolist()  # Convert to list for API response

    except Exception as e:
        return {"error": str(e)}


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


# Create Gradio interface for the prediction function
interface = gr.Interface(fn=make_predictions, inputs=gr.Dataframe(
    headers=["column1"], datatype="number"), outputs="json")
interface.launch(share=True)

if __name__ == "__main__":
    app.run(port=5000)
