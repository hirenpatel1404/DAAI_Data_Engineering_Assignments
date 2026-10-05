# app.py
# I create a small Flask web app that loads my saved model and returns predictions

import pickle
import numpy as np
from flask import Flask, request, jsonify

#I CREATE THE fLASK APPLICATION
app = Flask(__name__)

#i LOAD THE TRAINED MODEL ONCE , WHEN trhe app starts
with open ('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# The four input names my API expects, in the same order the model was trained on
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# The species name for each class number
species_names = ["setosa", "versicolor", "virginica"]

# I define the /predict address, which only accepts POST requests
@app.route("/predict", methods=["POST"])
def predict():
    # I read the JSON data that was sent with the request
    data = request.get_json(silent=True)

    # If no JSON was sent, I reply with a clear error message
    if data is None:
        return jsonify({"error": "Please send the flower measurements as JSON."}), 400

    # I check that all four measurements are present
    missing = [name for name in feature_names if name not in data]
    if missing:
        return jsonify({"error": "Missing fields: " + ", ".join(missing)}), 400

    # I check that every measurement is a number
    try:
        values = [float(data[name]) for name in feature_names]
    except (TypeError, ValueError):
        return jsonify({"error": "All four measurements must be numbers."}), 400

    # I put the four values into the shape the model expects: one row, four columns
    features = np.array([values])

    # I ask the model for its prediction
    prediction = int(model.predict(features)[0])

    # I send back the class number and the species name
    return jsonify({"prediction": prediction, "species": species_names[prediction]})

# I start the web server when I run this file directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)