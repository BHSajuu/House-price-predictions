from flask import Flask, request, render_template
import joblib
import pandas as pd
import os 

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

pipeline_path = os.path.join(BASE_DIR, "pipeline.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_rooms = float(request.form["total_rooms"])
    total_bedrooms = float(request.form["total_bedrooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]

    input_data = pd.DataFrame(
        {
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity],
        }
    )

    input_data_prepared = pipeline.transform(input_data)

    prediction = model.predict(input_data_prepared)

    prediction_text = f"Predicted House Price: ${prediction[0]:,.2f}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
    )