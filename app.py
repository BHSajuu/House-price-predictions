from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_rooms = float(request.form["total_rooms"])
    total_bedrooms = float(request.form["total_bedrooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]

    # Create a pandas DataFrame from the input data
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

    # Preprocess the input data using the loaded pipeline
    input_data_prepared = pipeline.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_prepared)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ${prediction[0]:,.2f}",
    )

if __name__ == "__main__":
    app.run(debug=True)