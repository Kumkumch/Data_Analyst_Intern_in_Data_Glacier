from fastapi import FastAPI
import joblib
import numpy as np

# Create the FastAPI app instance
app = FastAPI()

# Load the trained model
model = joblib.load("iris_model.pkl")

# Define the species names for better readability
species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API!"}

@app.get("/predict/{sepal_length}/{sepal_width}/{petal_length}/{petal_width}")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # Prepare input data for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the species
    prediction = model.predict(input_data)
    species = species_names[prediction[0]]  # Get species name from prediction

    return {"predicted_species": species}


