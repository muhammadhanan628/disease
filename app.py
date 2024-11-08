import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import joblib

# Load models
try:
    final_svm_model = joblib.load("model/svm_model.joblib")
    final_rf_model = joblib.load("model/rf_model.joblib")
    final_nb_model = joblib.load("model/nb_model.joblib")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    raise e

# Load dataset and prepare data structures
DATA_PATH = "data/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encoding the target value into numerical values
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Creating a symptom index dictionary
symptoms = X.columns.values
symptom_index = {symptom.replace("_", " ").title(): idx for idx, symptom in enumerate(symptoms)}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Define prediction function
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip()  # Remove extra spaces
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized and will be ignored.")
    
    # Convert to array format for model input
    input_data = np.array(input_data).reshape(1, -1)
    
    # Generate individual model predictions
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]

    # Final prediction by majority vote
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Prediction API"}

@app.get("/predict")
async def predict_disease(symptoms: str):
    try:
        results = predictDisease(symptoms)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


