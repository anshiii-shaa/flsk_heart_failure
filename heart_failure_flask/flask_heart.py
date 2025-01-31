import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import gradio as gr
import numpy as np


from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

def predict_heart_diseas(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    # Initialize LabelEncoders
    sex_encoder = LabelEncoder()
    sex_encoder.classes_ = np.array(["F", "M"])

    chest_pain_encoder = LabelEncoder()
    chest_pain_encoder.classes_ = np.array(["ATA", "NAP", "ASY", "TA"])

    resting_ecg_encoder = LabelEncoder()
    resting_ecg_encoder.classes_ = np.array(["Normal", "ST", "LVH"])

    exercise_angina_encoder = LabelEncoder()
    exercise_angina_encoder.classes_ = np.array(["N", "Y"])

    st_slope_encoder = LabelEncoder()
    st_slope_encoder.classes_ = np.array(["Up", "Flat", "Down"])

    # Encode categorical features
    sex_encoded = sex_encoder.transform([sex])[0]
    chest_pain_encoded = chest_pain_encoder.transform([chest_pain])[0]
    resting_ecg_encoded = resting_ecg_encoder.transform([resting_ecg])[0]
    exercise_angina_encoded = exercise_angina_encoder.transform([exercise_angina])[0]
    st_slope_encoded = st_slope_encoder.transform([st_slope])[0]

    # Create input data array
    input_data = [
        age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bs,
        resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded
    ]

    # Reshape input data for prediction
    input_data = [input_data]

    # Load the model
    with open("randomforest_model.pkl", "rb") as file:
        load_model = pickle.load(file)

    # Make prediction
    prediction = load_model.predict(input_data)[0]

    if prediction ==0:
        return f"Yes"
    else:
        return f"No"

iface = gr.Interface(
    fn=predict_heart_diseas,
    inputs=[
        gr.Number(label= "age"),
        gr.Text(label="Sex(F,M)"),
        gr.Text(label="chest_pain(ATA,NAP,ASY,TA)"),
        gr.Number(label="resting_bp"),
        gr.Number(label="cholosterol"),
        gr.Number(label="fasting_bs"),
        gr.Text(label="resting_ecg(Normal,ST,LVH)"),
        gr.Number(label="max_hr"),
        gr.Text(label="exercise_angina(N,Y)"),
        gr.Number(label="oldpeak"),
        gr.Text(label="st_slope(Up,Flat,Down)")
    ],
    outputs="text",
    title="Iris Flower Classifier with Voting Ensemble",
    description = "Enter the flower's measurements to predict its class using a voting Classfier"
)
iface.launch()



