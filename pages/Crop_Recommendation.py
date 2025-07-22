import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/crop_recommendation_model.pkl"
model = pickle.load(open(model_path, 'rb'))

# Streamlit UI
st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and climate conditions to get the best crop recommendation.")

# Input form
with st.form("crop_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        temperature = st.number_input("Temperature (°C)", format="%.2f", value=25.0)

    with col2:
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
        humidity = st.number_input("Humidity (%)", format="%.2f", value=60.0)

    with col3:
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
        ph = st.number_input("pH", format="%.2f", min_value=3.5, max_value=9.5, value=6.5)

    rainfall = st.number_input("Rainfall (mm)", format="%.2f", value=100.0)

    submitted = st.form_submit_button("Predict Crop")

# Prediction
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop = prediction[0]
    print(crop)
    st.success(f"✅ Recommended Crop: **{crop}**")
