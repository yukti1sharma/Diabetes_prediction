import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Set the page title and layout
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load the saved model and scaler
with open('diabetes_classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and description
st.title("Diabetes Prediction")
st.write("Fill in the following parameters to know!!")


# Create columns for a structured layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure Level", min_value=0, max_value=122, value=0, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=0, step=1)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=0, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)

# Prediction button
if st.button("Let's find out!"):
    # Validation check
    if glucose == 0 or blood_pressure == 0 or bmi == 0.0 or age == 0:
        st.error("Please fill out all the required fields.")
    else:
        # Create an array from the input data
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
        input_data_reshaped = input_data.reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)

        # Make a prediction
        prediction = classifier.predict(std_data)

        # Display the result
        if prediction[0] == 0:
            st.write("### The person is not diabetic.")
        else:
            st.write("### The person is diabetic.")