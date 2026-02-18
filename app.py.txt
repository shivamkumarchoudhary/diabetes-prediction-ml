import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")
st.write("Enter patient details below:")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
