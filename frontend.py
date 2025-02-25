import streamlit as st
import numpy as np
import pickle  # To load the trained model

# Load the trained diabetes prediction model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def predict_diabetes(features):
    """Make a prediction using the loaded model."""
    features = np.array(features).reshape(1, -1)  # Convert input to correct format
    prediction = model.predict(features)[0]
    return "diabetic" if prediction == 1 else "not diabetic"

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter your medical details below to check the risk of diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    prediction = predict_diabetes(features)
    
    st.subheader("Result:")
    if prediction.lower() == "diabetic":
        st.error("🚨 High risk of diabetes detected!")
    else:
        st.success("✅ Low risk of diabetes.")
