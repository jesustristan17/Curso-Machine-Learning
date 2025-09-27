# app.py
import streamlit as st
import numpy as np
import joblib

# Cargar modelo
model = joblib.load("modelo_vino.pkl")

st.title("Predicción de Calidad de Vino ")

# Inputs del usuario
inputs = []
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.3f")
    inputs.append(value)

# Predicción
if st.button("Predecir"):
    datos = np.array(inputs).reshape(1, -1)
    pred = model.predict(datos)[0]
    prob = model.predict_proba(datos)[0][1]

    if pred == 1:
        st.success(f"Vino de alta calidad (probabilidad: {prob:.2f})")
    else:
        st.warning(f"Vino de calidad estándar (probabilidad: {prob:.2f})")
