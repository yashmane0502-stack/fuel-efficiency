

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Fuel_Efiiciency_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Fuel efiiciency Prediction app")
mpg= st.number_input(mpg)

cylinders= st.number_input(" cylinders ")
displacement= st.number_input("displacement")
horsepower= st.number_input(" horsepower")
weight= st.number_input("weight")
acceleration= st.number_input(" acceleration")
model_year= st.number_input("enter model year ")
origin= st.number_input(" origin ")
car name= st.selectbox("car name", encoder["car name"].classes_)

df = pd.DataFrame({
    "mpg":[mpg],
    "cylinders":[cylinders],
    "displacement":[displacement],
    "horsepower":[horsepower],
    "weight":[weight],
    "acceleration":[acceleration],
    "model year":[model_year],
    "origin":[origin],
    "car name":[car name]
})

if st.button("Predict"):

    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    df = df[model.feature_names_in_]

    prediction = model.predict(df)

    st.success(f"Predicted fuel eficiency: {prediction[0]:,.2f}")
