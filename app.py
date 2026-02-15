import streamlit as st
import joblib
model=joblib.load("Revenue.pkl")
st.title("Cold stone Sales ")
st.header("Revenue Prediction")
user_int=st.number_input("Enter the Temperature")
if st.button("Predict Revenue"):
    st.write(model.predict([[user_int]]))
