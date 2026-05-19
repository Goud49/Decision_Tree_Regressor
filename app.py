import streamlit as st
import pickle
import numpy as np
import os


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# -------------------------
# Check Files
# -------------------------
if not os.path.exists("model.pkl"):

    st.error(
        "model.pkl file not found"
    )

    st.stop()


if not os.path.exists("scaler.pkl"):

    st.error(
        "scaler.pkl file not found"
    )

    st.stop()


# -------------------------
# Load Model
# -------------------------
with open(
    "model.pkl",
    "rb"
) as file:

    model = pickle.load(
        file
    )


# -------------------------
# Load Scaler
# -------------------------
with open(
    "scaler.pkl",
    "rb"
) as file:

    scaler = pickle.load(
        file
    )


# -------------------------
# Title
# -------------------------
st.title(
    "🏠 House Price Prediction"
)

st.write(
    "Predict house prices using Decision Tree Regressor."
)


# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)


with col1:

    feature1 = st.number_input(
        "Feature 1",
        value=10.0
    )

    feature2 = st.number_input(
        "Feature 2",
        value=20.0
    )


with col2:

    feature3 = st.number_input(
        "Feature 3",
        value=30.0
    )

    feature4 = st.number_input(
        "Feature 4",
        value=40.0
    )


# -------------------------
# Prediction
# -------------------------
if st.button(
    "Predict Price"
):

    input_data = np.zeros(
        (1, model.n_features_in_)
    )


    input_data[0][0] = feature1
    input_data[0][1] = feature2
    input_data[0][2] = feature3
    input_data[0][3] = feature4


    input_data = scaler.transform(
        input_data
    )


    prediction = model.predict(
        input_data
    )


    st.markdown("---")


    st.success(
        f"Predicted Price: ₹ {prediction[0]:,.2f}"
    )
