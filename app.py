import streamlit as st
import pickle
import numpy as np
import os


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


if not os.path.exists("model.pkl"):
    st.error("model.pkl file not found")
    st.stop()


if not os.path.exists("scaler.pkl"):
    st.error("scaler.pkl file not found")
    st.stop()


with open(
    "model.pkl",
    "rb"
) as file:

    model = pickle.load(
        file
    )


with open(
    "scaler.pkl",
    "rb"
) as file:

    scaler = pickle.load(
        file
    )


st.title(
    "🏠 House Price Prediction"
)

st.write(
    "Predict house price using Decision Tree Regressor."
)


col1, col2 = st.columns(2)


with col1:

    overall_qual = st.number_input(
        "Overall Quality",
        value=5
    )

    gr_liv_area = st.number_input(
        "Ground Living Area",
        value=1500
    )


with col2:

    garage_cars = st.number_input(
        "Garage Cars",
        value=2
    )

    total_bsmt_sf = st.number_input(
        "Basement Area",
        value=1000
    )


if st.button(
    "Predict Price"
):

    input_data = np.zeros(
        (1, model.n_features_in_)
    )


    input_data[0][17] = overall_qual
    input_data[0][46] = total_bsmt_sf
    input_data[0][61] = gr_liv_area
    input_data[0][81] = garage_cars


    input_data = scaler.transform(
        input_data
    )


    prediction = model.predict(
        input_data
    )


    st.success(
        f"Predicted House Price: ₹ {prediction[0]:,.2f}"
    )