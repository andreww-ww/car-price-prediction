import streamlit as st 
import pandas as pd
import joblib
import urllib.request
import os

@st.cache_resource
def loadModel():
    model = "random_search.pkl"
    model_url = "https://github.com/andreww-ww/car-price-prediction/releases/download/v1.0/random_search.pkl"

    if not os.path.exists(model):
        with st.spinner("---------------------- this takes about 30 seconds to load."):
            urllib.request.urlretrieve(model_url, model)
            
    return joblib.load(model)

@st.cache_data
def loadData():
    #Used for input selection
    return joblib.load('freq_dicts.pkl')

freq_dicts = loadData()
rdf = loadModel()

st.title("Car Resale Value Predictor")

st.write("Enter your car details to predict the resale value")
make=st.text_input("Select Make", value="Toyota").lower()

model=st.text_input("Enter Car Model", value="Camry").lower()


#0.0 for automatic 1.0 for manual
transmission = st.selectbox("Select Transmission",["Automatic", "Manual"]).lower()

trim=st.text_input("Enter Car Trim", value="LE").lower()

body=st.text_input("Enter Car Body", value="Sedan").lower()

ocolor=st.text_input("Enter Car Outside Color", value="Silver").lower()

icolor=st.text_input("Enter Car Interior Color", value="White").lower()

year = st.number_input("Enter Year", min_value=1990, max_value=2024, value=2015)

odometer = st.number_input("Enter Odometer (Mileage)", min_value=0, value=150000)

condition = st.number_input("Enter Condition (On a scale from 1-100)", min_value=1.0, max_value=100.0, value=42.0, step=1.0)



if st.button("Predict Resale Price"):

    seller = "Dudeman"
    state = "ca"

    # Converts bad/incorrect inputs to 1
    row_data = [[
            year, 
            condition, 
            odometer, 
            freq_dicts["make"].get(make, 1),
            freq_dicts["model"].get(model, 1),
            freq_dicts["trim"].get(trim, 1),
            freq_dicts["body"].get(body, 1),
            freq_dicts["transmission"].get(transmission, 1),
            freq_dicts["state"].get(state, 1),
            freq_dicts["color"].get(ocolor, 1),
            freq_dicts["interior"].get(icolor, 1),
            freq_dicts["seller"].get(seller, 1)
        ]]

    input_df = pd.DataFrame(
        row_data,
        columns=[
            "year", "condition", "odometer", "make_freq", "model_freq", "trim_freq", "body_freq", "transmission_freq",
            "state_freq", "color_freq", "interior_freq", "seller_freq"]
        )

    prediction = rdf.predict(input_df)

    st.success(f"Estimated Resale Value: ${prediction[0]:,.2f}")





