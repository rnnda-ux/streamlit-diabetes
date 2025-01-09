import streamlit as st
import pickle
import numpy as np
import os


with open('fish.pkl', 'rb') as model_file:
    data = pickle.load(model_file)
    loaded_model = data['model']
    scaler = data['scaler']



st.title('Prediksi Spesies Ikan')

length = st.number_input('Panjang (length):', min_value=0)
weight = st.number_input('Berat (weight):', min_value=0)
w_l_ratio = st.number_input('Rasio Panjang-Lebar (w_l_ratio):', min_value=0)


if st.button('Prediksi Spesies'):
    features = np.array([[length, weight, w_l_ratio]])
    scaled_features = scaler.transform(features)  # Scaling data input
    species_prediction = loaded_model.predict(scaled_features)[0]
    st.success(f'Spesies yang Diprediksi: {species_prediction}')
