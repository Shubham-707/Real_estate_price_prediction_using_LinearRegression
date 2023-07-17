import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the saved model from the pickle file
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))

# Load the column names for feature encoding
with open('columns.json', 'r') as f:
    data_columns = json.load(f)
    columns = data_columns['data_columns']

# Function to predict the house price using the loaded model
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(np.array(columns) == location)[0][0]

    # Create a feature vector with all zeros
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Set the corresponding location index to 1 to handle one-hot encoding
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict the price using the loaded model
    return model.predict([x])[0]

# Streamlit app definition
def main():
    st.title('Real Estate Price Prediction')

    # Input features for prediction
    location = st.selectbox('Location', columns)
    sqft = st.number_input('Total Square Feet')
    bath = st.number_input('Number of Bathrooms', step=0.5)
    bhk = st.number_input('Number of Bedrooms', step=1)

    # Prediction button
    if st.button('Predict'):
        result = predict_price(location, sqft, bath, bhk)
        st.success(f'Estimated House Price: ₹ {result:.2f} Lakhs')

if __name__ == '__main__':
    main()
