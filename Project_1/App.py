import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# To show the title and display an text
st.title("🏠 House Price Prediction App")
st.write("Enter the house size to predict its price!")

# Load the model
@st.cache_resource #It will lode the model once and remember it
def load_model():
    model = joblib.load("house_price_model.pkl") #Loads your saved trained model
    return model #Returns the model

model = load_model()

# Now splitting the page into two parts
col1, col2 = st.columns(2)

# Use the left part of the area as the input part
with col1:
    st.subheader("📊 Input features") #Sub-heading

    # house_size is the variable that holds the value
    house_size = st.slider(
        "House Size(sq ft)", #text message
        min_value=500, #Slider min value
        max_value=3500, #Sliders max value
        value= 2000, #Slider starts from
        step = 50 #Steps it moves
    )

    # Dispaly the size which user selected
    st.metric("Selected size",f"{house_size} sq ft")

    # Creating a clickable button
    if st.button("🔮 Predicted Price", type="primary"): #it runs when the button is clicked
        prediction = model.predict([[house_size]]) #Use the model to predict the price
        predicted_price = prediction[0] # Gets the actual price value

        #  Success shows a green success message with the price
        st.success(f"💰 Predicted Price: ${predicted_price:,.2f}") #Formats number as US Dollars


# Now Right side column
with col2:
    st.subheader("📈 Model Information")

    st.write("**Model Information:**")
    st.write(f"- Coefficient (Slope): {model.coef_[0]:.2f}")
    st.write(f"- Intercept: {model.intercept_:.2f}")

    st.latex(f"Price = {model.coef_[0]:.2f} \\times Size + {model.intercept_:.2f}")

st.markdown("---")
st.markdown("**Built with Streamlit** | Machine Learning House Price Predictor")

