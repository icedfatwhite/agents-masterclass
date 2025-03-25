import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

# **Streamlit File Upload Interface**
st.title("Revenue Forecasting with Prophet")
st.subheader("Upload your Excel file with Date and Revenue columns")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

if uploaded_file:
    # Load Excel file into DataFrame
    df = pd.read_excel(uploaded_file)

    # Check if the required columns exist
    if 'Date' in df.columns and 'Revenue' in df.columns:
        # Prepare the data for Prophet
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
        df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})  # Prophet expects 'ds' and 'y'
        
        # Instantiate Prophet and fit the model
        model = Prophet()
        model.fit(df)

        # Make future predictions (let's forecast for the next 365 days)
        future = model.make_future_dataframe(df, periods=365)
        forecast = model.predict(future)

        # Plot the forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Display forecasted data
        st.subheader("Forecasted Revenue")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Optional: Display components such as components for analysis (e.g., trend, yearly, weekly)
        st.subheader("Forecast Components")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)

    else:
        st.error("Please make sure your file has 'Date' and 'Revenue' columns.")
