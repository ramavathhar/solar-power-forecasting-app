import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Power Forecasting", layout="wide")

# -----------------------
# Global CSS Styling
st.markdown("""
    <style>
        body, .main, label, h1, h2, h3, h4, p {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1rem !important;
        }

        .sidebar .sidebar-content {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .stDownloadButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 1rem;
            padding: 10px 24px;
        }

        .stDownloadButton > button:hover {
            background-color: #43a047;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Header Section
st.markdown("""
    <h1 style='color:#4CAF50; font-size: 46px; font-weight: bold; text-align: center; 
    line-height: 1.4; word-break: break-word;'>🌞 Solar Power Forecasting App</h1>
    <p style='color: #666; font-size: 20px; text-align: center; font-style: italic;'>
        Predict the AC power output of solar panels based on environmental inputs.
    </p>
""", unsafe_allow_html=True)

# Logo Centered
try:
    logo = Image.open("logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=150)
except Exception as e:
    st.error(f"Error loading logo: {e}")

# -----------------------
# Load Model and Scaler
@st.cache_resource
def load_assets():
    model = load_model("solar_forecasting_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_assets()

# -----------------------
# Sidebar Inputs
st.sidebar.header("🌤️ Environmental Inputs")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
weekday = st.sidebar.selectbox("Weekday (0 = Monday)", list(range(7)), index=2)

st.sidebar.header("🌡️ Temperature Inputs")
ambient_temp = st.sidebar.number_input("Ambient Temperature (°C)", value=30.0)
module_temp = st.sidebar.number_input("Module Temperature (°C)", value=45.0)

st.sidebar.header("☀️ Irradiation")
irradiation = st.sidebar.number_input("Irradiation (W/m²)", value=800.0)

st.sidebar.header("⚙️ Power Trend Data")
lag1 = st.sidebar.slider("AC Power Lag-1 (scaled)", 0.0, 1.0, 0.5)
roll_mean = st.sidebar.slider("Rolling Mean (scaled)", 0.0, 1.0, 0.5)

# -----------------------
# Prediction
input_features = pd.DataFrame([[hour, day, weekday, ambient_temp, module_temp,
                                irradiation, lag1, roll_mean]],
                              columns=[
                                  'hour', 'day', 'weekday', 'AMBIENT_TEMPERATURE',
                                  'MODULE_TEMPERATURE', 'IRRADIATION',
                                  'AC_POWER_LAG1', 'AC_POWER_ROLL_MEAN3'])

input_scaled = scaler.transform(input_features)
input_scaled_reshaped = np.reshape(input_scaled, (1, 1, input_scaled.shape[1]))

with st.spinner("Predicting solar power output..."):
    prediction_scaled = model.predict(input_scaled_reshaped)[0][0]

st.markdown(f"""
    <div style="background-color:#ffffff; border-radius:10px; padding:30px; 
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); margin-top: 20px; text-align:center;">
        <h2 style="color:#2e7d32;">🔋 Forecast Result</h2>
        <h1 style="color: #388e3c; font-size: 60px; margin: 10px 0;">{prediction_scaled:.4f}</h1>
        <p style="font-size: 16px;">Predicted Scaled AC Power</p>
        <p style="color:#888; font-size: 14px;">ℹ️ Higher values indicate higher solar power generation potential.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------
# Upload CSV for Batch Forecast
st.sidebar.markdown("---")
st.sidebar.header("📄 Upload Test Data")
st.sidebar.markdown("ℹ️ Upload a CSV with columns like `hour`, `day`, `AC_POWER`, etc.")
uploaded_file = st.sidebar.file_uploader("Upload a test CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

        input_columns = ['hour', 'day', 'weekday', 'AMBIENT_TEMPERATURE',
                         'MODULE_TEMPERATURE', 'IRRADIATION',
                         'AC_POWER_LAG1', 'AC_POWER_ROLL_MEAN3']

        if all(col in test_df.columns for col in input_columns + ['AC_POWER']):
            X_test = test_df[input_columns]
            y_actual = test_df['AC_POWER']

            X_scaled = scaler.transform(X_test)
            X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
            y_pred_scaled = model.predict(X_reshaped)

            if 'AC_POWER' in scaler.feature_names_in_:
                ac_power_index = list(scaler.feature_names_in_).index('AC_POWER')
                dummy = np.zeros((len(y_pred_scaled), len(scaler.feature_names_in_)))
                dummy[:, ac_power_index] = y_pred_scaled.flatten()
                y_pred = scaler.inverse_transform(dummy)[:, ac_power_index]
            else:
                y_pred = y_pred_scaled.flatten()

            st.markdown("### 📊 Model Forecast Accuracy (Test Set)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_actual[:100], name="Actual", mode='lines+markers'))
            fig.add_trace(go.Scatter(y=y_pred[:100], name="Predicted", mode='lines+markers'))
            fig.update_layout(title="📉 Actual vs Predicted AC Power",
                              xaxis_title="Sample Index",
                              yaxis_title="AC Power (kW)",
                              template="plotly_white")
            st.plotly_chart(fig)

            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)

            st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:10px; border-radius:10px; margin-top: 10px;">
                    <h4>🧮 Model Evaluation Metrics</h4>
                    <p><strong>MAE:</strong> {mae:.4f} kW</p>
                    <p><strong>RMSE:</strong> {rmse:.4f} kW</p>
                    <p><strong>R² Score:</strong> {r2:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

            predictions_df = pd.DataFrame({
                'Hour': test_df['hour'],
                'Predicted AC Power (kW)': y_pred,
                'Actual AC Power (kW)': y_actual
            })

            st.download_button(
                label="Download Predictions as CSV",
                data=predictions_df.to_csv(index=False).encode('utf-8'),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("❗ Ensure CSV includes required columns, including 'AC_POWER' for actual values.")
    except Exception as e:
        st.error(f"🚫 Error processing uploaded file: {e}")

# -----------------------
# Model Info Section
with st.expander("📘 Model Info"):
    st.markdown("""
    - **Architecture:** LSTM with 2 layers
    - **Input Features:** 8 total (time, temp, irradiation, power trends)
    - **Training Set:** 34 days of inverter-level solar power data
    - **Loss Function:** Mean Squared Error
    """)
