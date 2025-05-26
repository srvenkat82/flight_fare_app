
import streamlit as st
import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, atan2

# Load model and IATA coordinate data
model = joblib.load("flight_fare_predictor.pkl")
iata_df = pd.read_csv("iata_coords.csv", index_col=0)

# Calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# UI
st.title("Flight Base Fare Predictor")

origin = st.selectbox("Origin Airport (IATA)", options=iata_df.index)
destination = st.selectbox("Destination Airport (IATA)", options=iata_df.index)
booking_days = st.slider("Days in advance booking", min_value=1, max_value=90, value=30)
seat_class = st.selectbox("Seat Class", options=["Economy", "Premium Economy", "Business", "First"])

if origin == destination:
    st.warning("Origin and destination cannot be the same.")
else:
    if st.button("Predict Base Fare"):
        # Get coordinates
        lat1, lon1 = iata_df.loc[origin]
        lat2, lon2 = iata_df.loc[destination]
        distance = haversine(lat1, lon1, lat2, lon2)

        # Prepare input
        input_df = pd.DataFrame.from_dict({
            'origin': [origin],
            'destination': [destination],
            'booking_days': [booking_days],
            'seat_class': [seat_class],
            'distance': [distance]
        })

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Base Fare: ${prediction:.2f}")
