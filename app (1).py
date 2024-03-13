import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

# Function to forecast next n days
def forecast_next_n_days(model, data, last_date, n):
    temp_input = list(data)
    lst_output = []
    n_steps = 100

    for i in range(n):
        if len(temp_input) >= n_steps:
            x_input = np.array(temp_input[-n_steps:]).reshape((1, n_steps, 1))
        else:
            padding = np.zeros((n_steps - len(temp_input), 1))
            x_input = np.concatenate((padding, np.array(temp_input).reshape((-1, 1))), axis=0)
            x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input)
        temp_input.append(yhat[0, 0])
        lst_output.append(yhat[0, 0])

    return lst_output, pd.date_range(start=last_date, periods=n + 1)[1:]

# Load the model weights
model = tf.keras.Sequential([
    LSTM(150, activation='relu', input_shape=(100, 1), recurrent_initializer='glorot_uniform'),
    Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="adam")
model.load_weights("lstm_model.h5")

# Streamlit app
st.title("Brent Crude Oil Price Prediction")

# User input for number of days to forecast
days_input = st.number_input("Enter the number of days for forecast", min_value=1, max_value=30, value=1)

# Placeholder for user data input
st.subheader("Historical Data")

# Placeholder for user data input
st.write("Enter historical data (comma-separated):")
data_input = st.text_area("Historical Data")

# Perform data preprocessing and make predictions
if st.button("Forecast Next {} Days".format(days_input)):
    try:
        historical_data = np.array(data_input.split(',')).astype(float)

        # Perform data preprocessing
        scaler = StandardScaler()
        historical_data = scaler.fit_transform(historical_data.reshape(-1, 1)).reshape(-1)

        # Forecast next n days
        forecasted_prices, forecasted_dates = forecast_next_n_days(model, historical_data, pd.Timestamp.today(), days_input)
        forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1)).reshape(-1)
        forecasted_data = pd.DataFrame({"Date": forecasted_dates, "Forecasted Price": forecasted_prices})

        st.subheader("Forecasted Prices for Next {} Days:".format(days_input))
        st.write(forecasted_data)
    except ValueError:
        st.error("Please enter valid historical data in comma-separated format.")
