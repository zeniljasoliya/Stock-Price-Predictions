import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to predict future prices
def predict_future_prices(model, data, scaler, periods=30):
    # Prepare the data for prediction
    last_100_days = data[-100:].values
    last_100_scaled = scaler.transform(last_100_days.reshape(-1, 1))
    x_pred = []
    x_pred.append(last_100_scaled)
    x_pred = np.array(x_pred)
    
    # Make predictions for future periods
    future_predictions = []
    for i in range(periods):
        predicted_price = model.predict(x_pred)
        future_predictions.append(predicted_price[0, 0])
        x_pred = np.roll(x_pred, -1)
        x_pred[-1][-1] = predicted_price
        
    # Inverse transform the predicted prices
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions.flatten()

# Load the pre-trained model
model = load_model('C:\\Users\\zenil\\Python Internship\\Stock Price Predictions\\Stock Predictions Model.keras')

# Streamlit app
st.header('Stock Market Predictor')


stock = st.sidebar.text_input('Enter Stock Symbol', 'TCS.NS')
start = '2012-01-01'
end = '2024-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Display current price in the sidebar
current_price = data['Close'].iloc[-1]
current_price_inr = '₹{:,.2f}'.format(current_price)
st.sidebar.subheader('Current Price (INR)')
st.sidebar.write(current_price_inr)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

# Get user input for the number of future days or years
future_periods_type = st.sidebar.selectbox('Select prediction period type', ['Days', 'Years'])

if future_periods_type == 'Days':
    future_periods = st.sidebar.number_input('Enter the number of days for prediction', value=30)
elif future_periods_type == 'Years':
    future_years = st.sidebar.number_input('Enter the number of years for prediction', value=1)
    future_periods = int(future_years * 365)  # Assuming 365 days in a year

# Get future predictions with corresponding dates
future_dates = pd.date_range(data.index[-1], periods=future_periods+1)[1:]
future_prices = predict_future_prices(model, data['Close'], scaler, periods=future_periods)

# Combine dates and prices into a DataFrame for easier display
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})

# Display future price predictions with dates in the sidebar
st.sidebar.subheader('Future Price Predictions')
st.sidebar.write(future_data)


# Plot future price predictions with current price
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Actual Price', color='blue')
plt.plot(future_dates, future_prices, label='Future Price Predictions', linestyle='--', marker='o', color='red')
plt.scatter([data.index[-1]], [data['Close'].iloc[-1]], color='green', label='Current Price')  # Plot current price
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Future Price Predictions')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Display future price predictions graph
st.subheader('Future Price Predictions Graph')
st.pyplot(plt)
