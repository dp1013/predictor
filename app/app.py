import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load your trained model (update the path to your model file)
model_path = r'C:\Users\Admin\OneDrive\Desktop\app\keras_model.h5'  # Use raw string for Windows paths
try:
    model = load_model(model_path)
    st.write(f"Model loaded from {model_path}")
except FileNotFoundError:
    st.error(f"Model file {model_path} not found. Please check the file path.")
    raise

start = '2010-01-01'
end = '2019-12-31'

st.title("Stock Trend Prediction")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010-2019')
st.write(df.describe())

# Visualizing the starting price vs. closing price
st.subheader('Starting Price vs Closing Price')
fig, ax = plt.subplots()
ax.plot(df['Open'], label='Opening Price')
ax.plot(df['Close'], label='Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Prepare the data for the model
data = df.filter(['Close'])
dataset = data.values

training_data_len = int(np.ceil(len(dataset) * 0.95))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Test data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the original vs predicted prices
st.subheader('Original vs Predicted Prices')
fig2, ax2 = plt.subplots()
ax2.plot(df.index[training_data_len:], y_test, label='Original Price')
ax2.plot(df.index[training_data_len:], predictions, label='Predicted Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)
