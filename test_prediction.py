import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, usecols=[1])  # Only get the passenger count column
data = data.values.astype(float)

# Scale the data to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Prepare the data for "one timestep to next timestep" prediction
X, y = [], []
for i in range(len(data) - 1):
    X.append(data[i])  # Use one timestep as input
    y.append(data[i + 1])  # Next timestep as output

X = np.array(X)
y = np.array(y)

# Reshape X to [samples, time steps, features] for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Define the model
model = Sequential([
    LSTM(50, input_shape=(1, 1)),  # 1 timestep, 1 feature
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# Make predictions
predictions = model.predict(X)

# Inverse scale to get actual values
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y)

# Calculate Mean Absolute Error
mae = mean_absolute_error(actual, predictions)
print(f"Mean Absolute Error: {mae}")

# Plot the results
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Passengers')
plt.title('One Timestep Prediction')
plt.legend()
plt.show()
