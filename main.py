import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split


def load_coordinate(file_path, layer_shift = 16):
    # here: 1212 time steps = 10:06 minutes
    # here: length ~7900 neurons in plane 15 etc
    length = len(np.load(file_path + f"/plane{0+layer_shift}/stat.npy", allow_pickle=True))
    x_array = np.empty(length)
    y_array = np.empty(length)
    z_array = np.empty(length)
    brightness = np.random.rand(length, 1212)
    
    for i in range(1):
        stat_file = file_path + f"/plane{i+layer_shift}/stat.npy"
        stat = np.load(stat_file, allow_pickle=True)

        f_file = file_path + f"/plane{i+layer_shift}/F.npy"
        f = np.load(f_file, allow_pickle=True)
        
        for j in range(length):
            x_array[j] = stat[j]['med'][0]
            y_array[j] = stat[j]['med'][1]
            z_array[j] = i
            brightness[j] = f[j]

    b_min = np.min(brightness)
    b_max = np.max(brightness)
    brightness = np.divide(brightness-b_min, b_max-b_min)
    x, y, z, brightness = filter(x_array, y_array, z_array, brightness)

    return x, y, z, brightness


def load_timesteps(file_path):
    for i in range(50):
        ts = np.load(file_path + f"/plane0/stat.npy", allow_pickle=True)
    print(len(ts))
    

def filter(x, y, z, brightness, brightness_threshold=0.32):
    average_brightness = np.mean(brightness, axis=1)
    mask = average_brightness >= brightness_threshold
    x, y, z = x[mask], y[mask], z[mask]
    
    brightness = brightness[mask]
    
    b_min = np.min(brightness)
    b_max = np.max(brightness)
    
    brightness = np.divide(brightness-b_min, b_max-b_min)

    return x, y, z, brightness


def load_and_stack_data(data_folder="data/suite2p", num_planes=50):
    stacked_data = []
    
    for i in range(num_planes):
        plane_folder = os.path.join(data_folder, f"plane{i}")
        file_path = os.path.join(plane_folder, "F.npy")
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            stacked_data.append(data)
        else:
            print(f"File not found: {file_path}")
    
    print("done")
    # Stack all arrays vertically
    combined_array = np.vstack(stacked_data)
    return combined_array


def nn(data):
    #data = np.random.rand(100, 50)  # Example data with 100 data points and 50 time steps

    # Transpose data so that time steps are rows instead of columns
    data = data.T  # New shape will be (time_steps, data_points), i.e., (50, 100)

    # Define input (X) and shifted output (y)
    X = data  # Use all rows (time steps) and all columns (data points) as input
    y = np.zeros_like(data)  # Initialize y with the same shape as data
    y[:-1, :data.shape[1] // 2] = data[1:, :data.shape[1] // 2]  # Shifted top half for target variable

    # Expand dimensions if necessary to make the data 3D (samples, timesteps, features)
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple model with a direct connection between input and output
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),  # Flatten to ensure direct mapping
        tf.keras.layers.Dense(y.shape[1], activation='linear')  # Direct connection to output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mae')

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Test loss:", loss)


def main():
    data = load_and_stack_data()
    nn(data)
    x, y, z, brightness = load_coordinate('data/suite2p')
    print(x, y, z, brightness)
    load_timesteps('data/suite2p')


if __name__=='__main__':
    main()
