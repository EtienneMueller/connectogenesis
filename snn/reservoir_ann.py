import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# Actually a ECHO STATE NETWORK

input_size = 1 
reservoir_size = 100
output_size = 1
spectral_radius = 0.95
sparsity = 0.1
epochs = 50


def generate_weights():
    W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
    W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_res)))
    W_res[np.random.rand(*W_res.shape) > sparsity] = 0

    W_in = np.random.rand(reservoir_size, input_size) - 0.5
    return W_res, W_in


def generate_training_data():
    t = np.linspace(0, 10, 1000)
    u_train = np.sin(t).reshape(-1, 1)
    y_train = np.cos(t).reshape(-1, 1)
    return t, u_train, y_train


def reservoir_update(x, u, W_in, W_res):
    return np.tanh(np.dot(W_in, u) + np.dot(W_res, x))


def main():
    W_res, W_in = generate_weights()
    t, u_train, y_train = generate_training_data()
    y = y_train

    # init reservoir
    x = np.zeros((reservoir_size, 1))

    # collect states
    states = []
    for u in u_train:
        x = reservoir_update(x, u, W_in, W_res)
        states.append(x)
    states = np.array(states) # Should be (1000, 100)

    model = Sequential([
        Dense(output_size, input_shape=(reservoir_size,), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    # show model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    img = plt.imread('model.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    model.fit(states, y_train, epochs=epochs, verbose=1)
    predictions = model.predict(states)

    # !!!!!!!!!!!!!!!!!!!!!
    # !!! FIX THIS SHIT !!!
    # !!!!!!!!!!!!!!!!!!!!!
    y_train = y_train[0][:][:]
    print(t.shape[0], y.shape)
    t = np.reshape(t, (t.shape[0], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
    print(t.shape, y.shape)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label='True')
    plt.plot(t, predictions, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
