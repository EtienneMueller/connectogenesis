import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

n_first_layer = 256
n_second_layer = 64
epochs = 10


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(n_first_layer, activation='relu'),
    Dense(n_second_layer, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_original = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# Save the trained weights
original_weights = model.get_weights()

# Ensure all weights are between 0 and 1
for i in range(len(original_weights)):
    original_weights[i] = np.clip(original_weights[i], 0, 1)


def modify_weights(weights):
    modified_weights = []
    for weight_matrix in weights:
        flat_weights = weight_matrix.flatten()
        num_weights = flat_weights.shape[0]
        indices = np.arange(num_weights)
        np.random.shuffle(indices)
        chosen_indices = indices[:int(0.2 * num_weights)]

        for idx in chosen_indices:
            if flat_weights[idx] > 0.67:
                flat_weights[idx] = 0.9
            elif flat_weights[idx] < 0.33:
                flat_weights[idx] = 0.1
            else:
                flat_weights[idx] = 0.5

        remaining_indices = indices[int(0.2 * num_weights):]
        flat_weights[remaining_indices] = np.random.rand(len(remaining_indices))

        modified_weights.append(flat_weights.reshape(weight_matrix.shape))
    
    return modified_weights

modified_weights = modify_weights(original_weights)


# Create a new model with the same layout
new_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(n_first_layer, activation='relu'),
    Dense(n_second_layer, activation='relu'),
    Dense(10, activation='softmax')
])

# Load the modified weights into the new model
new_model.set_weights(modified_weights)

# Compile the new model
new_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the new model
history_new = new_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))


# Function to find the epoch with the highest accuracy
def get_highest_accuracy_epoch(history):
    val_acc = history.history['val_accuracy']
    highest_acc = max(val_acc)
    epoch = val_acc.index(highest_acc)
    return highest_acc, epoch + 1

# Train the original model again for comparison
#history_original = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Train the new model again for comparison
#history_new = new_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Get the highest accuracy and corresponding epoch for both models
original_highest_acc, original_epoch = get_highest_accuracy_epoch(history_original)
new_highest_acc, new_epoch = get_highest_accuracy_epoch(history_new)

# Compare the results
print(f"Original Model: Highest Accuracy = {original_highest_acc*100:.2f}% at Epoch {original_epoch}")
print(f"New Model: Highest Accuracy = {new_highest_acc*100:.2f}% at Epoch {new_epoch}")


