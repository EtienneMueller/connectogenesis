import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

# the data is only 2D!
# convolution expects height x width x color
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# number of classes
K = len(set(y_train))

# model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),  # Additional layer
#     Dense(10, activation='softmax')
# ])

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation='softmax')(x)

model = Model(i, x)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_original = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=1024)

# Save the trained weights
original_weights = model.get_weights()

# Ensure all weights are between 0 and 1
for i in range(len(original_weights)):
    original_weights[i] = np.clip(original_weights[i], 0, 1)


def modify_weights(weights):
    print(len(weights), print(len(weights[0])))
    print(weights)
    modified_weights = []
    for weight_matrix in weights:
        flat_weights = weight_matrix.flatten()
        num_weights = flat_weights.shape[0]
        indices = np.arange(num_weights)
        np.random.shuffle(indices)
        chosen_indices = indices[:int(0.2 * num_weights)]

        print(np.max(flat_weights))

        max_weight = np.max(flat_weights)

        # for idx in chosen_indices:
        #     if flat_weights[idx] > (0.67*max_weight):
        #         value = (0.9*max_weight) + ((np.random.random(1)[0]-0.5)/20)
        #         print("Old:", flat_weights[idx], "New:", value)
        #         flat_weights[idx] = value
        #     elif 0 < flat_weights[idx] < (0.33*max_weight):
        #         value = (0.1*max_weight) + ((np.random.random(1)[0]-0.5)/20)
        #         print("Old:", flat_weights[idx], "New:", value)
        #         flat_weights[idx] = value
        #     elif (0.33*max_weight) < flat_weights[idx] < (0.67*max_weight):
        #         value = (0.5*max_weight) + ((np.random.random(1)[0]-0.5)/20)
        #         print("Old:", flat_weights[idx], "New:", value)
        #         flat_weights[idx] = value
        #     else:
        #         print("Old:", flat_weights[idx])

        for idx in chosen_indices:
            value = float("{:.2f}".format(flat_weights[idx]))
            #print("Old:", flat_weights[idx], "New:", value)
            flat_weights[idx] = value


        remaining_indices = indices[int(0.2 * num_weights):]
        flat_weights[remaining_indices] = np.random.rand(len(remaining_indices))

        modified_weights.append(flat_weights.reshape(weight_matrix.shape))

    
    
    return modified_weights

modified_weights = modify_weights(original_weights)


# Create a new model with the same layout
# new_model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),  # Additional layer
#     Dense(10, activation='softmax')
# ])

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation='softmax')(x)

new_model = Model(i, x)

# Load the modified weights into the new model
new_model.set_weights(modified_weights)

# Compile the new model
new_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the new model
history_new = new_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=1024)


# Function to find the epoch with the highest accuracy
def get_highest_accuracy_epoch(history):
    val_acc = history.history['val_accuracy']
    highest_acc = max(val_acc)
    epoch = val_acc.index(highest_acc)
    return highest_acc, epoch + 1

# Get the highest accuracy and corresponding epoch for both models
original_highest_acc, original_epoch = get_highest_accuracy_epoch(history_original)
new_highest_acc, new_epoch = get_highest_accuracy_epoch(history_new)

# Compare the results
print(f"Original Model: Highest Accuracy = {original_highest_acc*100:.2f}% at Epoch {original_epoch}")
print(f"New Model: Highest Accuracy = {new_highest_acc*100:.2f}% at Epoch {new_epoch}")
