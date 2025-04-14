import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import backend as K


def gcamp_activation(x):
    return K.switch(x >= 0, tf.math.tanh(x+0.1)*10, tf.math.tanh(x) + 1)


def main(activation):  # Fashion MNIST
    # Load the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.

    # only 2D!
    # conv2d: height x width x color
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    num_classes = len(set(y_train))

    # functional API!!!
    if activation == "gcamp":
        i = Input(shape=x_train[0].shape)
        x = Conv2D(32, (3, 3), strides=2, activation=gcamp_activation)(i)
        x = Conv2D(64, (3, 3), strides=2, activation=gcamp_activation)(x)
        x = Conv2D(128, (3, 3), strides=2, activation=gcamp_activation)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation=gcamp_activation)(x)
        x = Dropout(0.2)(x)
        x = Dense(num_classes, activation='softmax')(x)
    else:
        i = Input(shape=x_train[0].shape)
        x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
        x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(num_classes, activation='softmax')(x)

    model = Model(i, x)

    # Compile and fit
    # use the GPU!!!!!
    start_time = time.time()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.run_eagerly = False
    r = model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=15,
                  batch_size=8192)
    duration = time.time() - start_time
    return duration


if __name__ == "__main__":
    duration = main("gcamp")
    print("Duration:", duration, "seconds")
