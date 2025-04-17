import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
import time
from tensorflow.keras import mixed_precision


#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
#mixed_precision.set_global_policy('mixed_float16')


def ann(batches):  # Fashion MNIST
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.

    # height x width x color
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    K = len(set(y_train))

    i = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(K, activation='softmax')(x)

    model = Model(i, x)

    # use GPU!
    start_time = time.time()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.run_eagerly = False
    r = model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=10,
                  batch_size=batches)
    duration = time.time() - start_time
    return duration


if __name__ == "__main__":
    best_batches = 0
    best_duration = 1000
    # [None, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for i in [8192]:  # [1024, 2048, 4096, 8192, 16384]:
        duration = ann(i)
        print("[CudaCores] Batch Size:", i, "Duration:", duration, "seconds")
        if duration < best_duration:
            best_duration = duration
            best_batches = i
        print("[CudaCores] Best so far:", best_batches, best_duration)
    # mixed_precision.set_global_policy('mixed_float16')
    # for i in [1024, 2048, 4096, 8192, 16384]:
    #     duration = ann(i)
    #     print("[TensorCores] Batch Size:", i, "Duration:", duration, "seconds")
    #     if duration < best_duration:
    #         best_duration = duration
    #         best_batches = i
    #     print("[TensorCores] Best so far:", best_batches, best_duration)
