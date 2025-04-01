import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


#save_to = "generated_data/"
#save_to = "generated_data_mnist/"
save_to = "generated_data/mnist/"


def load_and_preprocess_mnist():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype("float32")
    x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
    return x_train


# def build_generator():
    # model = tf.keras.Sequential()
    # model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     assert model.output_shape == (None, 14, 14, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     assert model.output_shape == (None, 28, 28, 1)

#     return model

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def build_discriminator():
    # model = tf.keras.Sequential()
    # model.add(layers.Conv2D(64, (8, 8), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    #model.add(layers.ReLU())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dropout(0.35))
    model.add(layers.Dropout(0.3))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    #model.add(layers.ReLU())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dropout(0.35))
    model.add(layers.Dropout(0.3))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def build_gan(generator, discriminator):
    #model = tf.keras.Sequential(generator)
    #model = tf.keras.Sequential(discriminator)
    model = tf.keras.Sequential()

    #model.add(discriminator)
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, data, epochs=10000, batch_size=128):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # real images
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_images = data[idx]

        # Generate fakes
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)

        # train discriminator
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            
            #if epoch % 1000 == 0:
            save_generated_images(epoch, generator)


def save_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_to + f"{epoch}.png")
    plt.close()


if __name__ == "__main__":
    images = load_and_preprocess_mnist()
    print(images.shape)

    generator = build_generator()
    generator.summary()

    discriminator = build_discriminator()
    discriminator.summary()

    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.trainable = False

    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    train_gan(generator, discriminator, gan, images)
