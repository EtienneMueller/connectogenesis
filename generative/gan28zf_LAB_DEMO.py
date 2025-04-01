import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image
from tensorflow.keras import layers


PATH_TO_IMAGES = 'training_data/zf28'
SAVE_TO = "generated_data/zf28/"
EPOCHS=100000
SAVE_IMG_EPOCHS=1000


def load_images(image_folder, image_size=(28, 28)):
    images = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(image_size)
        img = np.asarray(img).astype("float32")
        images.append(img)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images


def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def build_gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, data, epochs=EPOCHS, batch_size=128):
    epoch_time = time.time()
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # train discriminator
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_images = data[idx]

        # generate fake images
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)

        # train discriminator
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            
            if epoch % SAVE_IMG_EPOCHS == 0:
                print(f"--- {(time.time()-epoch_time)}s ---")
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
    plt.savefig(SAVE_TO + f"{epoch}.png")
    plt.close()


def main():
    os.makedirs(SAVE_TO, exist_ok=True)
    images = load_images(PATH_TO_IMAGES)
    print(images.shape)

    generator = build_generator()
    generator.summary()

    discriminator = build_discriminator()
    discriminator.summary()

    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.trainable = False

    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    start_time = time.time()
    train_gan(generator, discriminator, gan, images)
    print(f"--- Total time: {(time.time()-start_time)}s ---")


if __name__ == "__main__":
    main()
