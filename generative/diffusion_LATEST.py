import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


PATH_TO_IMAGES = 'training_data/zf28'
SAVE_TO = 'generated_data/diffusion/'


def load_and_preprocess_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))
        img = np.asarray(img).astype("float32")
        images.append(img)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images


class NoiseSchedule:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = np.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def get_alpha_bars(self, t):
        return self.alpha_bars[t]

    def get_alphas(self, t):
        return self.alphas[t]

    def get_betas(self, t):
        return self.betas[t]


def build_denoising_model_SMALL(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))
    return model


def build_denoising_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))
    return model


def train_diffusion_model(model, data, noise_schedule, epochs=10000, batch_size=128):
    optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        batch = data[idx]

        t = np.random.randint(0, noise_schedule.timesteps, batch_size)
        noise = np.random.normal(size=batch.shape)

        alpha_bars = noise_schedule.get_alpha_bars(t)[:, None, None, None]
        noisy_images = np.sqrt(np.maximum(alpha_bars, 1e-10)) * batch + np.sqrt(np.maximum(1 - alpha_bars, 1e-10)) * noise

        with tf.GradientTape() as tape:
            predictions = model(noisy_images, training=True)
            loss = mse_loss(noise, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

            if epoch % 1000 == 0:
                save_generated_images(epoch, model, noise_schedule)


def save_generated_images(epoch, model, noise_schedule, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(size=(examples, 28, 28, 1))
    t = noise_schedule.timesteps - 1

    for i in range(noise_schedule.timesteps):
        alpha_bars = noise_schedule.get_alpha_bars(np.array([t] * examples))[:, None, None, None]
        noise_pred = model(noise, training=False)
        noise = (noise - noise_pred * np.sqrt(np.maximum(1 - alpha_bars, 1e-10))) / np.sqrt(np.maximum(alpha_bars, 1e-10))
        t -= 1

    generated_images = 0.5 * noise + 0.5  # Rescale to [0, 1]

    generated_images_np = generated_images.numpy()
    print(f"Generated images min: {generated_images_np.min()}, max: {generated_images_np.max()}")
    if np.isinf(generated_images_np).any() or np.isnan(generated_images_np).any():
        print("Warning: Generated images contain inf or NaN values!")

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images_np[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_TO + f"{epoch}.png")
    plt.close()


if __name__ == "__main__":
    images = load_and_preprocess_images(PATH_TO_IMAGES)

    denoising_model = build_denoising_model((28, 28, 1))
    denoising_model.summary()

    # Initialize noise schedule + train the model
    noise_schedule = NoiseSchedule(timesteps=1000)
    train_diffusion_model(denoising_model, images, noise_schedule)
