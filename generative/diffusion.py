import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def load_and_preprocess_mnist():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype("float32")
    x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
    return x_train


class NoiseSchedule:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        #self.betas = np.linspace(0.0002, 0.02, timesteps)
        #self.betas = np.linspace(0.0001, 0.01, timesteps)
        #self.betas = np.linspace(0.0001, 0.01, timesteps)
        self.betas = np.linspace(0.0001, 0.02, timesteps)
        #self.alphas = 0.5 - self.betas
        #self.alphas = 0 - self.betas
        self.alphas = 1.0 - self.betas
        #self.alphas = -1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def get_alpha_bars(self, t):
        return self.alpha_bars[t]

    def get_alphas(self, t):
        return self.alphas[t]

    def get_betas(self, t):
        return self.betas[t]


def build_denoising_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    #model.add(layers.Conv2D(64, (3, 3), padding='same', activation='sigmoid'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(256, (3, 3), padding='same', activation='sigmoid'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    #model.add(layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), padding='same', activation='sigmoid'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))
    return model


def train_diffusion_model(model, data, noise_schedule, epochs=10000, batch_size=128):
    
    #optimizer = tf.keras.optimizers.Adam(1e-3)
    #optimizer = tf.keras.optimizers.Adam(8e-4)
    #optimizer = tf.keras.optimizers.Adam(5e-4)
    #optimizer = tf.keras.optimizers.Adam(3e-4)
    #optimizer = tf.keras.optimizers.Adam(2e-4)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    #optimizer = tf.keras.optimizers.Adam(5e-5)
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    # SOMETHING IS WRONG IN HERE
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        batch = data[idx]

        t = np.random.randint(0, noise_schedule.timesteps, batch_size)
        noise = np.random.normal(size=batch.shape)

        alpha_bars = noise_schedule.get_alpha_bars(t)[:, None, None, None]
        noisy_images = np.sqrt(alpha_bars) * batch + np.sqrt(1 - alpha_bars) * noise

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
        noise_pred = model(noise)
        noise = (noise - noise_pred * np.sqrt(1 - alpha_bars)) / np.sqrt(alpha_bars)
        t -= 1

    generated_images = 0.5 * noise + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"diffusion_generated_image_epoch_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    images = load_and_preprocess_mnist()

    denoising_model = build_denoising_model((28, 28, 1))
    denoising_model.summary()

    noise_schedule = NoiseSchedule(timesteps=1000)
    train_diffusion_model(denoising_model, images, noise_schedule)