import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


# Parameters
num_nodes = 5
connection_prob = 0.5
max_distance = 1.0
latent_dim = 100


def create_graph(num_nodes=5, connection_prob=0.5, max_distance=1.0):
    G = nx.Graph()
    for i in range(num_nodes):
        x, y = np.random.rand(2)
        G.add_node(i, pos=(x, y))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < connection_prob:
                pos_i = np.array(G.nodes[i]['pos'])
                pos_j = np.array(G.nodes[j]['pos'])
                distance = np.linalg.norm(pos_i - pos_j)
                if distance <= max_distance:
                    G.add_edge(i, j)
    return G


def plot_graph(G, save_path=None):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2 * num_nodes, activation='tanh'))
    model.add(layers.Reshape((num_nodes, 2)))
    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(num_nodes, 2)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def generate_real_samples(n_samples):
    X = np.array([np.array([node['pos'] for node in create_graph(num_nodes).nodes.values()]) for _ in range(n_samples)])
    y = np.ones((n_samples, 1))
    return X, y


def generate_fake_samples(n_samples):
    input_noise = np.random.randn(n_samples, latent_dim)
    X = generator.predict(input_noise)
    y = np.zeros((n_samples, 1))
    return X, y


def save_generated_graph(epoch):
    noise = np.random.randn(1, latent_dim)
    generated_graph = generator.predict(noise)
    G = nx.Graph()
    for i, (x, y) in enumerate(generated_graph[0]):
        G.add_node(i, pos=(x, y))
    plot_graph(G, save_path=f'generated_graph_epoch_{epoch}.png')


def train_gan(gan, generator, discriminator, n_epochs=10000, n_batch=64, n_eval=2000):
    half_batch = n_batch // 2
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(half_batch)
        X_fake, y_fake = generate_fake_samples(half_batch)
        
        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        input_noise = np.random.randn(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        
        g_loss = gan.train_on_batch(input_noise, y_gan)
        
        if (i + 1) % n_eval == 0:
            print(f'{i+1}, d_loss={d_loss}, g_loss={g_loss}')
            save_generated_graph(i + 1)


if __name__ == "__main__":
    generator = create_generator()
    discriminator = create_discriminator()

    discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_graph = generator(gan_input)
    gan_output = discriminator(generated_graph)
    gan = tf.keras.Model(gan_input, gan_output)

    gan.compile(optimizer='adam', loss='binary_crossentropy')
    # Train the GAN
    train_gan(gan, generator, discriminator)