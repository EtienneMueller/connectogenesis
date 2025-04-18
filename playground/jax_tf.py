import tensorflow as tf
from tensorflow.keras import layers, models
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tensorflow.keras.datasets import mnist


def tf_script():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


def jax_script(): 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = jnp.array(x_train), jnp.array(x_test)
    y_train, y_test = jnp.array(y_train), jnp.array(y_test)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))  # Flatten
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.2)(x, deterministic=True)
            x = nn.Dense(10)(x)
            return x

    # initialize the model
    rng = jax.random.PRNGKey(0)
    model = MLP()
    variables = model.init(rng, x_train[:1])

    def cross_entropy_loss(logits, labels):
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))

    def compute_metrics(logits, labels):
        loss = cross_entropy_loss(logits, labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {'loss': loss, 'accuracy': accuracy}

    # train
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = model.apply(params, batch['image'])
            loss = cross_entropy_loss(logits, batch['label'])
            return loss, logits
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits, batch['label'])
        return state, metrics

    # eval
    @jax.jit
    def eval_step(params, batch):
        logits = model.apply(params, batch['image'])
        return compute_metrics(logits, batch['label'])

    # train state
    state = train_state.TrainState.create(apply_fn=model.apply,
                                        params=variables['params'],
                                        tx=optax.adam(1e-3))

    batch_size = 64
    num_epochs = 5
    num_train_steps = len(x_train) // batch_size

    for epoch in range(num_epochs):
        for i in range(num_train_steps):
            batch = {
                'image': x_train[i*batch_size:(i+1)*batch_size],
                'label': y_train[i*batch_size:(i+1)*batch_size],
            }
            state, metrics = train_step(state, batch)
        print(f'Epoch {epoch + 1}, Loss: {metrics["loss"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')

    # Evaluation
    test_metrics = eval_step(state.params, {'image': x_test, 'label': y_test})
    print(f'Test Loss: {test_metrics["loss"]:.4f}, Test Accuracy: {test_metrics["accuracy"]:.4f}')


if __name__ == "__main__":
    #tf_script()
    jax_script()
