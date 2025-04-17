import tensorflow as tf
import numpy as np
import convert
#from convert2snn import utils
import time


tf.random.set_seed(1234)
batch_size = 512
epochs = 10
act = 'relu'


class SpikingReLU(tf.keras.layers.Layer):
    """Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingReLU, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # suppress warning
    # training as keyword only needed when using with the decorator
    def call(self, input_at_t, states_at_t, training):
        potential = states_at_t[0] + input_at_t
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingSigmoid(tf.keras.layers.Layer):
    """Works like the SpikingReLU but is shiftet by 0.5 to the left.
    An neuron with spike adaptation might result in less conversion loss."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingSigmoid, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_at_t, states_at_t, training):
        potential = states_at_t[0] + (input_at_t + 0.5)
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingTanh(tf.keras.layers.Layer):
    """Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingTanh, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_at_t, states_at_t, training):
        potential = states_at_t[0] + (input_at_t)
        excitatory = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        inhibitory = -1*tf.cast(tf.math.less(potential, -1), dtype=tf.float32)
        output_at_t = excitatory + inhibitory
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class Accumulate(tf.keras.layers.Layer):
    """Accumulates all input as state for use with a softmax layer."""
    # ToDo: include softmax layer directly here?
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(Accumulate, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_at_t, states_at_t, training):
        output_at_t = states_at_t[0] + input_at_t
        states_at_t_plus_1 = output_at_t
        return output_at_t, states_at_t_plus_1


def convert2rate(model, weights, x_test, y_test, err="CategoricalCrossentropy"):
    print("Converted model:\n" + "-"*32)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            print("Input Layer")
            inputs = tf.keras.Input(shape=(1, model.layers[0].input_shape[0][1]), batch_size=y_test.shape[0])
            x = inputs
        elif isinstance(layer, tf.keras.layers.Dense):
            x = tf.keras.layers.Dense(layer.output_shape[1])(x)
            # x = tf.keras.layers.RNN(DenseRNN(layer.output_shape[1]),
            #                         return_sequences=True,
            #                         return_state=False,
            #                         stateful=True)(x)
            if layer.activation.__name__ == 'linear':
                print("Dense Layer w/o activation")
                pass
            elif layer.activation.__name__ == 'relu':
                print("Dense Layer with SpikingReLU")
                x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            elif layer.activation.__name__ == 'sigmoid':
                print("Dense Layer with SpikingSigmoid")
                x = tf.keras.layers.RNN(SpikingSigmoid(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            elif layer.activation.__name__ == 'tanh':
                print("Dense Layer with SpikingTanh")
                x = tf.keras.layers.RNN(SpikingTanh(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            else:
                print('[Info] Activation type',
                      layer.activation.__name__,
                      'not implemented')
        elif isinstance(layer, tf.keras.layers.ReLU):
            print("SpikingReLU Layer")
            x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]),
                                    return_sequences=True,
                                    return_state=False,
                                    stateful=True)(x)
        elif isinstance(layer, tf.keras.layers.Softmax):
            print("Accumulate + Softmax Layer")
            print(layer.output_shape[1])
            x = tf.keras.layers.RNN(Accumulate(layer.output_shape[1]),
                                    return_sequences=True,
                                    return_state=False,
                                    stateful=True)(x)
            x = tf.keras.layers.Softmax()(x)
        else:
            print("[Info] Layer type ", layer, "not implemented")
    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)
    print("-"*32 + "\n")

    if err == "CategoricalCrossentropy":
        print("CategoricalCrossentropy")
        spiking.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),  # (from_logits=True),
            optimizer="adam",
            metrics=["categorical_accuracy"],)
    elif err == "SparseCategoricalCrossentropy":
        print("SparseCategoricalCrossentropy")
        spiking.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["sparse_categorical_accuracy"],)
    elif err == "MeanSquaredError":
        print("MeanSquaredError")
        spiking.compile(
            loss=tf.keras.losses.MeanSquaredError(),  # (from_logits=True),
            optimizer="adam",
            metrics=["mean_squared_error"],)

    spiking.set_weights(weights)
    return spiking


def get_normalized_weights(model, x_test, percentile=100):
    all_activations = np.zeros([1, ])
    for layer in model.layers:
        activation = tf.keras.Model(inputs=model.inputs,
                                    outputs=layer.output)(x_test).numpy()
        all_activations = np.concatenate((all_activations, activation.flatten()))

    max_activation = np.percentile(all_activations, percentile)

    weights = model.get_weights()
    if max_activation == 0:
        print("\n" + "-"*32 + "\nNo normalization\n" + "-"*32)
    else:
        print("\n" + "-"*32 + "\nNormalizing by", max_activation, "\n" + "-"*32)
        for i in range(len(weights)):
            weights[i] /= (max_activation)

    # Testing normalized weights
    # model.set_weights(weights)
    # max_activation = 0
    # for layer in model.layers:
    #     print(type(layer))
    #     if isinstance(layer, tf.keras.layers.ReLU):
    #         activation = tf.keras.Model(inputs=model.inputs, outputs=layer.output)(x_test).numpy()
    #         print("Local max", np.amax(activation))
    #         if np.amax(activation) > max_activation:
    #             max_activation = np.amax(activation)
    #         print("Max", max_activation)
    return weights


def evaluate_conversion(converted_model, x_test, y_test, testacc, timesteps=100):
    for i in range(1, int(timesteps+1)):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=1)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
            "acc spiking (orig): %.2f%% (%.2f%%)" % (acc*100, testacc*100),
            "- conv loss: %+.2f%%" % ((-(1 - acc/testacc)*100)))


def main():
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    
    # Unroll (60000, 28, 28) to (60000, 784)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    # Analog model
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(500, activation=act)(inputs)
    # x = tf.keras.layers.ReLU()(x)  # max_value=1
    x = tf.keras.layers.Dense(100, activation=act)(x)
    # x = tf.keras.layers.Activation(tf.nn.relu)(x)  # not implemented yet
    x = tf.keras.layers.Dense(10, activation=act)(x)
    x = tf.keras.layers.Softmax()(x)
    ann = tf.keras.Model(inputs=inputs, outputs=x)

    ann.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ann.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs)

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    # weights = ann.get_weights()
    weights = get_normalized_weights(ann, x_train, percentile=85)

    # Preprocessing for RNN
    x_train = np.expand_dims(x_train, axis=1)  # (60000, 784)->(60000, 1, 784)
    x_test = np.expand_dims(x_test, axis=1)
    # x_rnn = np.tile(x_train, (1, 1, 1))
    # y_rnn = y_train  # np.tile(x_test, (1, timesteps, 1))

    # Conversion to spiking model
    snn = convert2rate(ann, weights, x_test, y_test, err="SparseCategoricalCrossentropy")
    # evaluate_conversion(snn, ann, x_test, y_test, testacc, timesteps=100)
    evaluate_conversion(snn, ann, x_test, y_test, testacc)
    duration = time.time() - start_time
    print("Duration:", duration, "seconds")


if __name__ == "__main__":
    main()
