import numpy as np
import tensorflow as tf
from brian2 import *


num_examples = 10000
num_inputs = 28 * 28
num_neurons = 100
num_epochs = 10
learning_rate = 1e-2

tau = 10*ms
v_rest = -65*mV
v_reset = -65*mV
v_threshold = -50*mV
refractory_period = 5*ms

eqs = '''dv/dt = (v_rest - v) / tau : volt (unless refractory)'''


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    input_layer = SpikeGeneratorGroup(
        num_inputs, 
        np.repeat(np.arange(num_inputs), num_examples), 
        np.tile(np.arange(num_examples), num_inputs)*ms
    )

    neurons = NeuronGroup(
        num_neurons, 
        eqs, 
        threshold='v>v_threshold', 
        reset='v = v_reset', 
        refractory=refractory_period, 
        method='exact'
    )
    neurons.v = v_rest

    synapses = Synapses(input_layer, neurons,
                        '''
                        w : 1
                        dApre/dt = -Apre / tau_pre : 1 (event-driven)
                        dApost/dt = -Apost / tau_post : 1 (event-driven)
                        ''',
                        on_pre='''
                        v_post += w*mV
                        Apre += dApre
                        w = clip(w + Apost, 0, w_max)
                        ''',
                        on_post='''
                        Apost += dApost
                        w = clip(w + Apre, 0, w_max)
                        ''')
    synapses.connect(p=0.1)
    synapses.w = 'rand()'

    # stdp params
    tau_pre = 20*ms
    tau_post = 20*ms
    dApre = 0.01
    dApost = -dApre * tau_pre / tau_post * 1.05
    w_max = 1.0

    spike_monitor = SpikeMonitor(neurons)
    
    # state_monitor = StateMonitor(neurons, 'v')
    state_monitor = StateMonitor(neurons, 'v', record=True)

    # accuracy tracking!
    def get_predictions():
        spike_counts = np.array([spike_monitor.count[i] for i in range(num_neurons)])
        return np.argmax(spike_counts)

    accuracies = []

    # run
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        
        for i in range(num_examples):
            input_image = x_train[i].flatten()
            input_spikes = np.where(input_image > np.random.rand(num_inputs))[0]
            input_times = np.zeros(len(input_spikes)) * ms
            
            input_layer.set_spikes(input_spikes, input_times)
            run(100*ms)
            neurons.v = v_rest
            
            # calc accuracy
            prediction = get_predictions()
            correct_label = y_train[i]
            accuracy = np.sum(prediction == correct_label) / 1.0  # one image at a time
            accuracies.append(accuracy)
            
            if (i+1) % 100 == 0 or i == num_examples - 1:
                avg_accuracy = np.mean(accuracies[-100:])  # av acc of last 100 examples
                print(f'Processed {i+1}/{num_examples} examples in epoch {epoch+1}/{num_epochs} - Accuracy: {avg_accuracy:.2f}')
        print(f'Completed epoch {epoch+1}/{num_epochs}')

    # TEST BETTER!!!


if __name__ == "__main__":
    main()
