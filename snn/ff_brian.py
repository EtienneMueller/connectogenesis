import time
from brian2 import *


def main():
    start_scope()

    tau = 10 * ms
    V_rest = -65 * mV
    V_reset = -65 * mV
    V_threshold = -50 * mV
    refractory_period = 5 * ms

    eqs = '''
    dv/dt = (V_rest - v) / tau : volt
    '''

    input_layer = PoissonGroup(100, rates=10 * Hz)  # Poisson group
    hidden_layer = NeuronGroup(100, 
                            eqs, 
                            threshold='v > V_threshold', 
                            reset='v = V_reset', 
                            refractory=refractory_period, 
                            method='exact')
    output_layer = NeuronGroup(100, 
                            eqs, 
                            threshold='v > V_threshold', 
                            reset='v = V_reset', 
                            refractory=refractory_period, 
                            method='exact')

    syn_input_hidden = Synapses(input_layer, hidden_layer, on_pre='v_post += 1 * mV')
    syn_hidden_output = Synapses(hidden_layer, output_layer, on_pre='v_post += 1 * mV')

    syn_input_hidden.connect(p=0.9)
    syn_hidden_output.connect(p=0.5)

    spike_mon_input = SpikeMonitor(input_layer)
    spike_mon_hidden = SpikeMonitor(hidden_layer)
    spike_mon_output = SpikeMonitor(output_layer)

    start_time = time.time()
    run(10 * second)
    runtime = time.time() - start_time
    print(f"{runtime}s")


    figure(figsize=(12, 6))

    subplot(311)
    plot(spike_mon_input.t / ms, spike_mon_input.i, '.k')
    xlabel('Time (ms)')
    ylabel('Input layer neuron index')
    title('Input Layer Spikes')

    subplot(312)
    plot(spike_mon_hidden.t / ms, spike_mon_hidden.i, '.k')
    xlabel('Time (ms)')
    ylabel('Hidden layer neuron index')
    title('Hidden Layer Spikes')

    subplot(313)
    plot(spike_mon_output.t / ms, spike_mon_output.i, '.k')
    xlabel('Time (ms)')
    ylabel('Output layer neuron index')
    title('Output Layer Spikes')

    tight_layout()
    
    #show()


if __name__ == "__main__":
    main()
