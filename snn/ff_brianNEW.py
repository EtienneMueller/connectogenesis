import time
from brian2 import *


def main():
    start_scope()

    tau = 10*ms
    v_rest = -70*mV
    v_reset = -65*mV
    v_threshold = -50*mV
    refractory_period = 5*ms
    input_rate = 10*Hz
    simulation_time = 100*ms

    input_group = PoissonGroup(40000, rates=input_rate)

    eqs = '''
    dv/dt = (v_rest - v) / tau : volt (unless refractory)
    '''

    output_group = NeuronGroup(40000, 
                            eqs, 
                            threshold='v>v_threshold', 
                            reset='v=v_reset', 
                            refractory=refractory_period, 
                            method='exact')
    output_group.v = v_rest

    synapses = Synapses(input_group, output_group, on_pre='v_post += 1*mV')
    synapses.connect(p=0.9)  # Connect neurons with a probability of 0.5

    spike_monitor_input = SpikeMonitor(input_group)
    spike_monitor_output = SpikeMonitor(output_group)
    state_monitor_output = StateMonitor(output_group, 'v', record=True)

    start_time = time.time()
    run(simulation_time)
    runtime = time.time() - start_time
    print(f"{runtime}s")

    figure(figsize=(12, 4))
    subplot(121)
    plot(spike_monitor_input.t/ms, spike_monitor_input.i, '.k')
    xlabel('Time (ms)')
    ylabel('Input neuron index')
    title('Input Layer Spikes')

    subplot(122)
    plot(spike_monitor_output.t/ms, spike_monitor_output.i, '.k')
    xlabel('Time (ms)')
    ylabel('Output neuron index')
    title('Output Layer Spikes')

    show()


if __name__ == "__main__":
    main()
