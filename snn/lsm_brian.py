import numpy as np
import matplotlib.pyplot as plt
import time
from brian2 import *


N_input = 100
N_reservoir = 4000
N_output = 10

tau = 10*ms
delay = 1*ms

sim_duration = 1  # in ms

show_plot = False


def spike_plot(reservoir_mon):
    # Plot results
    # plt.figure(figsize=(12, 4))
    # plt.subplot(131)
    # plt.plot(input_mon.t/ms, input_mon.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Input neuron index')

    plt.subplot()
    plt.plot(reservoir_mon.t/ms, reservoir_mon.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Reservoir neuron index')

    # plt.subplot(133)
    # plt.plot(output_mon.t/ms, output_mon.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Output neuron index')

    plt.tight_layout()
    plt.show()


def estimate_time(runtime):
    print(f"Runtime to simulate {sim_duration}s of {N_reservoir} LIF neurons: {runtime:.2f}s")
    estimate = (runtime * (100000 / N_reservoir) ** 2) / sim_duration * 600
    days = int(estimate // 86400)
    hours = int((estimate % 86400) // 3600)
    minutes = int((estimate % 3600) // 60)
    seconds = estimate % 60
    print(f"Estimate for 100k neurons over 600 seconds: {days}d{hours}h{minutes}m{seconds:.2f}s")


def main():
    # input neurons
    input_group = SpikeGeneratorGroup(N_input, 
                                    np.arange(N_input), 
                                    np.random.rand(N_input) * 100*ms)

    # reservoir
    reservoir_group = NeuronGroup(N_reservoir, 
                                'dv/dt = -v/tau : 1',
                                threshold='v > 1', reset='v = 0',
                                method='exact')

    # output
    output_group = NeuronGroup(N_output, 
                            'dv/dt = -v/tau : 1',
                            threshold='v > 1', reset='v = 0',
                            method='exact')

    # synapses
    input_syn = Synapses(input_group, reservoir_group, 
                        'w : 1', on_pre='v += w')
    input_syn.connect(p=0.1)
    input_syn.w = 'rand()'

    reservoir_syn = Synapses(reservoir_group, reservoir_group, 
                            'w : 1', on_pre='v += w')
    reservoir_syn.connect(p=0.1, condition='i != j')
    reservoir_syn.w = 'rand()'

    output_syn = Synapses(reservoir_group, output_group, 
                        'w : 1', on_pre='v += w')
    output_syn.connect(p=0.1)
    output_syn.w = 'rand()'

    # Monitors
    input_mon = SpikeMonitor(input_group)
    reservoir_mon = SpikeMonitor(reservoir_group)
    output_mon = SpikeMonitor(output_group)
    state_mon = StateMonitor(reservoir_group, 'v', record=range(10))

    # Run the simulation
    start_time = time.time()
    #run(100*ms)
    run(sim_duration * second)
    #run(sim_duration * ms)
    runtime = time.time() - start_time
    estimate_time(runtime)

    if show_plot:
        spike_plot(reservoir_mon)


if __name__ == "__main__":
    main()
