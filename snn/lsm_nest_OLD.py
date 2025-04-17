import nest
import numpy as np


# check: https://github.com/IGITUGraz/LSM/blob/master/lsm/nest/__init__.py
nest.ResetKernel()


num_input_neurons = 100
num_reservoir_neurons = 500
num_readout_neurons = 10
sim_time = 1000.0  # ms
input_rate = 10.0  # Hz


def main():
    input_neurons = nest.Create("poisson_generator", num_input_neurons, params={"rate": input_rate})
    reservoir_neurons = nest.Create("iaf_psc_alpha", num_reservoir_neurons)
    readout_neurons = nest.Create("iaf_psc_alpha", num_readout_neurons)

    # random connec
    conn_dict_input_reservoir = {"rule": "fixed_indegree", "indegree": 10}
    conn_dict_reservoir_reservoir = {"rule": "fixed_indegree", "indegree": 10}
    conn_dict_reservoir_readout = {"rule": "fixed_indegree", "indegree": 10}

    # syn_dict = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}
    syn_dict = {"weight": 1.0, "delay": 1.0}

    # connect layers
    nest.Connect(input_neurons, reservoir_neurons, conn_dict_input_reservoir, syn_dict)
    nest.Connect(reservoir_neurons, reservoir_neurons, conn_dict_reservoir_reservoir, syn_dict)
    nest.Connect(reservoir_neurons, readout_neurons, conn_dict_reservoir_readout, syn_dict)

    # spike detectors
    # input_spike_detectors = nest.Create("spike_detector", num_input_neurons)
    # reservoir_spike_detectors = nest.Create("spike_detector", num_reservoir_neurons)
    # readout_spike_detectors = nest.Create("spike_detector", num_readout_neurons)
    input_spike_detectors = nest.Create("spike_detector", num_input_neurons)
    reservoir_spike_detectors = nest.Create("spike_detector", num_reservoir_neurons)
    readout_spike_detectors = nest.Create("spike_detector", num_readout_neurons)
    # input_spike_detectors = nest.Create(num_input_neurons)
    # reservoir_spike_detectors = nest.Create(num_reservoir_neurons)
    # readout_spike_detectors = nest.Create(num_readout_neurons)

    # connect spike detectors
    nest.Connect(input_neurons, input_spike_detectors)
    nest.Connect(reservoir_neurons, reservoir_spike_detectors)
    nest.Connect(readout_neurons, readout_spike_detectors)

    nest.Simulate(sim_time)

    input_spikes = nest.GetStatus(input_spike_detectors, keys="events")
    reservoir_spikes = nest.GetStatus(reservoir_spike_detectors, keys="events")
    readout_spikes = nest.GetStatus(readout_spike_detectors, keys="events")

    input_spike_times = [spike["times"] for spike in input_spikes]
    reservoir_spike_times = [spike["times"] for spike in reservoir_spikes]
    readout_spike_times = [spike["times"] for spike in readout_spikes]

    print("Input spike times:", input_spike_times)
    print("Reservoir spike times:", reservoir_spike_times)
    print("Readout spike times:", readout_spike_times)


if __name__ == "__main__":
    main()
