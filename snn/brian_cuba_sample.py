from brian2 import *
import matplotlib.pyplot as plt
import time


# see: https://www.researchgate.net/figure/The-CUBA-network-in-Brian-with-code-on-the-left-neuron-model-equations-at-the-top-right_fig1_23712221
# This script defines a randomly connected network of 4000 leaky integrate-and-fire neurons with exponential synaptic currents, partitioned into a group of 3200 excitatory neurons and 800 inhibitory neurons. 
# The subgroup() method keeps track of which neurons have been allocated to subgroups and allocates the next available neurons. 
# The process starts from neuron 0, so Pe has neurons 0 through 3199 and Pi has neurons 3200 through 3999. 
# The script outputs a raster plot showing the spiking activity of the network for a few hundred ms. 
# This is Brian's implementation of the current-based (CUBA) network model used as one of the benchmarks in Brette et al. (2007), based on the network studied in Vogels and Abbott (2005). 
# The simulation takes 3–4 s on a typical PC (1.8 GHz Pentium), for 1 s of biological time (with dt = 0.1 ms). 
# The variables ge and gi are not conductances, we follow the variable names used in Brette et al. (2007). 
# The code :volt in the equations means that the unit of the variable being defined (V, ge and gi) has units of volts.


num_neurons = 4000
sim_duration = 1


def plot_raster(s_mon):
    # raster plot
    plot(s_mon.t/ms, s_mon.i, ',k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    show()


def main():
    taum = 20*ms
    taue = 5*ms
    taui = 10*ms
    Vt = -50*mV
    Vr = -60*mV
    El = -49*mV

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    P = NeuronGroup(num_neurons, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                    method='exact')
    P.v = 'Vr + rand() * (Vt - Vr)'
    P.ge = 0*mV
    P.gi = 0*mV

    we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
    wi = (-20*4.5/10)*mV # inhibitory synaptic weight
    Ce = Synapses(P, P, on_pre='ge += we')
    Ci = Synapses(P, P, on_pre='gi += wi')
    Ce.connect('i<3200', p=0.02)
    Ci.connect('i>=3200', p=0.02)

    s_mon = SpikeMonitor(P)

    start_time = time.time()
    run(sim_duration * second)
    print(f"--- {time.time() - start_time}s ---")

    plot_raster(s_mon)


if __name__ == "__main__":
    main()
