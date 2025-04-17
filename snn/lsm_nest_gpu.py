import nestgpu as ngpu
import numpy as np
import numpy.ma as ma
import pylab


def create_iaf_psc_exp(n_E, n_I):
    nodes = ngpu.Create('iaf_psc_exp', n_E + n_I,
                        {'C_m': 30.0,  # 1.0,
                         'tau_m': 30.0,  # Membrane time constant in ms
                         'E_L': 0.0,
                         'V_th': 15.0,  # Spike threshold in mV
                         'tau_syn_ex': 3.0,
                         'tau_syn_in': 2.0,
                         'V_reset': 13.8})

    ngpu.SetStatus(nodes, [{'I_e': 14.5} for _ in nodes])
    # ngpu.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    return nodes[:n_E], nodes[n_E:]


def connect_tsodyks(nodes_E, nodes_I):
    f0 = 10.0

    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)
    n_syn_exc = 2  # number of excitatory synapses per neuron
    n_syn_inh = 1  # number of inhibitory synapses per neuron

    w_scale = 10.0
    J_EE = w_scale * 5.0  # strength of E->E synapses [pA]
    J_EI = w_scale * 25.0  # strength of E->I synapses [pA]
    J_IE = w_scale * -20.0  # strength of inhibitory synapses [pA]
    J_II = w_scale * -20.0  # strength of inhibitory synapses [pA]

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))

    def gen_syn_param(tau_psc, tau_fac, tau_rec, U):
        return {"tau_psc": tau_psc,
                "tau_fac": tau_fac,  # facilitation time constant in ms
                "tau_rec": tau_rec,  # recovery time constant in ms
                "U": U,  # utilization
                "u": get_u_0(U, tau_rec, tau_fac),
                "x": get_x_0(U, tau_rec, tau_fac),
                }

    def connect(src, trg, J, n_syn, syn_param):
        ngpu.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     dict({'model': 'tsodyks_synapse', 'delay': delay,
                           'weight': {"distribution": "normal_clipped", "mu": J, "sigma": 0.7 * abs(J),
                                      "low" if J >= 0 else "high": 0.
                           }},
                          **syn_param))

    connect(nodes_E, nodes_E, J_EE, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1.0, tau_rec=813., U=0.59))
    connect(nodes_E, nodes_I, J_EI, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1790.0, tau_rec=399., U=0.049))
    connect(nodes_I, nodes_E, J_IE, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=376.0, tau_rec=45., U=0.016))
    connect(nodes_I, nodes_I, J_II, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=21.0, tau_rec=706., U=0.25))


def inject_noise(nodes_E, nodes_I):
    p_rate = 25.0  # this is used to simulate input from neurons around the populations
    J_noise = 1.0  # strength of synapses from noise input [pA]
    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

    noise = ngpu.Create('poisson_generator', 1, {'rate': p_rate})

    ngpu.Connect(noise, nodes_E + nodes_I, syn_spec={'model': 'static_synapse',
                                                     'weight': {
                                                         'distribution': 'normal',
                                                         'mu': J_noise,
                                                         'sigma': 0.7 * J_noise
                                                     },
                                                     'delay': dict(distribution='normal_clipped',
                                                                   mu=10., sigma=20.,
                                                                   low=3., high=200.)
    })


class LSM(object):
    def __init__(self, n_exc, n_inh, n_rec,
                 create=create_iaf_psc_exp, connect=connect_tsodyks, inject_noise=inject_noise):

        neurons_exc, neurons_inh = create(n_exc, n_inh)
        connect(neurons_exc, neurons_inh)
        inject_noise(neurons_exc, neurons_inh)

        self.exc_nodes = neurons_exc
        self.inh_nodes = neurons_inh
        self.inp_nodes = neurons_exc
        self.rec_nodes = neurons_exc[:n_rec]

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_rec = n_rec

        self._rec_detector = ngpu.Create('spike_detector', 1)

        ngpu.Connect(self.rec_nodes, self._rec_detector)

    def get_states(self, times, tau):
        spike_times = get_spike_times(self._rec_detector, self.rec_nodes)
        return LSM._get_liquid_states(spike_times, times, tau)

    @staticmethod
    def compute_readout_weights(states, targets, reg_fact=0):
        """
        Train readout with linear regression
        :param states: numpy array with states[i, j], the state of neuron j in example i
        :param targets: numpy array with targets[i], while target i corresponds to example i
        :param reg_fact: regularization factor; 0 results in no regularization
        :return: numpy array with weights[j]
        """
        if reg_fact == 0:
            w = np.linalg.lstsq(states, targets)[0]
        else:
            w = np.dot(np.dot(pylab.inv(reg_fact * pylab.eye(np.size(states, 1)) + np.dot(states.T, states)),
                              states.T),
                       targets)
        return w

    @staticmethod
    def compute_prediction(states, readout_weights):
        return np.dot(states, readout_weights)

    @staticmethod
    def _get_liquid_states(spike_times, times, tau, t_window=None):
        n_neurons = np.size(spike_times, 0)
        n_times = np.size(times, 0)
        states = np.zeros((n_times, n_neurons))
        if t_window is None:
            t_window = 3 * tau
        for n, spt in enumerate(spike_times):
            # TODO state order is reversed, as windowed_events are provided in reversed order
            for i, (t, window_spikes) in enumerate(windowed_events(np.array(spt), times, t_window)):
                states[n_times - i - 1, n] = sum(np.exp(-(t - window_spikes) / tau))
        return states


def get_spike_times(spike_rec, rec_nodes):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [ms]
    """
    events = ngpu.GetStatus(spike_rec)[0]['events']
    spikes = []
    for i in rec_nodes:
        idx = np.where(events['senders'] == i)
        spikes.append(events['times'][idx])
    return spikes


def windowed_events(events, window_times, window_size):
    """
    Generate subsets of events which belong to given time windows.

    Assumptions:
    * events are sorted
    * window_times are sorted

    :param events: one-dimensional, sorted list of event times
    :param window_times: the upper (exclusive) boundaries of time windows
    :param window_size: the size of the windows
    :return: generator yielding (window_time, window_events)
    """
    for window_time in reversed(window_times):
        events = events[events < window_time]
        yield window_time, events[events > window_time - window_size]


def poisson_generator(rate, t_start=0.0, t_stop=1000.0, rng=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)

    Examples:
        >> gen.poisson_generator(50, 0, 1000)

    See also:
        inh_poisson_generator
    """

    if rng is None:
        rng = np.random

    # less wasteful than double length method above
    n = (t_stop - t_start) / 1000.0 * rate
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    number = int(number)
    if number > 0:
        isi = rng.exponential(1.0 / rate, number) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun
        t_last = spikes[-1] + rng.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    return spikes


def generate_stimulus_xor(stim_times, gen_burst, n_inputs=2):
    inp_states = np.random.randint(2, size=(n_inputs, np.size(stim_times)))
    inp_spikes = []

    for times in ma.masked_values(inp_states, 0) * stim_times:
        # for each input (neuron): generate spikes according to state (=1) and stimulus time-grid
        spikes = np.concatenate([t + gen_burst() for t in times.compressed()])

        # round to simulation precision
        spikes *= 10
        spikes = spikes.round() + 1.0
        spikes = spikes / 10.0

        inp_spikes.append(spikes)

    # astype(int) could be omitted, because False/True has the same semantics
    targets = np.logical_xor(*inp_states).astype(int)

    return inp_spikes, targets


def inject_spikes(inp_spikes, neuron_targets):
    spike_generators = ngpu.Create("spike_generator", len(inp_spikes))

    for sg, sp in zip(spike_generators, inp_spikes):
        ngpu.SetStatus([sg], {'spike_times': sp})

    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)
    C_inp = 100  # int(N_E / 20)  # number of outgoing input synapses per input neuron

    ngpu.Connect(spike_generators, neuron_targets,
                 {'rule': 'fixed_outdegree',
                  'outdegree': C_inp},
                 {'model': 'static_synapse',
                  'delay': delay,
                  'weight': {'distribution': 'uniform',
                             'low': 2.5 * 10 * 5.0,
                             'high': 7.5 * 10 * 5.0}
                  })


def main():
    ngpu.SetKernelStatus({'print_time': True, 'local_num_threads': 11})

    sim_time = 200000

    # stimulus
    stim_interval = 300
    stim_length = 50
    stim_rate = 200  # [1/s]

    readout_delay = 10

    stim_times = np.arange(stim_interval, sim_time - stim_length - readout_delay, stim_interval)
    readout_times = stim_times + stim_length + readout_delay

    def gen_stimulus_pattern(): return poisson_generator(stim_rate, t_stop=stim_length)

    inp_spikes, targets = generate_stimulus_xor(stim_times, gen_burst=gen_stimulus_pattern)

    lsm = LSM(n_exc=1000, n_inh=250, n_rec=500)

    inject_spikes(inp_spikes, lsm.inp_nodes)

    ngpu.Simulate(sim_time)

    readout_times = readout_times[5:]
    targets = targets[5:]

    states = lsm.get_states(readout_times, tau=20)

    # weird
    states = np.hstack([states, np.ones((np.size(states, 0), 1))])

    n_examples = np.size(targets, 0)
    n_examples_train = int(n_examples * 0.8)

    train_states, test_states = states[:n_examples_train, :], states[n_examples_train:, :]
    train_targets, test_targets = targets[:n_examples_train], targets[n_examples_train:]

    readout_weights = lsm.compute_readout_weights(train_states, train_targets, reg_fact=5.0)

    def classify(prediction):
        return (prediction >= 0.5).astype(int)

    train_prediction = lsm.compute_prediction(train_states, readout_weights)
    train_results = classify(train_prediction)

    test_prediction = lsm.compute_prediction(test_states, readout_weights)
    test_results = classify(test_prediction)

    print("simulation time: {}ms".format(sim_time))
    print("number of stimuli: {}".format(len(stim_times)))
    print("size of each state: {}".format(np.size(states, 1)))

    print("---------------------------------------")

    def eval_prediction(prediction, targets, label):
        n_fails = sum(abs(prediction - targets))
        n_total = len(targets)
        print("mismatched {} examples: {:d}/{:d} [{:.1f}%]".format(label, n_fails, n_total, float(n_fails) / n_total * 100))

    eval_prediction(train_results, train_targets, "training")
    eval_prediction(test_results, test_targets, "test")


if __name__ == "__main__":
    main()