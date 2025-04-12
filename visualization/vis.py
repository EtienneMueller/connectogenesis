import nest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import utils.load_s2p as load_s2p
# from brian2 import *


# Suite2p output:
# F.npy:    array of fluorescence traces (ROIs by timepoints)
# Fneu.npy: array of neuropil fluorescence traces (ROIs by timepoints)
# spks.npy: array of deconvolved traces (ROIs by timepoints)
# stat.npy: list of statistics computed for each cell (ROIs by 1)
# ops.npy:  options and intermediate outputs (dictionary)
# iscell.npy: specifies whether an ROI is a cell, first column is 0/1, and 
#             second column is probability that the ROI is a cell based on the default classifier


# stat.npy:
# [{
# 'ypix': array([...]),
# 'lam': array([...]),
# 'xpix': array([...]), 
# 'mrs': np.float32(0.46733832), 
# 'mrs0': np.float64(1.0729838054991534), 
# 'compact': np.float64(0.999999999906802), 
# 'med': [np.float64(451.0), np.float64(257.0)], 
# 'npix': 19, 
# 'radius': np.float64(1.9311087842799621), 
# 'aspect_ratio': np.float64(1.1911410914839986), 
# 'footprint': np.float64(1.3460935510012362), 
# 'npix_norm': np.float32(0.24752475), 
# 'solidity': np.float64(0.9), 
# 'npix_soma': np.int64(9), 
# 'soma_crop': array([...bool...]), 
# 'overlap': array([...bool...]), 
# 'npix_norm_no_crop': np.float32(0.42812076), 
# 'skew': np.float32(1.6404574), 
# 'std': np.float32(175.55121), 
# 'neuropil_mask': array([...])}]


interval_ms = 100
file_path = 'big_data/suite2p_TessM_Looms_20190805_fish3_1_2Hz_SL50_TP606_TP50_St2_NP_range245_step5_exposure10_power50'
#file_path = 'big_data/suite2p_MW_FMR1_Auditory_2Hz_SL50_fish50_20231214'
layer_shift = 15
brigthness_threshold = 0.32
show_connection=False
connection_threshold = 0.9995


nest.ResetKernel()


def zfplot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


def anim(x, y, z, brightness):

    # adjacency matrix
    num_points = len(x)
    connections = np.random.rand(num_points, num_points) > connection_threshold

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # fixed color
    sc = ax.scatter(x, y, z, s=brightness[:, 0]*100, c='blue', cmap='viridis')

    def update(frame):
        if show_connection:
            # draw lines between connected neurons
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if connections[i, j]:
                        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='grey')

        # Update size for each neuron
        # -> set size to zero if brightness is zero
        sizes = brightness[:, frame] * 250
        sc._sizes = sizes
        return sc,

    # def update(frame):
    #     ax.cla()  # Clear the axes to redraw the lines and points
    #     sc = ax.scatter(x, y, z, s=brightness[:, frame] * 100, c='blue')
        
    #     # Draw lines between connected points
    #     for i in range(num_points):
    #         for j in range(i + 1, num_points):
    #             if connections[i, j]:
    #                 ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='grey')
        
    #     # Update the size for each point, set size to zero if brightness is zero
    #     sizes = brightness[:, frame] * 100
    #     colors = ['white' if size == 0 else 'blue' for size in sizes]
        
    #     # Set the very small sizes to be effectively invisible
    #     sizes[sizes == 0] = 0.01
        
    #     sc._sizes = sizes
    #     sc.set_color(colors)
    #     return sc,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=1212, interval=interval_ms, blit=True)

    # Show the animation
    plt.show()


def nest1(x, y, z):
    # Parameters
    num_neurons = 100  # Number of neurons
    neuron_model = 'iaf_psc_alpha'  # Neuron model
    x_coords = np.random.uniform(-10, 10, num_neurons)  # Random x coordinates
    y_coords = np.random.uniform(-10, 10, num_neurons)  # Random y coordinates
    z_coords = np.random.uniform(-10, 10, num_neurons)  # Random z coordinates

    # Create neurons
    neurons = nest.Create(neuron_model, num_neurons)

    # Assign coordinates to neurons
    for i, neuron in enumerate(neurons):
        nest.SetStatus([neuron], {'x': x_coords[i], 'y': y_coords[i], 'z': z_coords[i]})

    # Print neuron coordinates for verification
    for neuron in neurons:
        status = nest.GetStatus([neuron])[0]
        print(f"Neuron ID: {neuron}, x: {status['x']}, y: {status['y']}, z: {status['z']}")


def nest2(x, y, z):
    neuron_coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ]

    # Create neurons
    #neuron_ids = nest.Create("iaf_psc_alpha", len(neuron_coordinates))
    #neuron_ids = nest.Create("iaf_psc_alpha", positions=neuron_coordinates)
    #print(type(neuron_ids))
    #neuron_ids = nest.NodeCollection([0,1,2,3])

    nest.CopyModel('iaf_psc_alpha', 'pyr')
    newtest = nest.Create("pyr", positions=[1.,2.,3.])
    print(nest.GetStatus(newtest))


    neuron_test = nest.Create("iaf_psc_alpha", positions=[1.,2.,3.])
    print(nest.GetStatus(neuron_test))
    nest.SetStatus(neuron_test, {"C_m": 200.})
    print(nest.GetStatus(neuron_test))

    # Assign positions to neurons
    for neuron_id, coord in zip(neuron_ids, neuron_coordinates):
        nest.SetStatus([neuron_id], {"position": coord})

    # Define a function to connect neurons (example: all-to-all)
    def connect_neurons(neuron_ids):
        for i, src in enumerate(neuron_ids):
            for j, tgt in enumerate(neuron_ids):
                if i != j:  # Avoid self-connections
                    nest.Connect([src], [tgt])

    # Connect neurons
    connect_neurons(neuron_ids)

    # Simulate the network
    nest.Simulate(100.0)

    # Print the status of the neurons
    for neuron_id in neuron_ids:
        status = nest.GetStatus([neuron_id])[0]
        print(f"Neuron {neuron_id}: {status}")


def to_brian1(x, y, z):
    # Define number of neurons
    N = 100

    # Create a group of neurons
    neurons = NeuronGroup(N, 'dv/dt = -v / (10*ms) : 1')

    # Initialize neuron coordinates
    x = np.random.rand(N) * 100  # X coordinates between 0 and 100
    y = np.random.rand(N) * 100  # Y coordinates between 0 and 100
    z = np.random.rand(N) * 100  # Z coordinates between 0 and 100

    # Store coordinates as neuron group variables
    neurons.x = x
    neurons.y = y
    neurons.z = z

    # Optionally, use coordinates to define connectivity
    # Example: connect neurons within a certain distance
    synapses = Synapses(neurons, neurons, 'w : 1', on_pre='v += w')
    synapses.connect(condition='sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2 + (z_pre - z_post)**2) < 20')

    # Initialize synapse weights
    synapses.w = 'rand()'

    # Run the simulation
    run(100*ms)

    # Output the coordinates for verification
    print(neurons.x[:10], neurons.y[:10], neurons.z[:10])  # Print first 10 neurons' coordinates


def s2p_to_snn(coordinates):
    pos = nest.spatial.free(pos=coordinates)
    s_nodes = nest.Create('iaf_psc_alpha', positions=pos)

    nest.PlotLayer(s_nodes)
    plt.show()



def main():
    x, y, z, brightness = load_s2p.load_coordinate(file_path, layer_shift, brigthness_threshold)
    x, y, z, brightness = load_s2p.filter(x, y, z, brightness, brigthness_threshold)
    anim(x, y, z, brightness)

    coordinates = [[x, y, z] for x, y, z in zip(x, y, z)]

    s2p_to_snn(coordinates)


if __name__=='__main__':
    main()
