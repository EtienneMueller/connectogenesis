import nest
import numpy as np
import matplotlib.pyplot as plt

# create a spatial population
#pos = nest.spatial.grid(shape=[11, 11], extent=[11., 11.])
#pos = nest.spatial.free(pos=[[-0.5, -0.5], [-0.25, -0.25], [0.75, 0.75]])
pos = nest.spatial.free(pos=[[-0.5, -0.5, -0.5], [-0.25, -0.25, -0.25], [0.75, 0.75, 0.75]])

num_neurons = 20

# x_coords = np.random.uniform(-10, 10, num_neurons)
# y_coords = np.random.uniform(-10, 10, num_neurons)
# z_coords = np.random.uniform(-10, 10, num_neurons)

xyz = np.random.rand(num_neurons, 3).tolist()
pos = nest.spatial.free(pos=xyz)


s_nodes = nest.Create('iaf_psc_alpha', positions=pos)

# plot layer with all its nodes
nest.PlotLayer(s_nodes)
plt.show()