import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


input_size = 1
reservoir_size = 100
output_size = 1
spectral_radius = 0.95
sparsity = 0.1
input_connectivity = 0.1


def generate_weights():
    W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
    W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_res)))
    W_res[np.random.rand(*W_res.shape) > sparsity] = 0

    W_in = np.random.rand(reservoir_size, input_size) - 0.5
    input_indices = np.random.choice(reservoir_size, int(reservoir_size * input_connectivity), replace=False)
    return W_res, W_in, input_indices


def main():
    W_res, W_in, input_indices = generate_weights()

    # Create a directed graph
    G = nx.DiGraph()
    G.add_node('Input', pos=(0, 0))  # Add nodes for input, reservoir, and output layers

    # reservoir nodes with random positions
    reservoir_positions = {f'R{i}': (np.random.rand() * 10, np.random.rand() * 10) for i in range(reservoir_size)}
    for i, pos in reservoir_positions.items():
        G.add_node(i, pos=pos)
    G.add_node('Output', pos=(12, 5))  # output node

    # input to reservoir connections
    for i in input_indices:
        G.add_edge('Input', f'R{i}')

    # reservoir to reservoir connections
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if W_res[i, j] != 0:
                G.add_edge(f'R{i}', f'R{j}')

    # reservoir to output connections
    for i in range(reservoir_size):
        G.add_edge(f'R{i}', 'Output')

    pos = nx.get_node_attributes(G, 'pos')  # positions for all nodes

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=10)
    plt.title('Reservoir Neural Network Visualization')
    plt.show()


if __name__ == "__main__":
    main()
