import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np


num_nodes = 20


def rearrange_nodes(step, total_steps, num_nodes):
    left_line = [(0, i) for i in range(num_nodes)]
    right_line = [(1, i) for i in range(num_nodes)]
    
    new_pos = {}
    for i in range(num_nodes):
        if (step - 1) * 5 <= i < step * 5:
            new_pos[i] = right_line[i]
        else:
            new_pos[i] = left_line[i]
    return new_pos


def calculate_new_values(G, num_nodes, step, node_values, weights):
    new_values = node_values.copy()
    # left_nodes = [i for i in range(num_nodes) if (step - 1) > i or i >= step]
    # left_nodes = [i for i in range(num_nodes) if (step - 1) * 2 > i or i >= step * 2]
    left_nodes = [i for i in range(num_nodes) if (step - 1) * 5 > i or i >= step * 5]
    # right_nodes = [i for i in range((step - 1), step)]
    # right_nodes = [i for i in range((step - 1) * 2, step * 2)]
    right_nodes = [i for i in range((step - 1) * 5, step * 5)]
    
    for target_node in right_nodes:
        connected_nodes = [source for source, target in G.edges() if target == target_node and source in left_nodes]
        if connected_nodes:
            new_value = sum(node_values[source] * weights[(source, target_node)] for source in connected_nodes)
            new_values[target_node] = new_value
    
    return new_values


def main():
    G = nx.gnm_random_graph(num_nodes, 40)  # 20 nodes, 40 edges
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in G.nodes()}
    node_values = np.random.randint(1, 10, num_nodes)
    weights = {edge: random.uniform(0.1, 1.0) for edge in G.edges()}

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray')
    node_labels = {i: f"{node_values[i]}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.title("Initial Random Graph with Node Values")
    plt.show()

    # rearranging, calculate new values, plot
    total_steps = 4
    for step in range(1, total_steps + 1):
        pos = rearrange_nodes(step, total_steps, num_nodes)
        node_values = calculate_new_values(G, num_nodes, step, node_values, weights)
        
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray')
        node_labels = {i: f"{node_values[i]:.2f}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        plt.title(f"Rearranged Graph - Step {step} with Node Values")
        plt.show()


if __name__ == "__main__":
    main()
