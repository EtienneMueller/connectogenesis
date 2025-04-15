import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random

# Create an empty graph
G = nx.Graph()

# Add 50 nodes to the graph
num_nodes = 50
G.add_nodes_from(range(num_nodes))

# Randomly connect each node to another node
for node in range(num_nodes):
    target = random.choice(range(num_nodes))
    while target == node:
        target = random.choice(range(num_nodes))
    G.add_edge(node, target)

# Initialize the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
pos = nx.spring_layout(G, dim=3)  # Position the nodes using the spring layout in 3D

# Function to update the graph for the animation
def update(num):
    ax.clear()
    ax.set_title("Dynamic 3D Graph with Activity Flowing Through It")
    
    # Extract positions of nodes
    x_vals = [pos[i][0] for i in range(num_nodes)]
    y_vals = [pos[i][1] for i in range(num_nodes)]
    z_vals = [pos[i][2] for i in range(num_nodes)]

    # Draw nodes
    ax.scatter(x_vals, y_vals, z_vals, c='skyblue', s=100)

    # Draw edges
    for edge in G.edges:
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='gray')
    
    # Randomly select a node to start the activity
    start_node = random.choice(list(G.nodes))
    neighbors = list(G.neighbors(start_node))
    if neighbors:
        end_node = random.choice(neighbors)
        
        # Highlight the nodes and the edge with activity
        ax.scatter([pos[start_node][0]], [pos[start_node][1]], [pos[start_node][2]], c='red', s=200)
        ax.scatter([pos[end_node][0]], [pos[end_node][1]], [pos[end_node][2]], c='green', s=200)
        ax.plot([pos[start_node][0], pos[end_node][0]],
                [pos[start_node][1], pos[end_node][1]],
                [pos[start_node][2], pos[end_node][2]], color='red', linewidth=3)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=500, repeat=True)

# Display the animation
plt.show()

