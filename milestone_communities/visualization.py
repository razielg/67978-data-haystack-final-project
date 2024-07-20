import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


# # Generate a range of distinct colors using a colormap
# def generate_colors(num_colors):
#     cmap = plt.get_cmap('tab20', num_colors)  # Use 'tab20' colormap for a range of colors
#     return [cmap(i / num_colors) for i in range(num_colors)]


base_colors = [
    'red', 'green', 'blue', 'orange', 'purple', 'brown',
    'pink', 'grey', 'cyan', 'magenta', 'yellow', 'black'
]


# Generate additional colors if needed by mixing base colors
def generate_colors(num_colors):
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]

    colors = base_colors[:]
    while len(colors) < num_colors:
        # Mix existing colors to generate new colors
        for color in base_colors:
            new_color = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(base_colors)))[:, :3]
            new_color = list(new_color[np.random.randint(0, len(new_color))])
            if len(colors) < num_colors:
                colors.append(mcolors.to_hex(new_color))
            else:
                break
    return colors


def plot_graph_with_color(connection_matrix, communities):
    G = nx.from_numpy_array(np.copy(connection_matrix))

    # # Define a color for each community
    # colors = ['red', 'blue', 'green']
    # Generate a color map with a color for each community
    num_communities = len(communities)
    # Generate a color map with a color for each community
    colors = generate_colors(num_communities)

    # Create a dictionary to map node to its community color
    node_color_map = {}
    for i, community in enumerate(communities):
        color = colors[i % len(colors)]  # Cycle through colors if more communities than colors
        for node in community:
            node_color_map[node] = color

    # Create a color list for the graph nodes
    color_list = [node_color_map.get(node, 'grey') for node in G.nodes()]

    # Draw the graph with the node colors
    pos = nx.spring_layout(G)  # or use any other layout
    nx.draw(G, pos, node_color=color_list, with_labels=True, node_size=500, font_size=10, font_color='white')

    # Optionally, add a legend
    patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Community {i + 1}')
        for i, color in enumerate(colors)]
    plt.legend(handles=patches, loc='best')

    # Show the plot
    plt.show()
