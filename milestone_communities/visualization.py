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

shapes = ['o', '^', 's']
shape_labels = ['Coalition', 'Opposition', 'Other']  # Meaningful labels for shapes

partition_heads = ['Benjamin Netanyahu',
                   'Yair Lapid',
                   'Benjamin Gantz',
                   'Ayman Odeh',
                   'Aryeh Machluf Deri',
                   'Yakov Litzman',
                   'Avigdor Liberman',
                   'Mansour Abbas',
                   'Nitzan Horowitz',
                   'Naftali Bennett',
                   'Yoaz Hendel',
                   'Amir Peretz',
                   'Bezalel Smotrich',
                   'Orly Levi-Abekasis',
                   'Rafael Peretz']


def get_partition_heads_node_id(graph):
    """
    Given a graph and a list of partition heads, returns a dictionary mapping
    node numbers to their corresponding names if the name is in partition_heads.

    Parameters:
    - graph: A NetworkX graph object with node attributes including 'name'.
    - partition_heads: A list of names to look for in the graph.

    Returns:
    - A dictionary where the key is the node number and the value is the node's 'name'.
    """
    name_to_node_map = {}

    for node in graph.nodes:
        node_data = graph.nodes[node]
        if node_data['name'] in partition_heads:
            name_to_node_map[node] = node_data['name']

    return name_to_node_map
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

def get_random_coal_oppos(list_of_lists):
    import random
    # Flatten the list of lists
    flat_list = [num for sublist in list_of_lists for num in sublist]
    total_size = len(flat_list)

    # Generate new random numbers from 1 to the size of the flattened list
    new_numbers = list(range(1, total_size + 1))
    random.shuffle(new_numbers)

    # Create a new list of lists with the same structure as the original
    new_list_of_lists = []
    index = 0
    for sublist in list_of_lists:
        new_sublist = new_numbers[index:index + len(sublist)]
        new_list_of_lists.append(new_sublist)
        index += len(sublist)

    return new_list_of_lists

def get_coal_oppos_from_G(G):
    # Initialize lists for nodes based on is_coalition value
    coalition_nodes = []
    opposition_nodes = []

    for node in G.nodes(data=True):
        node_id, attributes = node
        if attributes['is_coalition'] == 1:
            coalition_nodes.append(node_id)
        else:
            opposition_nodes.append(node_id)

    # Create the final list
    result = [coalition_nodes, opposition_nodes]

    return result


def plot_graph_with_color(G: nx.Graph, communities):
    partition_heads_with_node_id = get_partition_heads_node_id(G)
    partition_heads_node_id_list = list(partition_heads_with_node_id.keys())

    # Generate a color map with a color for each community
    num_communities = len(communities)
    colors = generate_colors(num_communities)

    # Create a dictionary to map node to its community color
    node_color_map = {}
    for i, community in enumerate(communities):
        color = colors[i % len(colors)]  # Cycle through colors if more communities than colors
        for node in community:
            node_color_map[node] = color

    coal_oppos = get_coal_oppos_from_G(G)  # get_random_coal_oppos(communities)
    node_shape_map = {}
    for i, partition in enumerate(coal_oppos):
        shape = shapes[i % len(shapes)]  # Cycle through shapes if more communities than shapes
        for node in partition:
            if node in partition_heads_node_id_list:
                node_shape_map[node] = '*'  # Set shape to star for partition heads
            else:
                node_shape_map[node] = shape

    # Define sizes
    default_size = 200
    star_size = 6 * default_size

    # Create a dictionary for node sizes
    node_size_map = {node: star_size if node in partition_heads_node_id_list else default_size for node in G.nodes}

    # Set figure size to be larger
    plt.figure(figsize=(20, 15))

    # Draw the graph with the node colors and shapes
    pos = nx.spring_layout(G, k=0.1)  # Use spring layout with a lower k to spread nodes further

    # Draw edges with a very light transparent color
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3)  # Adjust alpha for transparency

    # Draw nodes with shapes
    for shape in shapes + ['*']:  # Include star shape
        # Get nodes of the current shape
        shape_nodes = [node for node in G.nodes if node_shape_map.get(node) == shape]
        shape_colors = [node_color_map[node] for node in shape_nodes]
        shape_sizes = [node_size_map[node] for node in shape_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=shape_nodes, node_color=shape_colors, node_shape=shape, node_size=shape_sizes)

    # Draw labels for nodes that are in partition_heads_node_id_list
    labels = {node: G.nodes[node]['name'] for node in partition_heads_node_id_list}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_family='sans-serif', font_weight='bold')

    # Create legend handles for colors
    color_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Community {i + 1}')
        for i, color in enumerate(colors)
    ]

    # Create legend handles for shapes
    shape_patches = [
        plt.Line2D([0], [0], marker=shapes[i], color='w', markerfacecolor='grey', markersize=10, label=label)
        for i, label in enumerate(shape_labels)
    ]
    # Add star to legend
    shape_patches.append(
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='grey', markersize=15, label='Partition Heads')
    )

    # Combine legend handles and add legend
    patches = color_patches + shape_patches
    plt.legend(handles=patches, loc='best', fontsize=20)
    plt.title("Colored Communities VS Shaped Coalition/Opposition", fontsize=30)

    # Show the plot
    plt.show()

