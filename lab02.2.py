import networkx as nx
import matplotlib.pyplot as plt


# step 2: A graph from the file
G0 = nx.read_adjlist('graph.txt')
pos = nx.circular_layout(G0)
nx.draw(G0, pos, node_color='w', edgecolors='black', with_labels=True)
plt.show()


# step 3: A graph by the methods
G = nx.Graph()
G.add_node(1, pos=(1, 2))
G.add_node(2, pos=(2, 2))
G.add_node(3, pos=(3, 1))
G.add_node(4, pos=(3, 3))
G.add_node(5, pos=(4, 1))
G.add_node(6, pos=(4, 3))
G.add_node(7, pos=(5, 1))
G.add_node(8, pos=(7, 3))
G.add_node(9, pos=(9, 1))
G.add_node(10, pos=(7, 2))
G.add_node(11, pos=(6, 1.75))
G.add_node(12, pos=(8, 1.75))
G.add_edge(3, 4)
G.add_edge(5, 6)
G.add_edges_from((i, j) for i in range(7, 13) for j in range(7, 13) if i != j and {i, j} != {11, 12})
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, node_color='w', edgecolors='black', with_labels=True)
plt.show()


# step 4: The analyse of the components
i = 1
for nodes in nx.connected_components(G):
    g = nx.subgraph(G, nodes)
    print(f'in the {i}th connected component of the graph:')
    print(f'The amount of nodes is {len(nodes)}.')
    print(f'The amount of nodes is {len(nx.edges(g, nodes))}.')
    for v in g:
        print(f'Degree of node {v} equals {nx.degree(g, v)}.')
        print(f'Eccentricity of node {v} equals {nx.eccentricity(g, v)}.')
    print(f'Radius equals {nx.radius(g)}.')
    print(f'Diameter equals {nx.diameter(g)}.')
    i += 1


# step 5: Searching for the diameters
dim_nodes = set()
dim_edges = set()
for nodes in nx.connected_components(G):
    g = nx.subgraph(G, nodes)
    if nx.is_empty(g):
        continue
    is_dim_found = 0
    d = nx.diameter(g)
    for v in nodes:
        if is_dim_found:
            break
        for u in nodes:
            dim = nx.shortest_path(g, v, u)
            if len(dim) == d + 1:
                is_dim_found = 1
                for w in dim:
                    dim_nodes.add(w)
                for i in range(d):
                    dim_edges.add((dim[i], dim[i+1]))
                break
color_node_map = ['green' if node in dim_nodes else 'white' for node in G]
color_edge_map = ['blue' if ((u, v) in dim_edges or (v, u) in dim_edges) else 'black' for u, v in G.edges()]
nx.draw(G, pos, node_color=color_node_map, edgecolors='black', edge_color=color_edge_map, with_labels=True)
plt.show()


# step 6: Building of spanning tree
D = set(nx.dfs_edges(G))
color_edge_map = ['red' if edge in D else 'black' for edge in G.edges()]
nx.draw(G, pos, node_color='white', edgecolors='black', edge_color=color_edge_map, with_labels=True)
plt.show()
