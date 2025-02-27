import pydot

# Create an undirected graph
graph = pydot.Dot(graph_type='graph')

# Add nodes
nodes = ['A', 'B', 'C', 'D', 'E']
for node in nodes:
    graph.add_node(pydot.Node(node))

# Add edges
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        graph.add_edge(pydot.Edge(nodes[i], nodes[j]))

# Save the graph to a PNG file
graph.write_png('./graphs/pcstep0graph.png')
