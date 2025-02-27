import pydot

# Create an undirected graph
graph = pydot.Dot(graph_type='graph')

# Add nodes
nodes = ['A', 'B', 'C', 'D', 'E']
for node in nodes:
    graph.add_node(pydot.Node(node))

# Add edges
edges = [('A', 'B'), ('C', 'B'), ('B', 'D'), ('B', 'E')]
for edge in edges:
    graph.add_edge(pydot.Edge(edge[0], edge[1]))

# Save the graph to a PNG file
graph.write_png('./graphs/pcstep0graph.png')
