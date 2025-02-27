import pydot

# Create an undirected graph
graph = pydot.Dot(graph_type='digraph')

# Add nodes
nodes = ['ma', 'mb', 'mc', 'A', 'B', 'C']
for node in nodes:
    graph.add_node(pydot.Node(node))

# Add edges
edges = [('ma', 'A'), ('mb', 'B'), ('mc', 'C'), ('A', 'mb'), ('A', 'mc')]
for edge in edges:
    graph.add_edge(pydot.Edge(edge[0], edge[1]))

# Save the graph to a PNG file
graph.write_png('./graphs/scmintervention.png')
