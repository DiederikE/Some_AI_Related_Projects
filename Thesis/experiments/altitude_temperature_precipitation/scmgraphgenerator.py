import networkx as nx
import matplotlib.pyplot as plt

# Maak een leeg gericht netwerk
G = nx.DiGraph()

# Voeg knooppunten toe aan het netwerk
G.add_node('D')
G.add_node('A')
G.add_node('B')
G.add_node('C')
G.add_node('f', shape='s')
G.add_node('g', shape='s')
G.add_node('h', shape='s')

# Voeg randen toe aan het netwerk
G.add_edge('D', 'f')
G.add_edge('f', 'A')
G.add_edge('A', 'g')
G.add_edge('g', 'B')
G.add_edge('A', 'h')
G.add_edge('h', 'C')

# Teken het netwerk
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
