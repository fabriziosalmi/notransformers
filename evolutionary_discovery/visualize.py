from graphviz import Digraph

def visualize_genome(genome, filename="best_genome.png"):
    dot = Digraph(format='png')
    # Nodi
    for i, node in enumerate(genome.nodes):
        dot.node(str(i), repr(node))
    # Connessioni
    for src, dst, slot in genome.connections:
        dot.edge(str(src), str(dst))
    dot.render(filename, cleanup=True)
    print(f"Grafo del genoma salvato in {filename}")
