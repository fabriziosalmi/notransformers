from graphviz import Digraph

def visualize_genome(genome, filename="best_genome.png"):
    """Visualize a grammar-based genome architecture.
    
    Args:
        genome: ModelGenome instance with grammar-based architecture
        filename: Output filename for the graph
    """
    dot = Digraph(format='png', comment='Grammar-Based Neural Architecture')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Get architecture from genome
    architecture = genome.build_from_grammar()
    
    if not architecture:
        dot.node('empty', 'Empty Architecture', shape='box')
        dot.render(filename.replace('.png', ''), cleanup=True)
        print(f"Graph saved to {filename}")
        return
    
    # Add input node
    dot.node('input', 'Input', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Add architecture layers
    prev_node = 'input'
    residual_starts = []
    
    for i, layer in enumerate(architecture):
        node_id = f'layer_{i}'
        
        # Style based on layer type
        if layer in ['Conv1D', 'GRU', 'Linear']:
            shape = 'box'
            color = 'lightgreen'
        elif layer in ['ReLU', 'Tanh', 'GELU', 'Sigmoid']:
            shape = 'diamond'
            color = 'lightyellow'
        elif layer in ['LayerNorm', 'BatchNorm']:
            shape = 'oval'
            color = 'lightcoral'
        elif layer == 'residual':
            shape = 'box'
            color = 'orange'
        elif layer == 'identity':
            continue  # Skip identity connections
        else:
            shape = 'box'
            color = 'lightgray'
        
        dot.node(node_id, layer, shape=shape, style='filled', fillcolor=color)
        
        # Add edge from previous layer
        if layer == 'residual' and residual_starts:
            # Add residual connection
            start_node = residual_starts.pop()
            dot.edge(start_node, node_id, style='dashed', color='red', label='skip')
            dot.edge(prev_node, node_id)
        else:
            dot.edge(prev_node, node_id)
        
        # Track potential residual starts (layers that could have residual connections)
        if layer in ['Conv1D', 'GRU', 'Linear']:
            residual_starts.append(node_id)
        
        prev_node = node_id
    
    # Add output node
    dot.node('output', 'Output', shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge(prev_node, 'output')
    
    # Render and save
    dot.render(filename.replace('.png', ''), cleanup=True)
    print(f"Graph saved to {filename}")

