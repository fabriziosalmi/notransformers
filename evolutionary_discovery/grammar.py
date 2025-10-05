"""
Grammar-based Neural Architecture Evolution

This grammar defines rules for building modular neural architectures
through a grammatical neuroevolution approach.

Keys are "non-terminals" (abstract symbols like <network>)
Values are lists of possible "expansions" (production rules)
Uppercase tokens (Conv1D, GRU) are concrete "terminals"
Tokens between <> are "non-terminals" to be expanded
"""

GRAMMAR = {
    "<start>": [["<network>"]],
    
    "<network>": [
        ["<block>"],
        ["<network>", "<block>"]  # Una rete può essere estesa con un nuovo blocco
    ],
    
    "<block>": [
        ["<conv_block>"],
        ["<recurrent_block>"],
        ["<dense_block>"]
    ],
    
    "<conv_block>": [
        ["Conv1D", "<activation>", "LayerNorm", "<maybe_residual>"]
    ],
    
    "<recurrent_block>": [
        ["GRU", "<activation>", "LayerNorm", "<maybe_residual>"]
    ],
    
    "<dense_block>": [
        ["Linear", "<activation>", "LayerNorm"]  # Dense blocks typically don't have residual
    ],
    
    "<maybe_residual>": [
        ["residual"],  # Add a residual connection
        ["identity"]   # Do nothing
    ],
    
    "<activation>": [
        ["ReLU"],
        ["Tanh"],
        ["GELU"]
    ]
}


def is_terminal(symbol):
    """Check if a symbol is terminal (doesn't contain < >)"""
    return not (symbol.startswith('<') and symbol.endswith('>'))


def expand_grammar(genes, grammar, max_expansions=50):
    """
    Expands a grammar using a gene sequence
    
    Args:
        genes: List of integers guiding grammatical choices
        grammar: Grammar dictionary with production rules
        max_expansions: Maximum number of expansions to avoid infinite loops
    
    Returns:
        List of terminals representing the final architecture
    """
    # Start with the start symbol
    sequence = ["<start>"]
    gene_index = 0
    expansions = 0
    
    while expansions < max_expansions:
        # Find the first non-terminal
        non_terminal_found = False
        for i, symbol in enumerate(sequence):
            if not is_terminal(symbol):
                non_terminal_found = True
                
                # Get the next gene (with wrapping if necessary)
                if gene_index < len(genes):
                    gene = genes[gene_index]
                    gene_index += 1
                else:
                    # If we run out of genes, use a default value
                    gene = 0
                
                # Choose a production rule
                if symbol in grammar:
                    rules = grammar[symbol]
                    rule_index = gene % len(rules)
                    chosen_rule = rules[rule_index]
                    
                    # Replace the non-terminal with the chosen rule
                    sequence = sequence[:i] + chosen_rule + sequence[i+1:]
                else:
                    # If the symbol is not in the grammar, remove it
                    sequence = sequence[:i] + sequence[i+1:]
                
                expansions += 1
                break
        
        if not non_terminal_found:
            # All symbols are terminals
            break
    
    # Filter only final terminals
    final_sequence = [symbol for symbol in sequence if is_terminal(symbol)]
    return final_sequence


def print_grammar_info():
    """Print information about the defined grammar"""
    print("=== NEUROEVOLUTION GRAMMAR ===")
    print("Available non-terminals:")
    for key in GRAMMAR.keys():
        print(f"  {key}: {len(GRAMMAR[key])} rules")
    
    print("\nUsed terminals:")
    terminals = set()
    for rules in GRAMMAR.values():
        for rule in rules:
            for symbol in rule:
                if is_terminal(symbol):
                    terminals.add(symbol)
    
    for terminal in sorted(terminals):
        print(f"  {terminal}")
    print("=" * 35)


if __name__ == "__main__":
    # Test the grammar
    print_grammar_info()
    
    # Test expansion with random genes
    import random
    test_genes = [random.randint(0, 10) for _ in range(20)]
    print(f"\nTest with genes: {test_genes[:10]}...")
    
    result = expand_grammar(test_genes, GRAMMAR)
    print(f"Generated architecture: {result}")