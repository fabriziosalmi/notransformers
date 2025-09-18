"""
Grammar-based Neural Architecture Evolution

Questa grammatica definisce le regole per costruire architetture neurali modulari
attraverso un approccio di neuroevoluzione grammaticale.

Le chiavi sono "non-terminali" (simboli astratti come <network>)
I valori sono liste di possibili "espansioni" (regole di produzione)
I token in maiuscolo (Conv1D, GRU) sono "terminali" concreti
I token tra <> sono "non-terminali" da espandere
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
        ["Linear", "<activation>", "LayerNorm"]  # I blocchi densi di solito non hanno residual
    ],
    
    "<maybe_residual>": [
        ["residual"],  # Aggiungi una connessione residua
        ["identity"]   # Non fare nulla
    ],
    
    "<activation>": [
        ["ReLU"],
        ["Tanh"],
        ["GELU"]
    ]
}


def is_terminal(symbol):
    """Verifica se un simbolo è terminale (non contiene < >)"""
    return not (symbol.startswith('<') and symbol.endswith('>'))


def expand_grammar(genes, grammar, max_expansions=50):
    """
    Espande una grammatica usando una sequenza di geni
    
    Args:
        genes: Lista di interi che guidano le scelte grammaticali
        grammar: Dizionario della grammatica con regole di produzione
        max_expansions: Numero massimo di espansioni per evitare loop infiniti
    
    Returns:
        Lista di terminali che rappresenta l'architettura finale
    """
    # Inizia con il simbolo di partenza
    sequence = ["<start>"]
    gene_index = 0
    expansions = 0
    
    while expansions < max_expansions:
        # Trova il primo non-terminale
        non_terminal_found = False
        for i, symbol in enumerate(sequence):
            if not is_terminal(symbol):
                non_terminal_found = True
                
                # Prendi il prossimo gene (con wrapping se necessario)
                if gene_index < len(genes):
                    gene = genes[gene_index]
                    gene_index += 1
                else:
                    # Se finiamo i geni, usa un valore di default
                    gene = 0
                
                # Scegli una regola di produzione
                if symbol in grammar:
                    rules = grammar[symbol]
                    rule_index = gene % len(rules)
                    chosen_rule = rules[rule_index]
                    
                    # Sostituisci il non-terminale con la regola scelta
                    sequence = sequence[:i] + chosen_rule + sequence[i+1:]
                else:
                    # Se il simbolo non è nella grammatica, rimuovilo
                    sequence = sequence[:i] + sequence[i+1:]
                
                expansions += 1
                break
        
        if not non_terminal_found:
            # Tutti i simboli sono terminali
            break
    
    # Filtra solo i terminali finali
    final_sequence = [symbol for symbol in sequence if is_terminal(symbol)]
    return final_sequence


def print_grammar_info():
    """Stampa informazioni sulla grammatica definita"""
    print("=== GRAMMATICA NEUROEVOLUZIONE ===")
    print("Non-terminali disponibili:")
    for key in GRAMMAR.keys():
        print(f"  {key}: {len(GRAMMAR[key])} regole")
    
    print("\nTerminali utilizzati:")
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
    # Test della grammatica
    print_grammar_info()
    
    # Test espansione con geni casuali
    import random
    test_genes = [random.randint(0, 10) for _ in range(20)]
    print(f"\nTest con geni: {test_genes[:10]}...")
    
    result = expand_grammar(test_genes, GRAMMAR)
    print(f"Architettura generata: {result}")