# Grammar System

This document provides an in-depth explanation of the grammar-based architecture generation system in NoTransformers.

## Table of Contents

- [Overview](#overview)
- [Context-Free Grammars](#context-free-grammars)
- [Grammar Definition](#grammar-definition)
- [Gene-to-Architecture Mapping](#gene-to-architecture-mapping)
- [Example Expansions](#example-expansions)
- [Customizing the Grammar](#customizing-the-grammar)
- [Best Practices](#best-practices)

## Overview

The grammar system is the foundation of NoTransformers' architecture search. Instead of randomly generating neural network architectures that might be invalid, we use **context-free grammars** to guarantee that every generated architecture is:

- **Syntactically valid**: All layer connections are legal
- **Semantically meaningful**: Operations are composed in sensible ways
- **Modular**: Built from reusable building blocks
- **Hierarchical**: Complex structures emerge from simple rules

## Context-Free Grammars

### What is a Context-Free Grammar?

A context-free grammar (CFG) is a set of production rules that describe how to transform symbols into sequences of other symbols. In NoTransformers, we use CFGs to transform gene sequences into neural network architectures.

### Grammar Components

**1. Non-Terminals** (symbols in `<angle brackets>`)
- Abstract symbols that need to be expanded
- Represent architectural concepts
- Example: `<network>`, `<block>`, `<activation>`

**2. Terminals** (Capitalized tokens)
- Concrete operations/layers
- Cannot be expanded further
- Example: `Conv1D`, `ReLU`, `LayerNorm`

**3. Production Rules**
- Define how to expand non-terminals
- Multiple rules per non-terminal enable choice
- Example: `<activation>` → [`ReLU`] or [`Tanh`] or [`GELU`]

**4. Start Symbol**
- Initial non-terminal (`<start>`)
- Root of the expansion tree

### Why Use Grammars?

**Traditional Random Generation Problems:**
```python
# Random architecture generation (problematic)
layers = []
for _ in range(random.randint(1, 10)):
    layer_type = random.choice(['Conv1D', 'GRU', 'Linear', 'residual'])
    layers.append(layer_type)
# Problem: May create invalid sequences like:
# ['residual', 'residual', 'ReLU']  # No layer to skip!
```

**Grammar-Based Generation (robust):**
```python
# Grammar ensures valid architectures
genes = [3, 0, 1, 2, ...]
architecture = expand_grammar(genes, GRAMMAR)
# Guaranteed: Valid, well-formed architecture
# Example: ['Conv1D', 'ReLU', 'LayerNorm', 'GRU', 'Tanh', 'LayerNorm']
```

## Grammar Definition

The complete grammar is defined in `grammar.py`:

```python
GRAMMAR = {
    "<start>": [["<network>"]],
    
    "<network>": [
        ["<block>"],                    # Single block
        ["<network>", "<block>"]        # Chain multiple blocks
    ],
    
    "<block>": [
        ["<conv_block>"],               # Convolutional block
        ["<recurrent_block>"],          # Recurrent block
        ["<dense_block>"]               # Dense block
    ],
    
    "<conv_block>": [
        ["Conv1D", "<activation>", "LayerNorm", "<maybe_residual>"]
    ],
    
    "<recurrent_block>": [
        ["GRU", "<activation>", "LayerNorm", "<maybe_residual>"]
    ],
    
    "<dense_block>": [
        ["Linear", "<activation>", "LayerNorm"]
    ],
    
    "<maybe_residual>": [
        ["residual"],                   # Add residual connection
        ["identity"]                    # No residual
    ],
    
    "<activation>": [
        ["ReLU"],                       # ReLU activation
        ["Tanh"],                       # Tanh activation
        ["GELU"]                        # GELU activation
    ]
}
```

### Grammar Hierarchy

```
<start>
  └─ <network>
       ├─ <block>
       │    ├─ <conv_block>
       │    │    └─ Conv1D, <activation>, LayerNorm, <maybe_residual>
       │    ├─ <recurrent_block>
       │    │    └─ GRU, <activation>, LayerNorm, <maybe_residual>
       │    └─ <dense_block>
       │         └─ Linear, <activation>, LayerNorm
       └─ <network>, <block> (recursive for multiple blocks)
```

### Non-Terminal Descriptions

| Non-Terminal | Purpose | Options |
|--------------|---------|---------|
| `<start>` | Entry point | Always expands to `<network>` |
| `<network>` | Network structure | Single block or chain of blocks |
| `<block>` | Building block type | Conv, recurrent, or dense |
| `<conv_block>` | Convolutional module | Conv1D + activation + norm + optional residual |
| `<recurrent_block>` | Recurrent module | GRU + activation + norm + optional residual |
| `<dense_block>` | Dense module | Linear + activation + norm |
| `<maybe_residual>` | Residual decision | Add skip connection or not |
| `<activation>` | Activation function | ReLU, Tanh, or GELU |

### Terminal Descriptions

| Terminal | Type | Purpose |
|----------|------|---------|
| `Conv1D` | Layer | 1D convolution for sequences |
| `GRU` | Layer | Gated Recurrent Unit |
| `Linear` | Layer | Fully connected layer |
| `ReLU` | Activation | Rectified Linear Unit |
| `Tanh` | Activation | Hyperbolic tangent |
| `GELU` | Activation | Gaussian Error Linear Unit |
| `LayerNorm` | Normalization | Layer normalization |
| `residual` | Structure | Skip connection |
| `identity` | Structure | No-op (pass through) |

## Gene-to-Architecture Mapping

### Expansion Algorithm

The `expand_grammar()` function converts a gene sequence into a terminal sequence:

```python
def expand_grammar(genes, grammar, max_expansions=50):
    sequence = ["<start>"]  # Start with start symbol
    gene_index = 0
    
    while gene_index < max_expansions:
        # Find first non-terminal
        for i, symbol in enumerate(sequence):
            if not is_terminal(symbol):
                # Get next gene
                gene = genes[gene_index] if gene_index < len(genes) else 0
                gene_index += 1
                
                # Choose production rule
                rules = grammar[symbol]
                rule_index = gene % len(rules)
                chosen_rule = rules[rule_index]
                
                # Replace non-terminal with chosen rule
                sequence = sequence[:i] + chosen_rule + sequence[i+1:]
                break
        
        # Check if all terminals
        if all(is_terminal(s) for s in sequence):
            break
    
    return [s for s in sequence if is_terminal(s)]
```

### Step-by-Step Example

**Input Genes:** `[3, 0, 1, 2, 0]`

**Expansion Process:**

1. **Initial state:**
   ```
   Sequence: ["<start>"]
   Gene index: 0
   ```

2. **Expand `<start>` with gene 3:**
   ```
   Gene: 3
   Rules for <start>: [["<network>"]]
   Choice: 3 % 1 = 0 → Rule 0 → ["<network>"]
   Sequence: ["<network>"]
   ```

3. **Expand `<network>` with gene 0:**
   ```
   Gene: 0
   Rules for <network>: [["<block>"], ["<network>", "<block>"]]
   Choice: 0 % 2 = 0 → Rule 0 → ["<block>"]
   Sequence: ["<block>"]
   ```

4. **Expand `<block>` with gene 1:**
   ```
   Gene: 1
   Rules for <block>: [["<conv_block>"], ["<recurrent_block>"], ["<dense_block>"]]
   Choice: 1 % 3 = 1 → Rule 1 → ["<recurrent_block>"]
   Sequence: ["<recurrent_block>"]
   ```

5. **Expand `<recurrent_block>` (no gene needed, deterministic):**
   ```
   Rules for <recurrent_block>: [["GRU", "<activation>", "LayerNorm", "<maybe_residual>"]]
   Choice: Rule 0 (only option)
   Sequence: ["GRU", "<activation>", "LayerNorm", "<maybe_residual>"]
   ```

6. **Expand `<activation>` with gene 2:**
   ```
   Gene: 2
   Rules for <activation>: [["ReLU"], ["Tanh"], ["GELU"]]
   Choice: 2 % 3 = 2 → Rule 2 → ["GELU"]
   Sequence: ["GRU", "GELU", "LayerNorm", "<maybe_residual>"]
   ```

7. **Expand `<maybe_residual>` with gene 0:**
   ```
   Gene: 0
   Rules for <maybe_residual>: [["residual"], ["identity"]]
   Choice: 0 % 2 = 0 → Rule 0 → ["residual"]
   Sequence: ["GRU", "GELU", "LayerNorm", "residual"]
   ```

8. **Final Architecture (all terminals):**
   ```
   ["GRU", "GELU", "LayerNorm", "residual"]
   ```

### Gene Usage

- **Each non-terminal** consumes one gene from the sequence
- **Gene modulo** determines which rule to apply
- **If genes run out**, default value (0) is used
- **Longer gene sequences** enable deeper architectures

## Example Expansions

### Example 1: Simple Architecture

**Genes:** `[0, 0, 2, 0, 1]`

**Expansion:**
```
[0] <start> → <network>
[0] <network> → <block>
[2] <block> → <dense_block>
    <dense_block> → Linear, <activation>, LayerNorm
[0] <activation> → ReLU

Final: ["Linear", "ReLU", "LayerNorm"]
```

### Example 2: Deep Network

**Genes:** `[0, 1, 0, 1, 0, 1, 1, 2, 1, 0]`

**Expansion:**
```
[0] <start> → <network>
[1] <network> → <network>, <block>
[0] <network> → <block>
[1] <block> → <recurrent_block>
    <recurrent_block> → GRU, <activation>, LayerNorm, <maybe_residual>
[0] <activation> → ReLU
[1] <maybe_residual> → identity
[1] <block> → <recurrent_block>
    <recurrent_block> → GRU, <activation>, LayerNorm, <maybe_residual>
[2] <activation> → GELU
[1] <maybe_residual> → identity

Final: ["GRU", "ReLU", "LayerNorm", "identity", "GRU", "GELU", "LayerNorm", "identity"]
```

### Example 3: Architecture with Residual

**Genes:** `[0, 0, 0, 2, 0]`

**Expansion:**
```
[0] <start> → <network>
[0] <network> → <block>
[0] <block> → <conv_block>
    <conv_block> → Conv1D, <activation>, LayerNorm, <maybe_residual>
[2] <activation> → GELU
[0] <maybe_residual> → residual

Final: ["Conv1D", "GELU", "LayerNorm", "residual"]
```

## Customizing the Grammar

### Adding New Activations

```python
GRAMMAR["<activation>"] = [
    ["ReLU"],
    ["Tanh"],
    ["GELU"],
    ["Sigmoid"],      # New activation
    ["LeakyReLU"]     # New activation
]
```

### Adding Attention Blocks

```python
# Add new block type
GRAMMAR["<block>"].append(["<attention_block>"])

# Define attention block
GRAMMAR["<attention_block>"] = [
    ["MultiHeadAttention", "<activation>", "LayerNorm", "<maybe_residual>"]
]
```

### Adding Pooling Operations

```python
# Add pooling choice
GRAMMAR["<pooling>"] = [
    ["MaxPool"],
    ["AvgPool"],
    ["identity"]
]

# Update conv block to include pooling
GRAMMAR["<conv_block>"] = [
    ["Conv1D", "<activation>", "<pooling>", "LayerNorm", "<maybe_residual>"]
]
```

### Creating Branching Architectures

```python
# Add parallel branches
GRAMMAR["<parallel_block>"] = [
    ["<block>", "<block>"]  # Two parallel paths
]

# Update network to support parallel
GRAMMAR["<network>"].append(["<parallel_block>"])
```

## Best Practices

### Grammar Design

1. **Keep rules simple**: Each production should be easy to understand
2. **Balance options**: Too few limits exploration, too many slows evolution
3. **Ensure validity**: Test that all expansion paths create valid architectures
4. **Use hierarchy**: Build complex structures from simple components

### Gene Length

- **Short genes (5-10)**: Simple, shallow architectures
- **Medium genes (10-20)**: Balanced depth and complexity
- **Long genes (20+)**: Deep, complex architectures

### Testing Grammars

```python
# Test grammar expansion
from grammar import GRAMMAR, expand_grammar, print_grammar_info

# Print grammar structure
print_grammar_info()

# Test with random genes
import random
test_genes = [random.randint(0, 10) for _ in range(20)]
architecture = expand_grammar(test_genes, GRAMMAR)
print(f"Generated: {architecture}")
```

### Common Pitfalls

**1. Infinite recursion:**
```python
# Bad: Can expand forever
GRAMMAR["<network>"] = [["<network>", "<block>"]]  # Only recursive!

# Good: Include base case
GRAMMAR["<network>"] = [
    ["<block>"],              # Base case
    ["<network>", "<block>"]  # Recursive case
]
```

**2. Invalid sequences:**
```python
# Bad: Residual without a layer
GRAMMAR["<block>"] = [["residual"]]

# Good: Layer before residual
GRAMMAR["<block>"] = [["Linear", "<activation>", "residual"]]
```

**3. Dimension mismatches:**
```python
# Handled by dynamic initialization in primitives
# Conv1DWrapper and GRUWrapper adapt to input shapes
```

## Advanced Topics

### Probabilistic Rules

You can bias evolution toward certain rules by duplicating them:

```python
GRAMMAR["<activation>"] = [
    ["ReLU"],
    ["ReLU"],     # ReLU appears twice
    ["Tanh"],     # More likely to be chosen
    ["GELU"]
]
```

### Conditional Expansions

Grammar can encode architectural patterns:

```python
# ResNet-style blocks
GRAMMAR["<resnet_block>"] = [
    ["Conv1D", "ReLU", "Conv1D", "residual"]
]

# DenseNet-style blocks
GRAMMAR["<densenet_block>"] = [
    ["<block>", "<concatenate>", "<block>"]
]
```

### Grammar Evolution

The grammar itself can be evolved:

```python
# Meta-evolution: Evolve production rules
def evolve_grammar(base_grammar, num_generations):
    # Add/remove rules
    # Modify expansions
    # Test fitness of resulting architectures
    pass
```

## Summary

The grammar system provides:
- **Guaranteed validity** of generated architectures
- **Structured exploration** of architecture space
- **Modularity** and reusability of components
- **Interpretability** of the search process
- **Flexibility** through customizable rules

For practical usage, see [Getting Started](getting-started.md).
For implementation details, see [Architecture Guide](architecture.md).
