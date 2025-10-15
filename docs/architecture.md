# Architecture Guide

This document describes the overall architecture and design of the NoTransformers framework.

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Module Descriptions](#module-descriptions)
- [Design Principles](#design-principles)

## System Overview

NoTransformers is a **grammar-guided neuroevolution framework** that automatically discovers neural network architectures without relying on transformers. The system combines:

1. **Genetic Algorithms** for population-based search
2. **Context-Free Grammars** for architectural constraints
3. **Multi-Objective Optimization** for balanced solutions
4. **Competitive Co-Evolution** for adversarial robustness

```
┌─────────────────────────────────────────────────┐
│         NoTransformers Framework                │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   Grammar    │──────│    Genome    │        │
│  │   System     │      │  Encoding    │        │
│  └──────────────┘      └──────────────┘        │
│         │                      │                │
│         ▼                      ▼                │
│  ┌──────────────────────────────────┐          │
│  │    Evolutionary Search Engine    │          │
│  │  - Selection                     │          │
│  │  - Crossover                     │          │
│  │  - Mutation                      │          │
│  │  - Fitness Evaluation            │          │
│  └──────────────────────────────────┘          │
│         │                      │                │
│         ▼                      ▼                │
│  ┌──────────────┐      ┌──────────────┐        │
│  │  PyTorch     │      │   Metrics    │        │
│  │  Models      │      │   & Export   │        │
│  └──────────────┘      └──────────────┘        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Core Components

### 1. Grammar System (`grammar.py`)

**Purpose:** Define valid neural architecture building blocks and rules for combining them.

**Key Elements:**
- **GRAMMAR**: Dictionary of production rules
- **expand_grammar()**: Converts gene sequences to architectures
- **is_terminal()**: Checks if a symbol is a terminal (concrete operation)

**Grammar Structure:**
```python
GRAMMAR = {
    "<start>": [["<network>"]],
    "<network>": [
        ["<block>"],
        ["<network>", "<block>"]
    ],
    "<block>": [
        ["<conv_block>"],
        ["<recurrent_block>"],
        ["<dense_block>"]
    ],
    ...
}
```

### 2. Genome Representation (`genome.py`)

**Purpose:** Encode neural architectures as evolvable gene sequences.

**Main Classes:**

#### ModelGenome
Represents a neural architecture as:
- **genes**: List of integers guiding grammar expansion
- **learning_params**: Hyperparameters (optimizer, learning rate, scheduler)
- **built_architecture**: Cached terminal sequence
- **model**: Cached PyTorch model

**Key Methods:**
```python
def build_from_grammar(self, grammar=None, max_expansions=50):
    """Expands genes into terminal sequence"""

def build_pytorch_model(self, input_dim, output_dim):
    """Constructs PyTorch model from architecture"""

def mutate(self, mutation_rate):
    """Applies mutations to genes and learning params"""

def crossover(self, other):
    """Combines genes from two parents"""
```

#### Wrapper Classes
- **Conv1DWrapper**: Handles Conv1D with proper transposition
- **GRUWrapper**: GRU with sequence-to-vector aggregation
- **SequenceToVector**: Mean pooling over time dimension

### 3. Evolutionary Search (`evolution.py`)

**Purpose:** Implement the evolutionary algorithm that discovers architectures.

**Main Class: EvolutionarySearch**

**Core Methods:**

```python
def run(self, generations, sequence_length, num_samples):
    """Standard evolution loop"""

def run_coevolution(self, generations):
    """Competitive co-evolution with saboteurs"""

def _evaluate_genome_fitness(self, genome, X_data, y_data):
    """Evaluate a single genome"""

def _tournament_selection(self, fitnesses):
    """Select parents via tournament selection"""

def _apply_diversity_pressure(self, population, fitnesses):
    """Penalize duplicate architectures"""

def _compute_novelty_scores(self, population):
    """Calculate novelty using architecture distance"""
```

**Evolutionary Mechanisms:**

1. **Selection**: Tournament selection picks parents
2. **Crossover**: Single-point crossover creates offspring
3. **Mutation**: Random gene changes and learning param adjustments
4. **Fitness Evaluation**: Train models on tasks and measure performance
5. **Elitism**: Best individuals always survive
6. **Diversity Pressure**: Penalize duplicate architectures
7. **Novelty Search**: Reward unique architectural patterns

### 4. Primitives Library (`primitives.py`)

**Purpose:** Low-level computational building blocks for neural networks.

**Base Class:**
```python
class ComputationalPrimitive(nn.Module):
    """Base class for all computational primitives"""
```

**Available Primitives:**

| Primitive | Purpose | Features |
|-----------|---------|----------|
| InputNode | Entry point | Passes input through |
| Linear | Fully connected | Dynamic dimension inference |
| Conv1D | Sequence convolution | Automatic shape handling |
| GRU | Recurrent processing | Sequence-to-vector output |
| ReLU, Tanh, GELU | Activations | Standard activation functions |
| LayerNorm, BatchNorm | Normalization | Stabilize training |

**Dynamic Initialization:**
Many primitives use lazy initialization to handle dimension mismatches:
```python
def _init_linear(self, actual_input_dim):
    """Initialize linear layer with actual input dimensions"""
    if self.linear is None or self.linear.in_features != actual_input_dim:
        self.linear = nn.Linear(actual_input_dim, self.output_dim)
```

### 5. Saboteur System (`saboteur.py`)

**Purpose:** Generate adversarial patterns for competitive co-evolution.

**Class: SaboteurGenome**

**Pattern Types:**
- `alternating`: 1,0,1,0,... patterns
- `repeating_chunk`: [1,1,0], [1,1,0], ...
- `mostly_zeros`: Sparse positive examples
- `mostly_ones`: Sparse negative examples  
- `edge_ones`: Active boundaries with quiet centers

**Key Methods:**
```python
def generate_sequence(self, seq_len):
    """Generate adversarial sequence"""

def generate_batch(self, batch_size, seq_len):
    """Generate batch of sequences with labels"""
```

### 6. Evaluation Suite (`evaluation_suite.py`)

**Purpose:** Extended benchmarking and evaluation capabilities.

**Functions:**
- `train_and_test_model()`: Train and evaluate models
- Parity benchmarks: Test on parity computation tasks
- Copy tasks: Sequence memorization tests
- Synthetic regression: Continuous value prediction

### 7. Visualization (`visualize.py`)

**Purpose:** Visualize evolved architectures using Graphviz.

**Function:**
```python
def visualize_genome(genome, filename="best_genome.png"):
    """Creates directed graph of architecture"""
```

**Output:**
- Nodes represent layers/operations
- Edges show data flow
- Colors indicate layer types
- Dashed red lines show residual connections

## Data Flow

### Standard Evolution Flow

```
1. Initialize Population
   ↓
   [Random genomes generated]
   
2. For each generation:
   ↓
   a. Evaluate Fitness
      ├─ Build PyTorch models from genes
      ├─ Train on task data
      └─ Measure performance
   ↓
   b. Selection
      └─ Tournament selection chooses parents
   ↓
   c. Generate Offspring
      ├─ Crossover between parents
      └─ Mutation of genes
   ↓
   d. Update Population
      ├─ Keep elite individuals
      ├─ Add offspring
      └─ Apply diversity pressure
   ↓
   e. Track Metrics
      └─ Record fitness, diversity, novelty
   ↓
3. Return Best Genome
```

### Co-Evolution Flow

```
Initialize Solver & Saboteur Populations
   ↓
For each generation:
   ↓
   ┌──────────────────┬──────────────────┐
   │   Solvers        │    Saboteurs     │
   ├──────────────────┼──────────────────┤
   │ Train on tasks   │ Generate patterns│
   │ Fight saboteurs  │ Challenge solvers│
   └──────────────────┴──────────────────┘
   ↓
   Calculate Fitness:
   - Solver fitness = Performance on adversarial data
   - Saboteur fitness = (1 - Solver performance)
   ↓
   Evolve Both Populations Independently
   ↓
Return Best Solver & Saboteur
```

## Module Descriptions

### main.py
**Entry Point**: Command-line interface and configuration

**Responsibilities:**
- Parse arguments
- Set up evolutionary search
- Run standard or co-evolution mode
- Export metrics and results

**Configuration Parameters:**
```python
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
COMPLEXITY_PENALTY = 1e-6
DIVERSITY_PRESSURE = 0.2
NOVELTY_WEIGHT = 0.15
```

### genome_old.py
Legacy genome implementation (kept for compatibility)

### test_evolution_exact.py
Unit tests for evolutionary search functionality

## Design Principles

### 1. Grammar-Guided Search
**Rationale:** Ensures all generated architectures are valid and well-formed.

**Benefits:**
- No invalid architectures
- Encodes domain knowledge
- Reduces search space
- Enables hierarchical composition

### 2. Lazy Initialization
**Rationale:** Handle dimension mismatches gracefully without architecture changes.

**Benefits:**
- Flexible to input shapes
- Adapts to data automatically
- Reduces errors during evolution

### 3. Caching Strategy
**Rationale:** Avoid redundant computations during evolution.

**Cached Elements:**
- Built architectures (from genes)
- PyTorch models
- Parameter counts
- Fitness values

### 4. Modular Primitives
**Rationale:** Build complex architectures from simple, reusable components.

**Benefits:**
- Easy to extend with new operations
- Consistent interface
- Testable components
- Clear abstractions

### 5. Multi-Objective Optimization
**Rationale:** Balance multiple competing goals.

**Objectives:**
- **Performance**: High task accuracy
- **Complexity**: Low parameter count
- **Diversity**: Unique architectural patterns
- **Novelty**: Different from existing solutions

### 6. Competitive Co-Evolution
**Rationale:** Create robust architectures through adversarial training.

**Benefits:**
- Better generalization
- Robust to distribution shifts
- Explores harder problem variants
- Evolutionary arms race dynamics

## Extension Points

Want to extend NoTransformers? Here are the key extension points:

### Adding New Primitives
1. Create new class in `primitives.py`
2. Inherit from `ComputationalPrimitive`
3. Implement `forward()` method
4. Add to grammar rules

### Modifying Grammar
1. Edit `GRAMMAR` in `grammar.py`
2. Add new non-terminals or production rules
3. Test with `expand_grammar()`

### Custom Fitness Functions
1. Implement evaluation in `evolution.py`
2. Override `_evaluate_genome_fitness()`
3. Return fitness score (0-1 range)

### New Evolutionary Mechanisms
1. Add methods to `EvolutionarySearch` class
2. Integrate into `run()` method
3. Update metrics tracking

## Performance Considerations

### Memory Usage
- **Populations**: Each genome stores genes, model cache, and parameters
- **Training**: Models created per evaluation
- **Optimization**: Use parameter caching and lazy initialization

### Computation Time
- **Dominant Factor**: Model training during fitness evaluation
- **Optimizations**: 
  - Smaller evaluation datasets
  - Fewer training epochs
  - Batch evaluations
  - GPU acceleration

### Scalability
- **Population Size**: Linear scaling with number of genomes
- **Generations**: Linear scaling with evolution steps
- **Architecture Complexity**: Polynomial with gene length

## Summary

The NoTransformers architecture is designed for:
- **Flexibility**: Easy to extend and modify
- **Efficiency**: Cached computations and lazy initialization
- **Robustness**: Grammar constraints and diverse evaluation
- **Interpretability**: Clear mapping from genes to architectures

For practical usage, see [Getting Started](getting-started.md).
For detailed API information, see [API Reference](api-reference.md).
