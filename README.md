# NoTransformers: Grammar-Guided Neuroevolution Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


---


### 🎯 Project Vision

**Can we automatically discover powerful neural architectures without relying on Transformers?**

NoTransformers answers this question through **grammar-guided neuroevolution**: instead of manually designing models or fine-tuning pre-trained Transformers, we evolve entire architectures from scratch using evolutionary algorithms constrained by formal grammars.

### 🔬 Technical Approach

Each **genome** is a sequence of integers decoded via production rules into valid neural architectures. The system co-optimizes:
- **Architecture topology** (Conv1D, GRU, residual connections, normalization)
- **Learning hyperparameters** (optimizer, learning rate, scheduler, activation functions)
- **Adversarial robustness** (optional competitive co-evolution)

This approach combines:
- **Genetic Algorithms** for population-based search
- **Context-Free Grammars** for architectural constraints
- **Multi-Objective Optimization** (fitness, complexity, novelty, diversity)
- **Competitive Co-Evolution** (solvers vs adversarial pattern generators)

---

### ✨ Key Features

#### 🧬 **Grammar-Based Architecture Search**
- Production rules guarantee **valid architectures only**
- Supports hierarchical building blocks (conv, recurrent, dense)
- Automatic residual connection management
- Dynamic shape inference and adapter insertion

#### 🛠️ **Rich Primitive Library**
```python
# Computational Primitives
- Conv1D / Conv1DWrapper  # Sequence convolution with automatic shape handling
- GRU / GRUWrapper        # Recurrent processing with sequence-to-vector output
- Linear / Dense blocks   # Fully connected layers with dynamic initialization
- LayerNorm / BatchNorm   # Normalization primitives
- ReLU / GELU / Tanh      # Activation functions

# Structural Primitives
- residual               # Skip connections (with automatic adapters)
- identity              # Passthrough for optional connections
- SequenceToVector      # Aggregate sequences to fixed-size vectors
```

#### 🥊 **Competitive Co-Evolution**
Two populations evolve simultaneously:
1. **Solvers** (ModelGenome): Neural architectures that solve parity tasks
2. **Saboteurs** (SaboteurGenome): Adversarial pattern generators

**Saboteur Strategies:**
- `alternating`: 1,0,1,0,... patterns
- `repeating_chunk`: [1,1,0], [1,1,0], ...
- `mostly_zeros`: Sparse positive examples
- `mostly_ones`: Sparse negative examples
- `edge_ones`: Active boundaries with quiet centers

Solvers are evaluated against saboteur-generated sequences, creating an **evolutionary arms race** that promotes robust architectures.

#### 📊 **Comprehensive Metrics & Analytics**

**Standard Evolution Metrics:**
```json
{
  "generation": 10,
  "fitness_mean": 0.823,
  "fitness_median": 0.841,
  "fitness_std": 0.092,
  "arch_entropy": 1.847,
  "unique_architectures": 12,
  "param_mean": 2341.5,
  "param_std": 892.3,
  "depth_mean": 7.2,
  "novelty_mean": 0.412
}
```

**Co-Evolution Metrics:**
```json
{
  "solver_avg": 0.612,
  "solver_best": 0.891,
  "saboteur_avg": 0.388,
  "saboteur_best": 0.742,
  "solver_arch_entropy": 1.523,
  "pattern_type_counts": {"alternating": 8, "edge_ones": 5},
  "noise_level_mean": 0.156,
  "pattern_length_mean": 5.2
}
```

#### 🎛️ **Advanced Evolutionary Mechanisms**

1. **Novelty Search**
   - Levenshtein distance-based architecture diversity
   - Blended fitness: `(1-w)*performance + w*novelty`
   - Prevents premature convergence

2. **Diversity Pressure**
   - Penalizes duplicate architectures proportionally
   - Formula: `penalty = 1 - pressure * (copies-1)/copies`
   - Encourages exploration of unique solutions

3. **Early Stopping**
   - Configurable fitness threshold
   - Patience mechanism (consecutive generations)
   - Prevents overfitting to evolutionary noise

4. **Top-K Reporting**
   - Tracks best unique architectures per generation
   - Enables multi-solution analysis
   - Supports Pareto front approximation

5. **Parameter Count Caching**
   - Avoids redundant model instantiation
   - Speeds up complexity penalty computation
   - Efficient memory usage

---

### 🚀 Installation & Quick Start

#### Prerequisites
```bash
# Python 3.8+
python --version

# Core dependencies
pip install torch numpy matplotlib

# Optional: Graphviz for architecture visualization
# macOS
brew install graphviz
pip install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
pip install graphviz
```

#### Basic Usage

**1. Standard Evolution (Parity Task)**
```bash
cd evolutionary_discovery
python main.py --population 50 --generations 40
```

**2. Competitive Co-Evolution**
```bash
python main.py --coevolution --population 40 --generations 20 --metrics-json coevo.json
```

**3. High-Diversity Search with Early Stopping**
```bash
python main.py \
  --population 100 \
  --generations 50 \
  --metrics-json diversity_run.json
```

#### Command-Line Interface

```
usage: main.py [-h] [--coevolution] [--generations G] [--population P] [--metrics-json PATH]

Grammar-based Neuroevolution with optional Competitive Co-Evolution

optional arguments:
  -h, --help           Show this help message
  --coevolution        Enable competitive Solvers vs Saboteurs mode
  --generations G      Number of evolutionary generations (default: 100)
  --population P       Population size (default: 100)
  --metrics-json PATH  Export per-generation metrics to JSON file
```

---

### � Project Architecture

```
notransformers/
├── evolutionary_discovery/
│   ├── main.py                # CLI entry point
│   ├── evolution.py           # Core evolutionary algorithms
│   │   ├── EvolutionarySearch      # Main evolution loop
│   │   ├── train_and_evaluate_genome
│   │   ├── plot_evolution          # Visualization utilities
│   │   └── _MetricsExportMixin     # JSON export functionality
│   │
│   ├── genome.py              # Neural architecture genome
│   │   ├── ModelGenome             # Grammar-based genome representation
│   │   ├── Conv1DWrapper           # Shape-adaptive Conv1D
│   │   ├── GRUWrapper              # GRU with sequence aggregation
│   │   └── SequenceToVector        # Mean pooling over time
│   │
│   ├── grammar.py             # Context-free grammar definitions
│   │   ├── GRAMMAR                 # Production rules dictionary
│   │   ├── expand_grammar          # Gene-to-architecture decoder
│   │   └── print_grammar_info      # Grammar introspection
│   │
│   ├── saboteur.py            # Adversarial pattern generation
│   │   └── SaboteurGenome          # Adversarial sequence genome
│   │
│   ├── primitives.py          # Low-level computational primitives
│   │   ├── ComputationalPrimitive  # Base class
│   │   ├── Conv1D, Linear, GRU     # Layer implementations
│   │   └── Activation functions
│   │
│   ├── evaluation_suite.py    # Extended benchmarking suite
│   │   ├── train_and_test_model
│   │   ├── Parity benchmarks
│   │   ├── Copy task
│   │   └── Synthetic regression
│   │
│   ├── visualize.py           # Graphviz architecture rendering
│   └── test_evolution_exact.py # Unit tests
│
├── README.md                  # This file
└── metrics_coevo.json         # Example output
```
A controlled search space reduces invalid architectures and encodes inductive bias toward modular, residual-friendly sequences.

### 🤝 Contributing
Ideas, issues, and PRs welcome. Open a discussion for grammar design extensions or new primitive suggestions.
