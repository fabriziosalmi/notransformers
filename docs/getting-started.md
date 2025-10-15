# Getting Started with NoTransformers

This guide will help you get started with the NoTransformers grammar-guided neuroevolution framework.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage Examples](#basic-usage-examples)
- [Understanding the Output](#understanding-the-output)
- [Next Steps](#next-steps)

## Prerequisites

Before installing NoTransformers, ensure you have the following:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB recommended for larger populations)
- **CPU**: Multi-core processor recommended for faster evolution
- **GPU**: Optional, CUDA-compatible GPU for accelerated training

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/fabriziosalmi/notransformers.git
cd notransformers
```

### Step 2: Set Up a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install torch numpy matplotlib
```

### Step 4: (Optional) Install Visualization Tools

For architecture visualization using Graphviz:

**On macOS:**
```bash
brew install graphviz
pip install graphviz
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
pip install graphviz
```

**On Windows:**
Download and install Graphviz from [graphviz.org](https://graphviz.org/download/), then:
```bash
pip install graphviz
```

## Quick Start

### Running Your First Evolution

Navigate to the evolutionary discovery module and run a basic evolution:

```bash
cd evolutionary_discovery
python main.py --population 50 --generations 40
```

This will:
- Create a population of 50 random neural architectures
- Evolve them over 40 generations
- Display progress and fitness metrics
- Show the best discovered architecture

**Expected Output:**
```
🧬 GRAMMAR-BASED NEUROEVOLUTION 🧬
==================================================
Mode: STANDARD
Population: 50, Generations: 40
...
Generation 1: mean=0.523, median=0.512, best=0.642
Generation 2: mean=0.587, median=0.578, best=0.701
...
```

### Running Co-Evolution

To run competitive co-evolution with solvers and saboteurs:

```bash
python main.py --coevolution --population 40 --generations 20
```

This creates an evolutionary arms race where:
- **Solvers** try to solve parity tasks
- **Saboteurs** generate adversarial patterns to challenge solvers

## Basic Usage Examples

### Example 1: Standard Evolution with Custom Settings

```bash
python main.py \
  --population 100 \
  --generations 50 \
  --metrics-json results.json
```

This runs evolution with:
- Population size of 100
- 50 generations
- Exports metrics to `results.json` for analysis

### Example 2: Quick Test Run

```bash
python main.py --population 20 --generations 10
```

Suitable for:
- Testing the installation
- Quick experiments
- Understanding the system behavior

### Example 3: High-Diversity Search

```bash
python main.py \
  --population 150 \
  --generations 100 \
  --metrics-json diversity_search.json
```

Best for:
- Exploring diverse architectural solutions
- Finding multiple good architectures
- Long-term evolution experiments

## Understanding the Output

### Console Output

During evolution, you'll see:

```
Generation 15
  Fitness: mean=0.823 ±0.092, median=0.841, best=0.891
  Architecture Diversity: entropy=1.847, unique=12/50
  Complexity: mean_params=2341.5 ±892.3, mean_depth=7.2
  Novelty: mean=0.412
```

**Key Metrics:**
- **Fitness**: Model performance (0-1 scale, higher is better)
- **Architecture Diversity**: How many unique architectures exist
- **Complexity**: Model size in parameters and depth
- **Novelty**: How different architectures are from each other

### Final Results

At the end of evolution, you'll see:

```
🏆 FINAL RESULTS 🏆
Best genome: ModelGenome(genes=[3, 7, 1, 2, 5, ...])

Discovered architecture:
  Conv1D -> ReLU -> LayerNorm -> GRU -> Tanh -> LayerNorm

Learning parameters:
  optimizer: adam
  learning_rate: 0.005
  lr_scheduler: cosine

Generated PyTorch model:
  SequentialModel(
    (0): Conv1DWrapper(...)
    (1): ReLU()
    (2): LayerNorm(...)
    ...
  )
```

### Metrics JSON Output

If you specified `--metrics-json`, you'll get a detailed JSON file with per-generation statistics:

```json
{
  "generations": [
    {
      "generation": 1,
      "fitness_mean": 0.523,
      "fitness_median": 0.512,
      "fitness_std": 0.089,
      "fitness_best": 0.642,
      "arch_entropy": 1.234,
      "unique_architectures": 15,
      "param_mean": 1823.4,
      "depth_mean": 5.6,
      "novelty_mean": 0.389
    },
    ...
  ]
}
```

## Next Steps

Now that you have NoTransformers running, explore:

1. **[Architecture Guide](architecture.md)** - Learn about system components and design
2. **[Grammar System](grammar.md)** - Understand how grammars define architectures
3. **[Evolutionary Algorithms](evolution.md)** - Deep dive into evolutionary mechanisms
4. **[API Reference](api-reference.md)** - Detailed API documentation
5. **[Examples and Tutorials](examples.md)** - More advanced usage patterns
6. **[Co-Evolution Guide](coevolution.md)** - Competitive evolution strategies

## Troubleshooting

### Common Issues

**Issue: "No module named torch"**
```bash
pip install torch
```

**Issue: "CUDA out of memory"**
- Reduce population size: `--population 20`
- Use CPU: The system automatically falls back to CPU

**Issue: Evolution is slow**
- Start with smaller populations (20-50)
- Reduce generations for testing
- Use fewer evaluation runs (configured in main.py)

**Issue: Low fitness values**
- Increase generations (try 50-100)
- Increase population size
- Check if the task is appropriate for grammar

For more help, see [Troubleshooting](troubleshooting.md) or open an issue on GitHub.
