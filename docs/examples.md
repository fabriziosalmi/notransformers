# Examples and Tutorials

This document provides practical examples and step-by-step tutorials for using NoTransformers.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Tutorials](#tutorials)
- [Best Practices](#best-practices)

## Basic Examples

### Example 1: Quick Start Evolution

The simplest way to start evolving architectures:

```python
from evolution import EvolutionarySearch

# Create evolutionary search
search = EvolutionarySearch(
    input_dim=1,
    output_dim=1,
    population_size=50
)

# Run for 30 generations
best_genome, fitness_hist, best_hist = search.run(
    generations=30,
    sequence_length=8,
    num_samples=500
)

print(f"Best fitness: {max(fitness_hist):.4f}")
print(f"Architecture: {best_genome.get_architecture_string()}")
```

**Expected Output:**
```
Best fitness: 0.8523
Architecture: Conv1D-ReLU-LayerNorm-GRU-Tanh-LayerNorm
```

### Example 2: Visualize Best Architecture

```python
from evolution import EvolutionarySearch
from visualize import visualize_genome

# Run evolution
search = EvolutionarySearch(input_dim=1, output_dim=1)
best_genome, _, _ = search.run(
    generations=20,
    sequence_length=8,
    num_samples=500
)

# Visualize
visualize_genome(best_genome, "best_architecture.png")
print("Visualization saved to best_architecture.png")
```

### Example 3: Custom Hyperparameters

```python
from evolution import EvolutionarySearch

search = EvolutionarySearch(
    input_dim=1,
    output_dim=1,
    population_size=100,
    mutation_rate=0.4,           # Higher exploration
    crossover_rate=0.6,          # Less recombination
    tournament_size=5,           # Stronger selection pressure
    complexity_penalty_coef=5e-6, # Prefer smaller models
    diversity_pressure=0.3,      # More diversity
    novelty_weight=0.2           # More novelty search
)

best, _, _ = search.run(generations=50, sequence_length=8, num_samples=500)
```

### Example 4: Export and Analyze Metrics

```python
from evolution import EvolutionarySearch
import json
import matplotlib.pyplot as plt

search = EvolutionarySearch(input_dim=1, output_dim=1)
best, fitness, best_fitness = search.run(
    generations=40,
    sequence_length=8,
    num_samples=500
)

# Export metrics
metrics = search.export_metrics()
with open('evolution_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Plot fitness over time
plt.figure(figsize=(10, 6))
plt.plot(fitness, label='Mean Fitness', alpha=0.7)
plt.plot(best_fitness, label='Best Fitness', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution Progress')
plt.legend()
plt.grid(True)
plt.savefig('fitness_evolution.png')
print("Plot saved to fitness_evolution.png")
```

### Example 5: Manual Genome Creation and Testing

```python
from genome import ModelGenome
from evolution import train_and_evaluate_genome
import torch

# Create custom genome
genes = [3, 0, 0, 1, 0]  # Specific architecture
learning_params = {
    'optimizer': 'adam',
    'learning_rate': 0.01,
    'lr_scheduler': 'cosine'
}
genome = ModelGenome(genes, learning_params)

# Generate test data
X = torch.randn(500, 8, 1)  # (samples, seq_len, features)
y = (X.sum(dim=1) % 2).float()  # Parity labels

# Evaluate
fitness = train_and_evaluate_genome(
    genome, X, y, learning_params,
    epochs=20, batch_size=32
)

print(f"Custom genome fitness: {fitness:.4f}")
print(f"Architecture: {genome.get_architecture_string()}")
```

## Advanced Examples

### Example 6: Co-Evolution with Analysis

```python
from evolution import EvolutionarySearch
import numpy as np
import matplotlib.pyplot as plt

# Run co-evolution
search = EvolutionarySearch(input_dim=1, output_dim=1, population_size=40)
solvers, saboteurs, (s_avg, s_best, sab_avg, sab_best) = search.run_coevolution(
    generations=25
)

# Analyze results
print(f"Best solver fitness: {max(s_best):.4f}")
print(f"Best saboteur score: {max(sab_best):.4f}")

# Plot co-evolution dynamics
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(s_avg, label='Solver Avg', color='blue', alpha=0.6)
ax.plot(s_best, label='Solver Best', color='blue', linewidth=2)
ax.plot(sab_avg, label='Saboteur Avg', color='red', alpha=0.6)
ax.plot(sab_best, label='Saboteur Best', color='red', linewidth=2)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Co-Evolution Dynamics')
ax.legend()
ax.grid(True)
plt.savefig('coevolution.png')

# Test best solver on hardest saboteur
best_solver_idx = np.argmax(s_best)
best_saboteur_idx = np.argmax(sab_best)
print(f"\nBest solver: {solvers[best_solver_idx].get_architecture_string()}")
print(f"Best saboteur: {saboteurs[best_saboteur_idx].params}")
```

### Example 7: Multi-Objective Evolution

Track and optimize multiple objectives:

```python
from evolution import EvolutionarySearch
import numpy as np

class MultiObjectiveSearch(EvolutionarySearch):
    def _evaluate_genome_fitness(self, genome, X_data, y_data):
        # Evaluate performance
        base_fitness = super()._evaluate_genome_fitness(genome, X_data, y_data)
        
        # Get model info
        model = genome.build_pytorch_model(self.input_dim, self.output_dim)
        param_count = sum(p.numel() for p in model.parameters())
        depth = len(genome.build_from_grammar())
        
        # Multi-objective: accuracy, size, depth
        # Normalize objectives
        accuracy_score = base_fitness
        size_score = 1.0 / (1.0 + param_count / 1000.0)  # Prefer smaller
        depth_score = 1.0 / (1.0 + depth / 10.0)  # Prefer shallower
        
        # Weighted combination
        final_fitness = (
            0.7 * accuracy_score +
            0.2 * size_score +
            0.1 * depth_score
        )
        
        return final_fitness

# Use custom search
search = MultiObjectiveSearch(input_dim=1, output_dim=1)
best, _, _ = search.run(generations=40, sequence_length=8, num_samples=500)

model = best.build_pytorch_model(1, 1)
print(f"Best architecture: {best.get_architecture_string()}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Depth: {len(best.build_from_grammar())}")
```

### Example 8: Custom Grammar

Define and use a custom grammar:

```python
from genome import ModelGenome
from grammar import expand_grammar

# Custom grammar with attention
CUSTOM_GRAMMAR = {
    "<start>": [["<network>"]],
    "<network>": [
        ["<block>"],
        ["<network>", "<block>"]
    ],
    "<block>": [
        ["<conv_block>"],
        ["<attention_block>"],
        ["<dense_block>"]
    ],
    "<conv_block>": [
        ["Conv1D", "<activation>", "LayerNorm"]
    ],
    "<attention_block>": [
        ["MultiHeadAttention", "LayerNorm", "<activation>"]
    ],
    "<dense_block>": [
        ["Linear", "<activation>"]
    ],
    "<activation>": [
        ["ReLU"],
        ["GELU"],
        ["Tanh"]
    ]
}

# Test custom grammar
genes = [5, 1, 1, 0, 2, 1]
architecture = expand_grammar(genes, CUSTOM_GRAMMAR)
print(f"Custom architecture: {architecture}")
```

### Example 9: Parallel Population Evaluation

Speed up evolution with parallel evaluation:

```python
from evolution import EvolutionarySearch, train_and_evaluate_genome
from multiprocessing import Pool
import torch

class ParallelEvolution(EvolutionarySearch):
    def __init__(self, *args, num_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
    
    def _evaluate_population(self, population, X_data, y_data):
        # Prepare evaluation tasks
        tasks = [(genome, X_data, y_data) for genome in population]
        
        # Parallel evaluation
        with Pool(self.num_workers) as pool:
            fitnesses = pool.starmap(self._eval_worker, tasks)
        
        return fitnesses
    
    @staticmethod
    def _eval_worker(genome, X_data, y_data):
        try:
            return train_and_evaluate_genome(
                genome, X_data, y_data,
                genome.learning_params,
                epochs=15, batch_size=32
            )
        except:
            return 0.0

# Use parallel evolution (requires adjustment to main loop)
search = ParallelEvolution(input_dim=1, output_dim=1, num_workers=4)
# Note: Requires modifying run() to use _evaluate_population()
```

### Example 10: Architecture Search Space Analysis

Analyze the search space defined by the grammar:

```python
from grammar import GRAMMAR, expand_grammar, is_terminal
import random

def sample_architectures(num_samples=100, max_gene_length=20):
    """Sample random architectures from grammar"""
    architectures = []
    
    for _ in range(num_samples):
        # Random genes
        gene_length = random.randint(5, max_gene_length)
        genes = [random.randint(0, 10) for _ in range(gene_length)]
        
        # Expand
        arch = expand_grammar(genes, GRAMMAR)
        arch_str = '-'.join(arch)
        architectures.append(arch_str)
    
    return architectures

# Sample and analyze
samples = sample_architectures(500)
unique = set(samples)

print(f"Sampled: {len(samples)} architectures")
print(f"Unique: {len(unique)} architectures")
print(f"Diversity: {len(unique) / len(samples) * 100:.1f}%")

# Most common patterns
from collections import Counter
counter = Counter(samples)
print("\nTop 5 most common architectures:")
for arch, count in counter.most_common(5):
    print(f"  {count:3d}x: {arch}")
```

## Tutorials

### Tutorial 1: Building Your First Neural Architecture

**Goal:** Understand the genome-to-model pipeline

**Step 1: Create a genome**
```python
from genome import ModelGenome

# Define genes (integers)
genes = [3, 0, 1, 0, 0]

# Define learning parameters
learning_params = {
    'optimizer': 'adam',
    'learning_rate': 0.005,
    'lr_scheduler': 'none'
}

# Create genome
genome = ModelGenome(genes, learning_params)
```

**Step 2: Expand to architecture**
```python
from grammar import GRAMMAR

# Expand genes using grammar
architecture = genome.build_from_grammar(GRAMMAR)
print(f"Architecture: {architecture}")
# Output: ['GRU', 'ReLU', 'LayerNorm', 'identity']
```

**Step 3: Build PyTorch model**
```python
# Build model
model = genome.build_pytorch_model(input_dim=1, output_dim=1)
print(model)
```

**Step 4: Train on data**
```python
import torch
from evolution import train_and_evaluate_genome

# Generate parity data
X = torch.randint(0, 2, (500, 8, 1)).float()
y = (X.sum(dim=1) % 2).float()

# Train and evaluate
fitness = train_and_evaluate_genome(
    genome, X, y, learning_params,
    epochs=20, batch_size=32
)
print(f"Fitness: {fitness:.4f}")
```

### Tutorial 2: Evolving for a Specific Task

**Goal:** Evolve architectures for sequence classification

**Step 1: Define task data generator**
```python
import torch

def generate_task_data(num_samples, seq_len):
    """Generate sequences where label = 1 if sum > threshold"""
    X = torch.rand(num_samples, seq_len, 1)
    y = (X.sum(dim=1) > seq_len/2).float()
    return X, y

X_train, y_train = generate_task_data(1000, 10)
X_test, y_test = generate_task_data(200, 10)
```

**Step 2: Configure evolution**
```python
from evolution import EvolutionarySearch

search = EvolutionarySearch(
    input_dim=1,
    output_dim=1,
    population_size=50,
    mutation_rate=0.3,
    crossover_rate=0.7
)
```

**Step 3: Run evolution**
```python
best_genome, fitness_hist, best_hist = search.run(
    generations=40,
    sequence_length=10,
    num_samples=1000
)
```

**Step 4: Evaluate on test set**
```python
from evolution import train_and_evaluate_genome

test_fitness = train_and_evaluate_genome(
    best_genome, X_test, y_test,
    best_genome.learning_params,
    epochs=20, batch_size=32
)
print(f"Test fitness: {test_fitness:.4f}")
```

**Step 5: Deploy best architecture**
```python
# Build final model
final_model = best_genome.build_pytorch_model(input_dim=1, output_dim=1)

# Save
torch.save(final_model.state_dict(), 'best_model.pth')

# Use for inference
final_model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(final_model(X_test))
    accuracy = ((predictions > 0.5).float() == y_test).float().mean()
print(f"Test accuracy: {accuracy:.4f}")
```

### Tutorial 3: Debugging Evolution Issues

**Problem:** Evolution gets stuck at low fitness

**Step 1: Check data generation**
```python
import torch

X, y = search._generate_data(sequence_length=8, num_samples=500)
print(f"X shape: {X.shape}, range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y shape: {y.shape}, unique: {torch.unique(y)}")
print(f"y balance: {y.mean():.2f}")  # Should be ~0.5 for parity
```

**Step 2: Test individual genomes**
```python
from genome import ModelGenome

# Create simple genome
simple_genome = ModelGenome([8, 0], {})  # Minimal architecture
print(f"Simple arch: {simple_genome.get_architecture_string()}")

fitness = search._evaluate_genome_fitness(simple_genome, X, y)
print(f"Simple fitness: {fitness:.4f}")
```

**Step 3: Check population diversity**
```python
def check_diversity(population):
    archs = [g.get_architecture_string() for g in population]
    unique = len(set(archs))
    print(f"Unique: {unique}/{len(population)} ({unique/len(population)*100:.1f}%)")

# During evolution
check_diversity(search.population)
```

**Step 4: Increase exploration**
```python
# Adjust hyperparameters
search.mutation_rate = 0.5  # Higher
search.diversity_pressure = 0.4  # Higher
search.novelty_weight = 0.3  # Higher
```

## Best Practices

### 1. Start Small

```python
# Good: Quick iteration
search = EvolutionarySearch(
    input_dim=1, output_dim=1,
    population_size=20,  # Small
)
best, _, _ = search.run(generations=10, sequence_length=8, num_samples=200)

# Then scale up
search.population_size = 100
best, _, _ = search.run(generations=50, sequence_length=8, num_samples=1000)
```

### 2. Monitor Diversity

```python
def log_generation_stats(generation, population, fitnesses):
    print(f"Gen {generation}:")
    print(f"  Fitness: {np.mean(fitnesses):.3f} ± {np.std(fitnesses):.3f}")
    archs = [g.get_architecture_string() for g in population]
    print(f"  Unique: {len(set(archs))}/{len(population)}")
```

### 3. Save Checkpoints

```python
import pickle

# During evolution, save every N generations
if generation % 10 == 0:
    checkpoint = {
        'generation': generation,
        'population': population,
        'fitness_history': fitness_history,
        'best_genome': best_genome
    }
    with open(f'checkpoint_gen{generation}.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
```

### 4. Use Early Stopping

```python
search = EvolutionarySearch(
    input_dim=1, output_dim=1,
    early_stop_fitness=0.95,  # Stop at 95% accuracy
    early_stop_patience=3      # Wait 3 generations
)
```

### 5. Experiment with Hyperparameters

```python
# Grid search over hyperparameters
for pop_size in [50, 100, 150]:
    for mut_rate in [0.2, 0.3, 0.4]:
        search = EvolutionarySearch(
            input_dim=1, output_dim=1,
            population_size=pop_size,
            mutation_rate=mut_rate
        )
        best, _, _ = search.run(generations=30, sequence_length=8, num_samples=500)
        print(f"Pop={pop_size}, Mut={mut_rate}: Best fitness={best.fitness:.4f}")
```

## Summary

These examples and tutorials cover:
- **Basic usage** for quick starts
- **Advanced techniques** for custom needs
- **Step-by-step tutorials** for learning
- **Best practices** for effective evolution

For more details:
- [Getting Started](getting-started.md) - Installation and basics
- [API Reference](api-reference.md) - Complete API documentation
- [Evolutionary Algorithms](evolution.md) - Algorithm details
