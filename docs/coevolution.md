# Co-Evolution Guide

This document explains the competitive co-evolution system in NoTransformers where solvers and saboteurs evolve together in an adversarial arms race.

## Table of Contents

- [Overview](#overview)
- [Co-Evolution Concept](#co-evolution-concept)
- [Solver Population](#solver-population)
- [Saboteur Population](#saboteur-population)
- [Co-Evolution Loop](#co-evolution-loop)
- [Saboteur Strategies](#saboteur-strategies)
- [Metrics and Analysis](#metrics-and-analysis)
- [Use Cases](#use-cases)

## Overview

**Competitive Co-Evolution** is an advanced evolutionary strategy where two populations evolve simultaneously in an adversarial relationship:

- **Solvers** (ModelGenomes): Neural architectures that try to solve tasks
- **Saboteurs** (SaboteurGenomes): Adversarial pattern generators that create hard examples

The key insight: **Making the problem harder drives innovation in solutions.**

### Why Co-Evolution?

**Traditional Evolution Problems:**
- Fixed datasets can be "gamed" by overfitting
- Solutions may not generalize to distribution shifts
- Limited pressure for robust architectures

**Co-Evolution Benefits:**
- **Adaptive difficulty**: Task difficulty increases with solver capability
- **Robustness**: Solvers learn to handle adversarial inputs
- **Generalization**: Less overfitting to specific data distributions
- **Red team/Blue team**: Continuous challenge-response dynamics

## Co-Evolution Concept

### Evolutionary Arms Race

```
Generation 1:
Solvers: [Simple architectures with ~60% accuracy]
Saboteurs: [Basic patterns, easy to solve]

Generation 5:
Solvers: [More complex, ~75% accuracy on adversarial data]
Saboteurs: [Harder patterns discovered, challenging solvers]

Generation 10:
Solvers: [Robust architectures, ~85% accuracy]
Saboteurs: [Sophisticated adversarial patterns]
```

### Fitness Coupling

The populations are coupled through their fitness functions:

```python
# Solver fitness: How well it performs on saboteur-generated data
solver_fitness = accuracy_on_saboteur_data

# Saboteur fitness: How much it challenges solvers
saboteur_fitness = 1 - solver_accuracy

# Key property: Zero-sum game (approximately)
solver_fitness + saboteur_fitness ≈ 1.0
```

## Solver Population

### Solver Architecture

Solvers are standard `ModelGenome` instances that:
- Build neural architectures via grammar expansion
- Have learning parameters (optimizer, learning rate, etc.)
- Evolve through standard genetic operators

### Solver Evaluation

```python
def evaluate_solver(solver, saboteur_population):
    total_fitness = 0.0
    
    # Test against multiple saboteurs
    for saboteur in saboteur_population:
        # Generate adversarial data
        X_adv, y_adv = saboteur.generate_batch(batch_size=500, seq_len=8)
        
        # Train and evaluate solver
        fitness = train_and_evaluate_genome(
            solver, X_adv, y_adv,
            learning_params=solver.learning_params,
            epochs=15
        )
        total_fitness += fitness
    
    # Average performance across saboteurs
    return total_fitness / len(saboteur_population)
```

### Solver Evolution Focus

Solvers evolve to:
- **Generalize better**: Handle diverse pattern types
- **Robust features**: Learn features that work across distributions
- **Adaptive capacity**: Adjust to changing adversarial patterns

## Saboteur Population

### Saboteur Genome Structure

```python
class SaboteurGenome:
    def __init__(self, params):
        self.params = {
            'pattern_length': int,      # Length of pattern unit
            'pattern_type': str,        # Type of adversarial strategy
            'noise_level': float,       # Amount of noise to add
            'density': float,           # For sparse patterns
        }
```

### Pattern Generation

```python
def generate_sequence(self, seq_len):
    pattern_type = self.params['pattern_type']
    
    if pattern_type == 'alternating':
        # Simple alternating pattern
        seq = [(i % 2) for i in range(seq_len)]
    
    elif pattern_type == 'repeating_chunk':
        # Repeat a small chunk
        chunk = random.choice([[1,1,0], [1,0,0], [1,0,1,0]])
        seq = [chunk[i % len(chunk)] for i in range(seq_len)]
    
    elif pattern_type == 'mostly_zeros':
        # Sparse positive examples
        density = self.params['density']
        seq = [1 if random.random() < density else 0 
               for _ in range(seq_len)]
    
    elif pattern_type == 'mostly_ones':
        # Sparse negative examples
        density = self.params['density']
        seq = [0 if random.random() < density else 1 
               for _ in range(seq_len)]
    
    elif pattern_type == 'edge_ones':
        # Active at boundaries, quiet in center
        L = self.params['pattern_length']
        seq = [1 if (i < L or i >= seq_len - L) else 0 
               for i in range(seq_len)]
    
    # Add noise
    noise_level = self.params['noise_level']
    for i in range(seq_len):
        if random.random() < noise_level:
            seq[i] = 1 - seq[i]
    
    return torch.tensor(seq, dtype=torch.float32).view(seq_len, 1)
```

### Saboteur Evaluation

```python
def evaluate_saboteur(saboteur, solver_population):
    total_difficulty = 0.0
    
    # Test against multiple solvers
    for solver in solver_population:
        # Generate adversarial data
        X_adv, y_adv = saboteur.generate_batch(batch_size=500, seq_len=8)
        
        # Measure solver's difficulty
        solver_accuracy = train_and_evaluate_genome(
            solver, X_adv, y_adv,
            learning_params=solver.learning_params,
            epochs=15
        )
        
        # Saboteur wants low solver accuracy
        difficulty = 1 - solver_accuracy
        total_difficulty += difficulty
    
    # Average difficulty caused
    return total_difficulty / len(solver_population)
```

## Co-Evolution Loop

### Main Algorithm

```python
def run_coevolution(self, generations):
    # Initialize both populations
    solver_pop = [ModelGenome.create_random_genome() 
                  for _ in range(self.population_size)]
    saboteur_pop = [SaboteurGenome.random_genome() 
                    for _ in range(self.population_size)]
    
    for generation in range(generations):
        # 1. Evaluate Solvers against Saboteurs
        solver_fitnesses = []
        for solver in solver_pop:
            # Sample saboteurs for evaluation
            test_saboteurs = random.sample(saboteur_pop, k=10)
            fitness = self._evaluate_solver_vs_saboteurs(solver, test_saboteurs)
            solver_fitnesses.append(fitness)
        
        # 2. Evaluate Saboteurs against Solvers
        saboteur_fitnesses = []
        for saboteur in saboteur_pop:
            # Sample solvers for evaluation
            test_solvers = random.sample(solver_pop, k=10)
            fitness = self._evaluate_saboteur_vs_solvers(saboteur, test_solvers)
            saboteur_fitnesses.append(fitness)
        
        # 3. Evolve Solver Population
        solver_parents = [self._tournament_selection(solver_fitnesses) 
                         for _ in range(len(solver_pop))]
        solver_offspring = self._generate_offspring(solver_parents, 
                                                     mutation_rate=0.3)
        solver_pop = solver_offspring
        
        # 4. Evolve Saboteur Population
        saboteur_parents = [self._tournament_selection(saboteur_fitnesses) 
                           for _ in range(len(saboteur_pop))]
        saboteur_offspring = []
        for i in range(0, len(saboteur_parents), 2):
            child1, child2 = saboteur_parents[i].crossover(saboteur_parents[i+1])
            child1.mutate(rate=0.2)
            child2.mutate(rate=0.2)
            saboteur_offspring.extend([child1, child2])
        saboteur_pop = saboteur_offspring
        
        # 5. Track Co-evolution Metrics
        self._update_coevo_metrics(generation, 
                                    solver_fitnesses, 
                                    saboteur_fitnesses)
    
    return solver_pop, saboteur_pop
```

### Visualization of Co-Evolution Dynamics

```
Fitness Over Generations:

1.0 ┤
    │     Solvers ───────
    │    ╱              ╲
0.8 ┤   ╱                ╲     ╱
    │  ╱                  ╲   ╱
0.6 ┤ ╱                    ╲ ╱
    │╱                      ╳
0.4 ┤                      ╱ ╲
    │                     ╱   ╲
0.2 ┤                    ╱     ╲
    │    Saboteurs ─────      ╲───
0.0 ┤
    └─────────────────────────────
    1    5   10   15   20   25
         Generation

Pattern: Oscillating competition
- Solvers improve → Saboteurs get weaker
- Saboteurs adapt → Solvers struggle
- Repeat with increasing sophistication
```

## Saboteur Strategies

### 1. Alternating Pattern

**Strategy:** Simple 1,0,1,0,... pattern

**Example:**
```
Sequence: [1, 0, 1, 0, 1, 0, 1, 0]
Parity:   Sum = 4 (even) → Label = 0
```

**Difficulty:** Easy initially, solvers learn quickly

**Evolution:** Rarely survives past early generations

### 2. Repeating Chunk

**Strategy:** Repeat small patterns like [1,1,0]

**Example:**
```
Chunk: [1, 1, 0]
Sequence: [1, 1, 0, 1, 1, 0, 1, 1]
Parity:   Sum = 6 (even) → Label = 0
```

**Difficulty:** Moderate, requires sequence learning

**Evolution:** Common in mid-generation saboteurs

### 3. Mostly Zeros

**Strategy:** Sparse 1s in a sea of 0s

**Example:**
```
Sequence: [0, 0, 1, 0, 0, 0, 1, 0]  (density=0.25)
Parity:   Sum = 2 (even) → Label = 0
```

**Difficulty:** Hard due to class imbalance

**Evolution:** Emerges when solvers get too confident

### 4. Mostly Ones

**Strategy:** Sparse 0s in a sea of 1s

**Example:**
```
Sequence: [1, 1, 0, 1, 1, 1, 0, 1]  (density=0.25)
Parity:   Sum = 6 (even) → Label = 0
```

**Difficulty:** Hard, opposite of mostly_zeros

**Evolution:** Counter-strategy to mostly_zeros solvers

### 5. Edge Ones

**Strategy:** Ones at boundaries, zeros in center

**Example:**
```
Pattern length: 2
Sequence: [1, 1, 0, 0, 0, 0, 1, 1]
Parity:   Sum = 4 (even) → Label = 0
```

**Difficulty:** Very hard, requires position awareness

**Evolution:** Late-game saboteur strategy

### Pattern Complexity Evolution

```
Early Generations (1-5):
- Alternating: 40%
- Repeating chunk: 30%
- Random: 30%

Mid Generations (6-15):
- Repeating chunk: 35%
- Mostly zeros: 25%
- Edge ones: 20%
- Alternating: 20%

Late Generations (16+):
- Edge ones: 40%
- Mostly zeros/ones: 35%
- Repeating chunk: 25%
```

## Metrics and Analysis

### Standard Co-Evolution Metrics

```json
{
  "generation": 10,
  "solver_fitness_mean": 0.612,
  "solver_fitness_best": 0.891,
  "solver_fitness_std": 0.124,
  "saboteur_fitness_mean": 0.388,
  "saboteur_fitness_best": 0.742,
  "saboteur_fitness_std": 0.089,
  "solver_arch_entropy": 1.523,
  "saboteur_pattern_entropy": 1.234
}
```

### Pattern Type Distribution

```json
{
  "pattern_type_counts": {
    "alternating": 8,
    "repeating_chunk": 12,
    "mostly_zeros": 5,
    "mostly_ones": 6,
    "edge_ones": 9
  }
}
```

### Solver Architecture Diversity

Track unique architectures in solver population:

```python
def compute_solver_diversity(solver_population):
    architectures = [s.get_architecture_string() for s in solver_population]
    unique_count = len(set(architectures))
    entropy = -sum((architectures.count(a)/len(architectures)) * 
                   np.log(architectures.count(a)/len(architectures)) 
                   for a in set(architectures))
    return unique_count, entropy
```

### Co-Evolution Phase Analysis

**Phase 1: Initial Chaos (Generations 1-5)**
- Both populations random
- High variance in fitness
- No clear strategies

**Phase 2: Solver Dominance (Generations 6-12)**
- Solvers find good architectures
- Saboteurs struggle
- Solver fitness > 0.7

**Phase 3: Saboteur Adaptation (Generations 13-20)**
- Saboteurs discover hard patterns
- Solver fitness drops
- Arms race begins

**Phase 4: Co-Adaptation (Generations 20+)**
- Both populations specialized
- Oscillating fitness
- Stable competition

## Use Cases

### 1. Robust Architecture Discovery

**Goal:** Find architectures that generalize well

**Setup:**
```bash
python main.py --coevolution --population 40 --generations 30
```

**Benefits:**
- Architectures tested on diverse data
- Less overfitting to specific distributions
- Better out-of-distribution performance

### 2. Adversarial Training Research

**Goal:** Study adversarial robustness

**Analysis:**
```python
# Train best solver on adversarial data
best_solver = solver_pop[np.argmax(solver_fitnesses)]
best_saboteur = saboteur_pop[np.argmax(saboteur_fitnesses)]

# Generate hard test set
X_hard, y_hard = best_saboteur.generate_batch(1000, 8)

# Evaluate
test_accuracy = evaluate_model(best_solver, X_hard, y_hard)
```

### 3. Curriculum Learning

**Goal:** Progressive difficulty increase

**Implementation:**
```python
def curriculum_coevolution(generations):
    # Start with weak saboteurs
    saboteur_pop = [SaboteurGenome({'pattern_type': 'alternating', ...}) 
                    for _ in range(population_size)]
    
    # Evolve in phases
    for phase in range(3):
        solver_pop, saboteur_pop = run_coevolution(
            generations=generations//3,
            solver_pop=solver_pop,
            saboteur_pop=saboteur_pop
        )
        # Difficulty increases automatically through evolution
```

### 4. Multi-Task Learning

**Goal:** Evolve solvers for multiple tasks

**Setup:**
```python
# Multiple saboteur types for different tasks
task1_saboteurs = [SaboteurGenome(...) for _ in range(20)]
task2_saboteurs = [DifferentSaboteur(...) for _ in range(20)]

# Evaluate solvers on both
fitness = 0.5 * evaluate_on_task1(solver, task1_saboteurs) + \
          0.5 * evaluate_on_task2(solver, task2_saboteurs)
```

## Advanced Topics

### Unbalanced Co-Evolution

Use different population sizes:

```python
solver_population_size = 50
saboteur_population_size = 100  # More saboteurs for diversity
```

### Multi-Population Co-Evolution

Multiple solver types compete:

```python
# Different solver starting points
conv_solvers = [initialize_conv_biased_genome() for _ in range(25)]
rnn_solvers = [initialize_rnn_biased_genome() for _ in range(25)]

# Evolve against shared saboteurs
all_solvers = conv_solvers + rnn_solvers
```

### Asymmetric Fitness

Weight solver/saboteur fitness differently:

```python
# Emphasize solver improvement
solver_fitness = accuracy
saboteur_fitness = 0.5 * (1 - accuracy)  # Half weight
```

## Troubleshooting

### Problem: Solvers dominate (fitness > 0.9 consistently)

**Solutions:**
- Increase saboteur mutation rate
- Add more pattern types
- Increase saboteur population size

### Problem: Saboteurs dominate (solver fitness < 0.4)

**Solutions:**
- Decrease saboteur mutation rate
- Increase solver population size
- Use curriculum learning (start easy)

### Problem: No arms race (both populations stagnate)

**Solutions:**
- Increase mutation rates for both
- Add novelty search
- Reset saboteur population periodically

### Problem: High variance in fitness

**Solutions:**
- Increase number of evaluation runs
- Use larger batch sizes
- Smooth fitness over multiple generations

## Summary

Co-evolution in NoTransformers:
- **Creates robust solvers** through adversarial pressure
- **Discovers hard patterns** automatically via saboteurs
- **Enables curriculum learning** through adaptive difficulty
- **Provides research platform** for adversarial robustness

For basic usage, see [Getting Started](getting-started.md).
For evolutionary mechanisms, see [Evolutionary Algorithms](evolution.md).
