# Evolutionary Algorithms

This document explains the evolutionary mechanisms used in NoTransformers to discover neural architectures.

## Table of Contents

- [Overview](#overview)
- [Evolutionary Loop](#evolutionary-loop)
- [Selection Mechanisms](#selection-mechanisms)
- [Genetic Operators](#genetic-operators)
- [Fitness Evaluation](#fitness-evaluation)
- [Advanced Mechanisms](#advanced-mechanisms)
- [Hyperparameters](#hyperparameters)

## Overview

NoTransformers uses **genetic algorithms** to evolve neural network architectures. The system treats architecture design as an optimization problem where:

- **Population**: Collection of candidate architectures (genomes)
- **Fitness**: Performance on a specific task
- **Evolution**: Iterative improvement through selection, crossover, and mutation
- **Generations**: Discrete time steps in the evolutionary process

### Why Evolutionary Algorithms?

**Advantages:**
- **No gradient required**: Optimizes discrete structures (architectures)
- **Population-based**: Explores multiple solutions simultaneously
- **Robust**: Handles noisy fitness landscapes
- **Flexible**: Easy to add custom objectives and constraints

**Trade-offs:**
- **Computational cost**: Requires many fitness evaluations
- **Sample efficiency**: May need many generations to converge
- **Hyperparameter sensitivity**: Requires tuning of evolutionary parameters

## Evolutionary Loop

The main evolution loop in `EvolutionarySearch.run()`:

```python
def run(self, generations, sequence_length, num_samples):
    # 1. Initialize population
    population = [ModelGenome.create_random_genome() 
                  for _ in range(self.population_size)]
    
    # 2. Main evolutionary loop
    for generation in range(generations):
        # 3. Evaluate fitness
        fitnesses = [self._evaluate_genome_fitness(genome, X_data, y_data)
                     for genome in population]
        
        # 4. Apply diversity pressure
        fitnesses = self._apply_diversity_pressure(population, fitnesses)
        
        # 5. Compute novelty
        novelty_scores = self._compute_novelty_scores(population)
        
        # 6. Blend fitness with novelty
        combined_fitness = [
            (1 - novelty_weight) * f + novelty_weight * n
            for f, n in zip(fitnesses, novelty_scores)
        ]
        
        # 7. Selection
        parents = [self._tournament_selection(combined_fitness)
                   for _ in range(len(population))]
        
        # 8. Generate offspring
        offspring = []
        for i in range(0, len(parents), 2):
            if random.random() < crossover_rate:
                child1, child2 = parents[i].crossover(parents[i+1])
            else:
                child1, child2 = parents[i].clone(), parents[i+1].clone()
            
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            offspring.extend([child1, child2])
        
        # 9. Update population (elitism)
        best_idx = np.argmax(fitnesses)
        offspring[0] = population[best_idx].clone()
        population = offspring
        
        # 10. Track metrics
        self._update_metrics(generation, population, fitnesses, novelty_scores)
        
        # 11. Early stopping check
        if self._check_early_stopping(fitnesses):
            break
    
    # 12. Return best genome
    best_idx = np.argmax(fitnesses)
    return population[best_idx]
```

## Selection Mechanisms

### Tournament Selection

**Algorithm:**
```python
def _tournament_selection(self, fitnesses):
    # Randomly sample k individuals
    tournament_indices = random.sample(range(len(fitnesses)), self.tournament_size)
    
    # Select the best from the tournament
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    
    return self.population[winner_idx]
```

**Properties:**
- **Tournament size**: Controls selection pressure
  - Small (2-3): More exploration, slower convergence
  - Large (5-7): More exploitation, faster convergence
- **Probabilistic**: Weaker individuals can still be selected
- **Efficient**: O(k) per selection where k is tournament size

**Example:**
```
Population fitness: [0.5, 0.8, 0.3, 0.9, 0.6]
Tournament size: 3

Round 1: Sample indices [1, 3, 4] → fitness [0.8, 0.9, 0.6]
Winner: Index 3 (fitness 0.9)

Round 2: Sample indices [0, 2, 4] → fitness [0.5, 0.3, 0.6]
Winner: Index 4 (fitness 0.6)
```

### Alternative Selection Methods

While NoTransformers uses tournament selection, other methods include:

**1. Roulette Wheel Selection:**
```python
def roulette_wheel_selection(fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=probabilities)[0]
```

**2. Rank Selection:**
```python
def rank_selection(fitnesses):
    ranks = np.argsort(np.argsort(fitnesses))  # Convert to ranks
    probabilities = ranks / sum(ranks)
    return random.choices(population, weights=probabilities)[0]
```

**3. Truncation Selection:**
```python
def truncation_selection(fitnesses, keep_top=0.5):
    n_keep = int(len(fitnesses) * keep_top)
    top_indices = np.argsort(fitnesses)[-n_keep:]
    return random.choice([population[i] for i in top_indices])
```

## Genetic Operators

### Crossover

**Single-Point Crossover:**
```python
def crossover(self, other):
    # Crossover genes
    if len(self.genes) > 1 and len(other.genes) > 1:
        point1 = random.randint(1, len(self.genes) - 1)
        point2 = random.randint(1, len(other.genes) - 1)
        
        child1_genes = self.genes[:point1] + other.genes[point2:]
        child2_genes = other.genes[:point2] + self.genes[point1:]
    else:
        child1_genes = self.genes.copy()
        child2_genes = other.genes.copy()
    
    # Crossover learning parameters
    child1_params = {}
    child2_params = {}
    for key in self.learning_params:
        if random.random() < 0.5:
            child1_params[key] = self.learning_params[key]
            child2_params[key] = other.learning_params.get(key, self.learning_params[key])
        else:
            child1_params[key] = other.learning_params.get(key, self.learning_params[key])
            child2_params[key] = self.learning_params[key]
    
    return ModelGenome(child1_genes, child1_params), ModelGenome(child2_genes, child2_params)
```

**Visualization:**
```
Parent 1: [3, 7, 1, 2, 5, 8, 4]
Parent 2: [2, 4, 6, 1, 3, 9, 7]
                    ↑ crossover point

Child 1:  [3, 7, 1, 2, 3, 9, 7]  (Parent1[:4] + Parent2[4:])
Child 2:  [2, 4, 6, 1, 5, 8, 4]  (Parent2[:4] + Parent1[4:])
```

**Other Crossover Methods:**

**Uniform Crossover:**
```python
def uniform_crossover(parent1, parent2):
    child_genes = []
    for g1, g2 in zip(parent1.genes, parent2.genes):
        child_genes.append(g1 if random.random() < 0.5 else g2)
    return ModelGenome(child_genes, ...)
```

**Two-Point Crossover:**
```python
def two_point_crossover(parent1, parent2):
    p1, p2 = sorted(random.sample(range(len(parent1.genes)), 2))
    child_genes = parent1.genes[:p1] + parent2.genes[p1:p2] + parent1.genes[p2:]
    return ModelGenome(child_genes, ...)
```

### Mutation

**Gene Mutation:**
```python
def mutate(self, mutation_rate):
    # Mutate architecture genes
    for i in range(len(self.genes)):
        if random.random() < mutation_rate:
            self.genes[i] = random.randint(0, 10)
    
    # Mutate learning parameters
    if random.random() < mutation_rate:
        self.learning_params['learning_rate'] = random.choice([0.001, 0.005, 0.01])
    
    if random.random() < mutation_rate:
        self.learning_params['optimizer'] = random.choice(['adam', 'sgd', 'rmsprop'])
    
    if random.random() < mutation_rate:
        self.learning_params['lr_scheduler'] = random.choice(['none', 'step', 'cosine'])
    
    # Clear caches after mutation
    self.built_architecture = None
    self.model = None
```

**Mutation Types:**

**1. Point Mutation (used by NoTransformers):**
```
Before: [3, 7, 1, 2, 5, 8, 4]
Mutate:         ↑ (position 2)
After:  [3, 7, 9, 2, 5, 8, 4]
```

**2. Insertion Mutation:**
```python
def insert_mutation(genes):
    pos = random.randint(0, len(genes))
    genes.insert(pos, random.randint(0, 10))
```

**3. Deletion Mutation:**
```python
def delete_mutation(genes):
    if len(genes) > 1:
        pos = random.randint(0, len(genes) - 1)
        del genes[pos]
```

**4. Swap Mutation:**
```python
def swap_mutation(genes):
    i, j = random.sample(range(len(genes)), 2)
    genes[i], genes[j] = genes[j], genes[i]
```

## Fitness Evaluation

### Training and Evaluation Process

```python
def _evaluate_genome_fitness(self, genome, X_data, y_data):
    # Build PyTorch model from genome
    try:
        model = genome.build_pytorch_model(input_dim, output_dim)
    except Exception as e:
        return 0.0  # Invalid architecture
    
    # Train the model
    fitness = train_and_evaluate_genome(
        genome, X_data, y_data,
        learning_params=genome.learning_params,
        epochs=15,
        batch_size=32
    )
    
    # Apply complexity penalty
    param_count = sum(p.numel() for p in model.parameters())
    complexity_penalty = self.complexity_penalty_coef * param_count
    
    # Final fitness
    return max(0.0, fitness - complexity_penalty)
```

### Multi-Run Evaluation

For robust fitness estimates:

```python
def _evaluate_genome_fitness_multi_run(self, genome, X_data, y_data):
    fitnesses = []
    for run in range(self.num_eval_runs):
        seed = self.base_seed + run if self.base_seed else None
        fitness = train_and_evaluate_genome(
            genome, X_data, y_data,
            learning_params=genome.learning_params,
            random_seed=seed
        )
        fitnesses.append(fitness)
    
    # Return mean fitness
    return np.mean(fitnesses)
```

### Fitness Components

**1. Task Performance (primary):**
- Classification accuracy
- Regression MSE
- Task-specific metrics

**2. Complexity Penalty (regularization):**
```python
complexity_penalty = complexity_coef * num_parameters
```

**3. Novelty Reward (exploration):**
```python
novelty_score = mean_distance_to_population(genome)
```

**4. Combined Fitness:**
```python
final_fitness = (
    (1 - novelty_weight) * task_performance
    + novelty_weight * novelty_score
    - complexity_penalty
)
```

## Advanced Mechanisms

### 1. Novelty Search

**Purpose:** Encourage exploration of diverse architectures

**Algorithm:**
```python
def _compute_novelty_scores(self, population):
    novelty_scores = []
    
    for i, genome_i in enumerate(population):
        arch_i = genome_i.get_architecture_string()
        distances = []
        
        # Compute distance to all other genomes
        for j, genome_j in enumerate(population):
            if i != j:
                arch_j = genome_j.get_architecture_string()
                distance = self._levenshtein_distance(arch_i, arch_j)
                distances.append(distance)
        
        # Novelty = mean distance to k-nearest neighbors
        k = min(15, len(distances))
        distances.sort()
        novelty = np.mean(distances[:k])
        novelty_scores.append(novelty)
    
    # Normalize to [0, 1]
    max_novelty = max(novelty_scores) if novelty_scores else 1.0
    return [n / max_novelty for n in novelty_scores]
```

**Levenshtein Distance:**
```python
def _levenshtein_distance(self, s1, s2):
    """Edit distance between two strings"""
    if len(s1) < len(s2):
        return self._levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
```

### 2. Diversity Pressure

**Purpose:** Penalize duplicate architectures to maintain population diversity

**Algorithm:**
```python
def _apply_diversity_pressure(self, population, fitnesses):
    arch_counts = {}
    for genome in population:
        arch_str = genome.get_architecture_string()
        arch_counts[arch_str] = arch_counts.get(arch_str, 0) + 1
    
    adjusted_fitnesses = []
    for genome, fitness in zip(population, fitnesses):
        arch_str = genome.get_architecture_string()
        count = arch_counts[arch_str]
        
        if count > 1:
            # Penalty proportional to number of copies
            penalty = self.diversity_pressure * (count - 1) / count
            adjusted_fitness = fitness * (1 - penalty)
        else:
            adjusted_fitness = fitness
        
        adjusted_fitnesses.append(adjusted_fitness)
    
    return adjusted_fitnesses
```

**Example:**
```
Original population:
- Arch A (3 copies): fitness = 0.8
- Arch B (1 copy): fitness = 0.75
- Arch C (2 copies): fitness = 0.7

With diversity_pressure = 0.2:
- Arch A: 0.8 * (1 - 0.2 * 2/3) = 0.8 * 0.867 = 0.693
- Arch B: 0.75 * (1 - 0) = 0.75
- Arch C: 0.7 * (1 - 0.2 * 1/2) = 0.7 * 0.9 = 0.63
```

### 3. Early Stopping

**Purpose:** Terminate evolution when fitness plateaus

**Algorithm:**
```python
def _check_early_stopping(self, fitnesses):
    best_fitness = max(fitnesses)
    
    # Check if fitness exceeds threshold
    if best_fitness >= self.early_stop_fitness:
        self.early_stop_counter += 1
    else:
        self.early_stop_counter = 0
    
    # Stop if threshold maintained for patience generations
    if self.early_stop_counter >= self.early_stop_patience:
        print(f"Early stopping: fitness {best_fitness:.4f} >= {self.early_stop_fitness}")
        return True
    
    return False
```

### 4. Elitism

**Purpose:** Preserve best solutions across generations

**Algorithm:**
```python
def _apply_elitism(self, population, offspring, fitnesses, num_elites=1):
    # Find best individuals
    elite_indices = np.argsort(fitnesses)[-num_elites:]
    elites = [population[i].clone() for i in elite_indices]
    
    # Replace worst offspring with elites
    offspring[:num_elites] = elites
    
    return offspring
```

### 5. Adaptive Mutation Rate

**Purpose:** Adjust mutation rate based on population diversity

```python
def _adaptive_mutation_rate(self, population, base_rate=0.3):
    # Measure diversity
    unique_archs = len(set(g.get_architecture_string() for g in population))
    diversity = unique_archs / len(population)
    
    # Increase mutation if diversity is low
    if diversity < 0.3:
        mutation_rate = base_rate * 1.5
    elif diversity > 0.7:
        mutation_rate = base_rate * 0.5
    else:
        mutation_rate = base_rate
    
    return mutation_rate
```

## Hyperparameters

### Key Parameters and Their Effects

| Parameter | Default | Effect | Tuning Advice |
|-----------|---------|--------|---------------|
| `population_size` | 100 | Number of architectures per generation | Larger = more diversity, slower |
| `generations` | 100 | Evolution duration | More = better solutions, longer runtime |
| `mutation_rate` | 0.3 | Probability of gene mutation | Higher = more exploration |
| `crossover_rate` | 0.7 | Probability of crossover | Higher = more recombination |
| `tournament_size` | 3 | Selection pressure | Larger = faster convergence |
| `complexity_penalty` | 1e-6 | Regularization strength | Higher = smaller models |
| `diversity_pressure` | 0.2 | Duplicate architecture penalty | Higher = more unique solutions |
| `novelty_weight` | 0.15 | Exploration vs exploitation | Higher = more exploration |
| `early_stop_fitness` | 0.999 | Termination threshold | Task-dependent |
| `early_stop_patience` | 2 | Generations to confirm stopping | Higher = less premature stopping |

### Tuning Strategies

**For faster convergence:**
- Increase tournament size (5-7)
- Decrease mutation rate (0.1-0.2)
- Increase crossover rate (0.8-0.9)
- Reduce novelty weight (0.05-0.1)

**For better exploration:**
- Increase population size (150-200)
- Increase mutation rate (0.4-0.5)
- Increase novelty weight (0.2-0.3)
- Increase diversity pressure (0.3-0.4)

**For computational efficiency:**
- Reduce population size (20-50)
- Enable early stopping
- Reduce number of training epochs
- Use smaller datasets

## Summary

The evolutionary algorithms in NoTransformers provide:
- **Robust architecture search** through population-based optimization
- **Diverse solutions** via novelty search and diversity pressure
- **Efficient exploration** through tournament selection and genetic operators
- **Quality control** via elitism and early stopping

For practical usage, see [Getting Started](getting-started.md).
For competitive evolution, see [Co-Evolution Guide](coevolution.md).
