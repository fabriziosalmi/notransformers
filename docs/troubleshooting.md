# Troubleshooting and FAQ

Common issues, solutions, and frequently asked questions about NoTransformers.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Evolution Problems](#evolution-problems)
- [Co-Evolution Issues](#co-evolution-issues)
- [Frequently Asked Questions](#frequently-asked-questions)

## Installation Issues

### Problem: "No module named 'torch'"

**Cause:** PyTorch not installed

**Solution:**
```bash
pip install torch
```

For GPU support:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Problem: "No module named 'graphviz'" when visualizing

**Cause:** Graphviz not installed

**Solution:**

**macOS:**
```bash
brew install graphviz
pip install graphviz
```

**Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Windows:**
1. Download from [graphviz.org](https://graphviz.org/download/)
2. Install and add to PATH
3. Run `pip install graphviz`

**Workaround:** Skip visualization
```python
# Comment out visualization calls
# visualize_genome(best_genome, "output.png")
```

### Problem: Import errors in evolutionary_discovery

**Cause:** Running from wrong directory

**Solution:**
```bash
cd /path/to/notransformers/evolutionary_discovery
python main.py --population 50 --generations 40
```

Or add to Python path:
```python
import sys
sys.path.append('/path/to/notransformers/evolutionary_discovery')
```

## Runtime Errors

### Problem: "CUDA out of memory"

**Cause:** GPU memory exhausted

**Solutions:**

**1. Reduce batch size**
```python
# In evolution.py, modify train_and_evaluate_genome call
fitness = train_and_evaluate_genome(
    genome, X_data, y_data,
    learning_params=genome.learning_params,
    epochs=15,
    batch_size=16  # Reduced from 32
)
```

**2. Reduce population size**
```bash
python main.py --population 20 --generations 40
```

**3. Force CPU usage**
```python
import torch
# Add at the top of evolution.py
torch.cuda.is_available = lambda: False
```

**4. Clear cache between evaluations**
```python
# In evolution.py _evaluate_genome_fitness
torch.cuda.empty_cache()  # After each evaluation
```

### Problem: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**Cause:** Dimension mismatch in architecture

**Explanation:** The dynamic initialization should handle this, but complex architectures may fail.

**Solutions:**

**1. Check architecture**
```python
genome = ModelGenome([3, 0, 1, 2, 0], {})
print(f"Architecture: {genome.get_architecture_string()}")
arch = genome.build_from_grammar()
print(f"Terminals: {arch}")
```

**2. Test model building**
```python
try:
    model = genome.build_pytorch_model(input_dim=1, output_dim=1)
    print("Model built successfully")
except Exception as e:
    print(f"Error building model: {e}")
```

**3. Simplify grammar**
Remove complex operations that cause issues:
```python
# In grammar.py, temporarily simplify
GRAMMAR["<block>"] = [
    ["<dense_block>"]  # Only dense blocks for testing
]
```

### Problem: "Model has no trainable parameters"

**Cause:** Architecture contains only pass-through operations

**Solution:** This is handled automatically - fitness returns baseline ~0.3-0.4

**To prevent:**
```python
# In genome.py build_pytorch_model, ensure final layer
if len(layers) == 0:
    layers.append(nn.Linear(input_dim, output_dim))
```

### Problem: "TypeError: can't convert np.ndarray of type numpy.object_"

**Cause:** Mixing numpy and torch tensors

**Solution:**
```python
# Ensure consistent types
if isinstance(X_data, np.ndarray):
    X_data = torch.from_numpy(X_data).float()
if isinstance(y_data, np.ndarray):
    y_data = torch.from_numpy(y_data).float()
```

## Performance Issues

### Problem: Evolution is very slow

**Causes and Solutions:**

**1. Large population**
```bash
# Reduce population size
python main.py --population 30 --generations 50
```

**2. Many training epochs**
```python
# In main.py, reduce epochs in train_and_evaluate_genome
# Default is 15, try 10 or even 5
```

**3. Large dataset**
```python
# In main.py, reduce NUM_SAMPLES
NUM_SAMPLES = 200  # Instead of 500
```

**4. Multiple evaluation runs**
```python
# In main.py
NUM_EVAL_RUNS = 1  # Instead of 3
```

**5. Use GPU**
```python
# Check if GPU is being used
import torch
print(f"Using GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Problem: High memory usage

**Solutions:**

**1. Reduce population size**
```python
population_size = 50  # Instead of 100
```

**2. Clear model caches**
```python
# After evaluation
genome.model = None
genome.built_architecture = None
```

**3. Disable caching**
```python
# In genome.py build_pytorch_model
# Don't cache models
# self.model = model  # Comment out
return model
```

**4. Use smaller models**
```python
complexity_penalty_coef = 1e-5  # Higher penalty for large models
```

## Evolution Problems

### Problem: Fitness stuck at low values (< 0.5)

**Diagnostic Steps:**

**1. Check data generation**
```python
search = EvolutionarySearch(input_dim=1, output_dim=1)
X, y = search._generate_data(sequence_length=8, num_samples=500)
print(f"X shape: {X.shape}")
print(f"y unique: {torch.unique(y)}")
print(f"y mean: {y.mean():.3f}")  # Should be ~0.5
```

**2. Test simple baseline**
```python
# Try simple linear model
simple_genome = ModelGenome([8, 0], {})  # Just Linear layer
fitness = search._evaluate_genome_fitness(simple_genome, X, y)
print(f"Baseline fitness: {fitness:.4f}")
```

**3. Increase exploration**
```python
search = EvolutionarySearch(
    input_dim=1, output_dim=1,
    mutation_rate=0.5,        # Higher
    diversity_pressure=0.3,   # Higher
    novelty_weight=0.25       # Higher
)
```

**4. Check learning parameters**
```python
# Try different learning rates
for lr in [0.001, 0.005, 0.01, 0.05]:
    genome.learning_params['learning_rate'] = lr
    fitness = search._evaluate_genome_fitness(genome, X, y)
    print(f"LR {lr}: fitness {fitness:.4f}")
```

### Problem: No diversity in population

**Symptoms:** All genomes have same architecture

**Solutions:**

**1. Increase diversity pressure**
```python
diversity_pressure = 0.4  # Default 0.2
```

**2. Increase mutation rate**
```python
mutation_rate = 0.4  # Default 0.3
```

**3. Add novelty search**
```python
novelty_weight = 0.3  # Default 0.15
```

**4. Check initialization**
```python
# Ensure random initialization
population = [ModelGenome.create_random_genome(input_dim=1, output_dim=1)
              for _ in range(population_size)]
archs = [g.get_architecture_string() for g in population]
print(f"Initial unique: {len(set(archs))}/{len(population)}")
```

### Problem: Evolution plateaus early

**Cause:** Premature convergence

**Solutions:**

**1. Disable early stopping**
```python
early_stop_fitness = 1.0  # Effectively disable
```

**2. Reduce tournament size**
```python
tournament_size = 2  # Less selection pressure
```

**3. Increase population**
```python
population_size = 150
```

**4. Adaptive mutation**
```python
# Increase mutation when diversity drops
if unique_architectures < population_size * 0.3:
    current_mutation_rate = 0.6
else:
    current_mutation_rate = 0.3
```

### Problem: Best fitness decreases over generations

**Cause:** Elitism not working or fitness evaluation variance

**Solutions:**

**1. Check elitism**
```python
# In evolution.py run(), ensure elite preservation
best_idx = np.argmax(fitnesses)
offspring[0] = population[best_idx].clone()
```

**2. Use multiple evaluation runs**
```python
num_eval_runs = 3  # Average over multiple runs
```

**3. Increase training epochs**
```python
epochs = 20  # More stable fitness estimates
```

## Co-Evolution Issues

### Problem: Solvers dominate (fitness > 0.9)

**Cause:** Saboteurs not challenging enough

**Solutions:**

**1. Increase saboteur mutation**
```python
# In run_coevolution
saboteur.mutate(rate=0.3)  # Higher than default 0.2
```

**2. Add harder pattern types**
```python
# In saboteur.py PATTERN_TYPES
PATTERN_TYPES = [
    'alternating',
    'repeating_chunk',
    'mostly_zeros',
    'mostly_ones',
    'edge_ones',
    'random_sparse',     # New
    'gradient_pattern'   # New
]
```

**3. Increase saboteur population**
```python
# More saboteurs than solvers
solver_pop_size = 30
saboteur_pop_size = 50
```

### Problem: Saboteurs dominate (solver fitness < 0.4)

**Cause:** Task too hard or solvers not evolving

**Solutions:**

**1. Curriculum learning**
```python
# Start with easier patterns
for gen in range(generations):
    if gen < 10:
        # Use only simple patterns
        test_saboteurs = [s for s in saboteur_pop 
                         if s.params['pattern_type'] == 'alternating']
    else:
        test_saboteurs = random.sample(saboteur_pop, k=10)
```

**2. Increase solver mutation**
```python
solver.mutate(mutation_rate=0.4)
```

**3. Give solvers more training**
```python
# In solver evaluation
fitness = train_and_evaluate_genome(
    solver, X_adv, y_adv,
    learning_params=solver.learning_params,
    epochs=25  # Increased from 15
)
```

### Problem: No arms race (both populations stagnate)

**Solutions:**

**1. Reset weak population**
```python
if generation % 20 == 0:
    # Reset weaker population
    if np.mean(solver_fitnesses) < 0.5:
        solver_pop = [ModelGenome.create_random_genome(1, 1) 
                      for _ in range(len(solver_pop))]
```

**2. Increase exploration**
```python
mutation_rate = 0.5
diversity_pressure = 0.4
```

**3. Cross-population innovation**
```python
# Allow best solver genes to influence saboteurs
if generation % 10 == 0:
    best_solver = solver_pop[np.argmax(solver_fitnesses)]
    # Use solver complexity to inform saboteur patterns
```

## Frequently Asked Questions

### Q: How long does evolution take?

**A:** Depends on:
- Population size: 50 genomes × 40 generations = 2000 evaluations
- Training time: ~1-5 seconds per genome
- **Total:** 30 minutes to 3 hours for typical runs

**Speed up:**
- Reduce population: 20-30 genomes
- Reduce generations: 20-30 generations
- Use GPU
- Reduce training epochs: 10 instead of 15

### Q: What fitness values are good?

**A:** Task-dependent:
- **Parity task (8-bit):** 
  - Random baseline: ~0.5
  - Good: > 0.7
  - Excellent: > 0.85
- **Classification tasks:**
  - Depends on class balance
  - Compare to simple baseline (linear model)
- **Regression tasks:**
  - Normalized MSE or R²
  - Compare to baseline predictor

### Q: Can I use my own tasks?

**A:** Yes! Modify data generation:

```python
def custom_data_generator(num_samples, seq_len):
    # Your custom task
    X = generate_custom_inputs(num_samples, seq_len)
    y = generate_custom_labels(X)
    return X, y

# In evolution.py, replace _generate_data
search._generate_data = custom_data_generator
```

### Q: How do I save and load evolved architectures?

**A:** 

**Save genome:**
```python
import pickle
with open('best_genome.pkl', 'wb') as f:
    pickle.dump(best_genome, f)
```

**Load and use:**
```python
with open('best_genome.pkl', 'rb') as f:
    loaded_genome = pickle.load(f)

model = loaded_genome.build_pytorch_model(input_dim=1, output_dim=1)
torch.save(model.state_dict(), 'model_weights.pth')
```

### Q: Can I use different neural network layers?

**A:** Yes! Add to grammar:

**1. Define in primitives.py:**
```python
class LSTM(ComputationalPrimitive):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out.mean(dim=1)  # Aggregate
```

**2. Add to grammar.py:**
```python
GRAMMAR["<block>"].append(["<lstm_block>"])
GRAMMAR["<lstm_block>"] = [
    ["LSTM", "<activation>", "LayerNorm"]
]
```

**3. Handle in genome.py build_pytorch_model:**
```python
elif terminal == "LSTM":
    hidden_size = max(8, current_dim)
    lstm = nn.LSTM(current_dim, hidden_size, batch_first=True)
    layers.append(LSTMWrapper(lstm))
    current_dim = hidden_size
```

### Q: Why does the same genome give different fitness values?

**A:** Training variance due to:
- Random weight initialization
- Random batch ordering
- Dropout (if added)

**Solutions:**
- Use `num_eval_runs > 1` for averaging
- Set random seeds for reproducibility
- Increase training epochs for stability

### Q: How do I tune hyperparameters?

**A:** Start with defaults, then:

**For faster convergence:**
- ↑ Tournament size (5-7)
- ↓ Mutation rate (0.1-0.2)
- ↑ Crossover rate (0.8-0.9)

**For better exploration:**
- ↑ Population size (150-200)
- ↑ Mutation rate (0.4-0.5)
- ↑ Novelty weight (0.2-0.3)

**For smaller models:**
- ↑ Complexity penalty (5e-6 to 1e-5)

### Q: Can I visualize evolution progress in real-time?

**A:** Yes, with callbacks:

```python
import matplotlib.pyplot as plt
from IPython import display

def plot_progress(generation, fitness_history):
    plt.clf()
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Generation {generation}')
    display.clear_output(wait=True)
    display.display(plt.gcf())

# In evolution loop
for generation in range(generations):
    # ... evolution code ...
    plot_progress(generation, fitness_history)
```

### Q: How do I contribute or extend NoTransformers?

**A:** 

1. Fork the repository
2. Create feature branch
3. Add functionality (new primitives, grammar rules, etc.)
4. Test thoroughly
5. Submit pull request

**Common extensions:**
- New computational primitives
- Different selection mechanisms
- Custom fitness functions
- Alternative grammar structures

## Getting More Help

If your issue isn't covered:

1. **Check the docs:**
   - [Getting Started](getting-started.md)
   - [Architecture Guide](architecture.md)
   - [API Reference](api-reference.md)

2. **Search GitHub issues:**
   - Existing solutions
   - Similar problems

3. **Open a new issue:**
   - Describe the problem
   - Include code snippet
   - Share error messages
   - Provide system info

4. **Community discussion:**
   - Open GitHub discussion
   - Share experiments and results

## Summary

Most issues can be resolved by:
- Checking installation and environment
- Starting with small populations and generations
- Monitoring diversity and fitness trends
- Adjusting hyperparameters based on behavior
- Using error messages to guide debugging

For more help, see other documentation or open a GitHub issue.
