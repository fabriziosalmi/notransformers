# API Reference

Complete API documentation for all modules, classes, and functions in NoTransformers.

## Table of Contents

- [genome Module](#genome-module)
- [grammar Module](#grammar-module)
- [evolution Module](#evolution-module)
- [saboteur Module](#saboteur-module)
- [primitives Module](#primitives-module)
- [evaluation_suite Module](#evaluation_suite-module)
- [visualize Module](#visualize-module)

## genome Module

### ModelGenome

Main class for representing neural architectures as evolvable genomes.

```python
class ModelGenome:
    def __init__(self, genes: List[int], learning_params: Dict[str, Any])
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `genes` | `List[int]` | Integer sequence guiding grammar expansion |
| `learning_params` | `Dict[str, Any]` | Training hyperparameters |
| `built_architecture` | `List[str]` or `None` | Cached terminal sequence |
| `model` | `nn.Module` or `None` | Cached PyTorch model |
| `input_dim` | `int` or `None` | Input dimension |
| `output_dim` | `int` or `None` | Output dimension |

#### Methods

##### `create_random_genome()`

```python
@staticmethod
def create_random_genome(input_dim: int, output_dim: int, 
                        min_genes: int = 5, max_genes: int = 20) -> ModelGenome
```

**Description:** Creates a random genome with random genes and learning parameters.

**Parameters:**
- `input_dim` (int): Input dimension for the model
- `output_dim` (int): Output dimension for the model  
- `min_genes` (int): Minimum number of genes
- `max_genes` (int): Maximum number of genes

**Returns:** New `ModelGenome` instance

**Example:**
```python
genome = ModelGenome.create_random_genome(input_dim=1, output_dim=1)
```

##### `build_from_grammar()`

```python
def build_from_grammar(self, grammar=None, max_expansions: int = 50) -> List[str]
```

**Description:** Expands genes into terminal sequence using grammar rules.

**Parameters:**
- `grammar` (dict, optional): Grammar dictionary (uses default if None)
- `max_expansions` (int): Maximum expansion steps

**Returns:** List of terminal symbols representing architecture

**Example:**
```python
architecture = genome.build_from_grammar()
# Returns: ['Conv1D', 'ReLU', 'LayerNorm', 'GRU', 'Tanh', 'LayerNorm']
```

##### `build_pytorch_model()`

```python
def build_pytorch_model(self, input_dim: int, output_dim: int) -> nn.Module
```

**Description:** Builds PyTorch model from grammatical architecture.

**Parameters:**
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension

**Returns:** PyTorch `nn.Module`

**Example:**
```python
model = genome.build_pytorch_model(input_dim=1, output_dim=1)
```

##### `mutate()`

```python
def mutate(self, mutation_rate: float)
```

**Description:** Applies mutations to genes and learning parameters.

**Parameters:**
- `mutation_rate` (float): Probability of mutation per gene

**Example:**
```python
genome.mutate(mutation_rate=0.3)
```

##### `crossover()`

```python
def crossover(self, other: ModelGenome) -> Tuple[ModelGenome, ModelGenome]
```

**Description:** Performs single-point crossover with another genome.

**Parameters:**
- `other` (ModelGenome): Parent genome to crossover with

**Returns:** Tuple of two child genomes

**Example:**
```python
child1, child2 = parent1.crossover(parent2)
```

##### `clone()`

```python
def clone(self) -> ModelGenome
```

**Description:** Creates a deep copy of the genome.

**Returns:** New `ModelGenome` with copied genes and parameters

##### `get_architecture_string()`

```python
def get_architecture_string(self) -> str
```

**Description:** Returns string representation of architecture.

**Returns:** Architecture as string (e.g., "Conv1D-ReLU-LayerNorm")

##### `count_parameters()`

```python
def count_parameters(self) -> int
```

**Description:** Counts total parameters in the model.

**Returns:** Number of trainable parameters

### Wrapper Classes

#### Conv1DWrapper

```python
class Conv1DWrapper(nn.Module):
    def __init__(self, conv: nn.Conv1d)
```

**Description:** Wrapper for Conv1D that handles shape transposition.

**Input:** `(batch, seq_len, channels)`  
**Output:** `(batch, seq_len, out_channels)`

#### GRUWrapper

```python
class GRUWrapper(nn.Module):
    def __init__(self, gru: nn.GRU)
```

**Description:** Wrapper for GRU with sequence-to-vector aggregation.

**Input:** `(batch, seq_len, input_dim)`  
**Output:** `(batch, hidden_size)`

#### SequenceToVector

```python
class SequenceToVector(nn.Module):
    def __init__(self)
```

**Description:** Aggregates sequences to fixed-size vectors via mean pooling.

**Input:** `(batch, seq_len, features)`  
**Output:** `(batch, features)`

## grammar Module

### GRAMMAR

```python
GRAMMAR: Dict[str, List[List[str]]]
```

**Description:** Dictionary defining production rules for architecture generation.

**Structure:**
```python
{
    "<non_terminal>": [
        ["expansion", "rule", "1"],
        ["expansion", "rule", "2"]
    ]
}
```

### Functions

#### `expand_grammar()`

```python
def expand_grammar(genes: List[int], grammar: Dict, 
                  max_expansions: int = 50) -> List[str]
```

**Description:** Expands grammar using gene sequence.

**Parameters:**
- `genes` (List[int]): Gene sequence
- `grammar` (Dict): Grammar rules
- `max_expansions` (int): Maximum expansions

**Returns:** List of terminal symbols

**Example:**
```python
from grammar import GRAMMAR, expand_grammar

genes = [3, 0, 1, 2, 0]
architecture = expand_grammar(genes, GRAMMAR)
```

#### `is_terminal()`

```python
def is_terminal(symbol: str) -> bool
```

**Description:** Checks if a symbol is terminal.

**Parameters:**
- `symbol` (str): Grammar symbol

**Returns:** True if terminal, False if non-terminal

#### `print_grammar_info()`

```python
def print_grammar_info()
```

**Description:** Prints grammar structure and available terminals.

## evolution Module

### EvolutionarySearch

Main class implementing evolutionary algorithm.

```python
class EvolutionarySearch:
    def __init__(self, input_dim: int, output_dim: int,
                 population_size: int = 100,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 num_eval_runs: int = 1,
                 complexity_penalty_coef: float = 1e-6,
                 diversity_pressure: float = 0.2,
                 novelty_weight: float = 0.15,
                 early_stop_fitness: float = 0.999,
                 early_stop_patience: int = 2,
                 top_k_report: int = 3)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Input dimension |
| `output_dim` | int | - | Output dimension |
| `population_size` | int | 100 | Number of genomes per generation |
| `mutation_rate` | float | 0.3 | Mutation probability |
| `crossover_rate` | float | 0.7 | Crossover probability |
| `tournament_size` | int | 3 | Tournament selection size |
| `num_eval_runs` | int | 1 | Evaluations per genome |
| `complexity_penalty_coef` | float | 1e-6 | Parameter count penalty |
| `diversity_pressure` | float | 0.2 | Duplicate architecture penalty |
| `novelty_weight` | float | 0.15 | Novelty vs fitness weight |
| `early_stop_fitness` | float | 0.999 | Early stopping threshold |
| `early_stop_patience` | int | 2 | Generations before stopping |
| `top_k_report` | int | 3 | Top architectures to report |

#### Methods

##### `run()`

```python
def run(self, generations: int, sequence_length: int, 
        num_samples: int) -> Tuple[ModelGenome, List[float], List[float]]
```

**Description:** Runs standard evolution.

**Parameters:**
- `generations` (int): Number of generations
- `sequence_length` (int): Sequence length for tasks
- `num_samples` (int): Number of training samples

**Returns:** Tuple of (best_genome, fitness_history, best_fitness_history)

**Example:**
```python
search = EvolutionarySearch(input_dim=1, output_dim=1)
best, fitness_hist, best_hist = search.run(
    generations=50, 
    sequence_length=8, 
    num_samples=500
)
```

##### `run_coevolution()`

```python
def run_coevolution(self, generations: int) -> Tuple[List[ModelGenome], 
                                                       List[SaboteurGenome],
                                                       Tuple]
```

**Description:** Runs competitive co-evolution.

**Parameters:**
- `generations` (int): Number of generations

**Returns:** Tuple of (solvers, saboteurs, histories)

**Example:**
```python
solvers, saboteurs, (s_avg, s_best, sab_avg, sab_best) = search.run_coevolution(30)
```

##### `export_metrics()`

```python
def export_metrics(self) -> Dict
```

**Description:** Exports collected metrics as dictionary.

**Returns:** Dictionary with per-generation metrics

### Functions

#### `train_and_evaluate_genome()`

```python
def train_and_evaluate_genome(genome: ModelGenome, 
                              X_data: torch.Tensor,
                              y_data: torch.Tensor,
                              learning_params: Dict,
                              epochs: int = 15,
                              batch_size: int = 32,
                              random_seed: int = None) -> float
```

**Description:** Trains and evaluates a genome on data.

**Parameters:**
- `genome` (ModelGenome): Genome to evaluate
- `X_data` (Tensor): Input data
- `y_data` (Tensor): Target labels
- `learning_params` (Dict): Learning configuration
- `epochs` (int): Training epochs
- `batch_size` (int): Batch size
- `random_seed` (int, optional): Random seed

**Returns:** Fitness score (0-1)

#### `plot_evolution()`

```python
def plot_evolution(fitness_history: List[float],
                  best_fitness_history: List[float],
                  solver_histories: Tuple = None)
```

**Description:** Plots evolution progress.

**Parameters:**
- `fitness_history` (List[float]): Mean fitness per generation
- `best_fitness_history` (List[float]): Best fitness per generation
- `solver_histories` (Tuple, optional): Co-evolution histories

## saboteur Module

### SaboteurGenome

```python
class SaboteurGenome:
    def __init__(self, params: Dict[str, Any])
```

**Description:** Genome for generating adversarial patterns.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | Dict | Saboteur parameters |

**Parameter Keys:**
- `pattern_length` (int): Length of pattern unit
- `pattern_type` (str): Type of adversarial strategy
- `noise_level` (float): Amount of noise to add
- `density` (float): For sparse patterns

#### Methods

##### `random_genome()`

```python
@staticmethod
def random_genome() -> SaboteurGenome
```

**Description:** Creates random saboteur genome.

##### `generate_sequence()`

```python
def generate_sequence(self, seq_len: int) -> torch.Tensor
```

**Description:** Generates single adversarial sequence.

**Parameters:**
- `seq_len` (int): Sequence length

**Returns:** Tensor of shape `(seq_len, 1)`

##### `generate_batch()`

```python
def generate_batch(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]
```

**Description:** Generates batch of sequences with labels.

**Parameters:**
- `batch_size` (int): Number of sequences
- `seq_len` (int): Sequence length

**Returns:** Tuple of (X, y) tensors

##### `mutate()`

```python
def mutate(self, rate: float = 0.2)
```

**Description:** Mutates saboteur parameters.

##### `crossover()`

```python
def crossover(self, other: SaboteurGenome) -> SaboteurGenome
```

**Description:** Performs crossover with another saboteur.

## primitives Module

### ComputationalPrimitive

Base class for all primitives.

```python
class ComputationalPrimitive(nn.Module):
    def forward(self, x) -> torch.Tensor
    def reset(self)
```

### Primitive Classes

#### InputNode

```python
class InputNode(ComputationalPrimitive):
    def __init__(self, input_dim: int)
```

#### Linear

```python
class Linear(ComputationalPrimitive):
    def __init__(self, input_dim: int, output_dim: int)
```

#### Conv1D

```python
class Conv1D(ComputationalPrimitive):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3)
```

#### GRU

```python
class GRU(ComputationalPrimitive):
    def __init__(self, input_dim: int, hidden_size: int)
```

## evaluation_suite Module

### Functions

#### `train_and_test_model()`

```python
def train_and_test_model(model: nn.Module,
                         train_loader: DataLoader,
                         test_loader: DataLoader,
                         task_type: str,
                         epochs: int = 20) -> float
```

**Description:** Trains and evaluates a model on train/test split.

**Parameters:**
- `model` (nn.Module): PyTorch model
- `train_loader` (DataLoader): Training data
- `test_loader` (DataLoader): Test data
- `task_type` (str): 'classification' or 'regression'
- `epochs` (int): Training epochs

**Returns:** Final metric (accuracy or MSE)

## visualize Module

### Functions

#### `visualize_genome()`

```python
def visualize_genome(genome: ModelGenome, 
                     filename: str = "best_genome.png")
```

**Description:** Creates Graphviz visualization of architecture.

**Parameters:**
- `genome` (ModelGenome): Genome to visualize
- `filename` (str): Output filename

**Example:**
```python
from visualize import visualize_genome

best_genome = ...
visualize_genome(best_genome, "architecture.png")
```

## Usage Examples

### Basic Evolution

```python
from evolution import EvolutionarySearch

# Create search instance
search = EvolutionarySearch(
    input_dim=1,
    output_dim=1,
    population_size=50,
    mutation_rate=0.3
)

# Run evolution
best_genome, fitness_hist, best_hist = search.run(
    generations=40,
    sequence_length=8,
    num_samples=500
)

# Get architecture
architecture = best_genome.get_architecture_string()
print(f"Best architecture: {architecture}")
```

### Custom Genome Creation

```python
from genome import ModelGenome

# Manual genome
genes = [3, 0, 1, 2, 0, 1]
learning_params = {
    'optimizer': 'adam',
    'learning_rate': 0.005,
    'lr_scheduler': 'cosine'
}
genome = ModelGenome(genes, learning_params)

# Build model
model = genome.build_pytorch_model(input_dim=1, output_dim=1)
```

### Co-Evolution

```python
search = EvolutionarySearch(input_dim=1, output_dim=1)
solvers, saboteurs, histories = search.run_coevolution(generations=30)

# Best solver
best_solver_idx = np.argmax(histories[1])  # Best fitness history
best_solver = solvers[best_solver_idx]
```

## Summary

This API reference covers all major components of NoTransformers. For:
- **Practical guides**: See [Getting Started](getting-started.md)
- **Conceptual understanding**: See [Architecture](architecture.md) and [Grammar](grammar.md)
- **Examples**: See [Examples and Tutorials](examples.md)
