# NoTransformers: Grammar-Guided Neuroevolution Framework
## Un Framework Avanzato per la Scoperta Automatica di Architetture Neurali

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bilingual Documentation** (English 🇬🇧 / Italiano 🇮🇹) | Jump to: [English](#english) | [Italiano](#italiano)

---

<a name="english"></a>
## 🇬🇧 English Documentation

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
A controlled search space reduces invalid architectures and encodes inductive bias toward modular, residual-friendly sequences.

### 🧭 Future Roadmap
- Multi-objective Pareto fronts (accuracy vs params vs latency)
- Hall-of-Fame archival + lineage tracking
- Larger datasets (IMDB / WikiText / synthetic memory tasks)
- Attention-like or gating primitives integration
- Parallel fitness via multiprocessing / distributed
- Visualization dashboard (Streamlit or Lite web UI)

### 🤝 Contributing
Ideas, issues, and PRs welcome. Open a discussion for grammar design extensions or new primitive suggestions.

---
## Panoramica Italiana 🇮🇹

### ❓ Domanda Centrale
Possiamo scoprire automaticamente architetture neurali efficaci per dati sequenziali **senza** affidarci al design del Transformer? Questo progetto esplora la risposta tramite un sistema evolutivo guidato da una **grammatica strutturata**.

### 🧠 Concetto
Invece di evolvere connessioni grezze o grafi arbitrari, ogni individuo è una sequenza di interi che—interpretata dalla grammatica—compone un'architettura valida. L'evoluzione co-ottimizza:
- Struttura del modello.
- Strategia di apprendimento (ottimizzatore, learning rate, scheduler, attivazione).

### ✨ Caratteristiche Chiave
- **Ricerca Architetturale Grammaticale:** Spazio architetturale sicuro guidato da regole di produzione.
- **Libreria di Primitivi Moderna:** `Conv1D`, `GRU`, `LayerNorm`, `GELU`, `ReLU`, marcatori di residuo, identity e impilamento ibrido.
- **Co-evoluzione (Competitiva):** Due popolazioni: Solutori (modelli) vs Saboteurs (generatori di sequenze avversarie) in una corsa agli armamenti.
- **Metriche di Fitness Ricche:** Statistiche per generazione (media/mediana/std, entropia, novità, parametri e statistiche sulla profondità) + analisi dei pattern di coevoluzione.
- **Addestramento Automatico:** Ogni genoma è istanziato e addestrato con PyTorch (opzionalmente accelerato da GPU).
- **Reporting Top-K Unico & Early Stopping:** Pressione sulla diversità + miscelazione della novità.
- **Esportazione JSON delle Metriche:** Per analisi / grafico downstream.

### 🧪 Esempio di Vincitore Scoperto (Compito Parità)
Migliore architettura finale (fitness ≈ `0.9979`):
```
Conv1D -> ReLU -> LayerNorm -> identity
```
Nonostante l'esplorazione di grafi ibridi più profondi, l'evoluzione ha selezionato un modello minimale, un'illustrazione del **rasoio di Occam** sotto penalità di complessità.

### 🔁 Co-Evoluzione Competitiva
Il sistema lancia opzionalmente una seconda popolazione di entità **SaboteurGenome** che generano sequenze binarie avversarie strutturate (pattern: alternativi, chunk ripetuti, per lo più zeri/uno, attivi ai margini, ecc.).
I solutori vengono addestrati brevemente su sequenze di parità casuali e poi valutati su lotti di saboteur. La fitness diventa un rapporto di vittoria; i saboteurs massimizzano il fallimento del solver.

### 📊 Metriche (Estratto)
Per ogni generazione (standard):
- `fitness_mean / median / std`
- `arch_entropy`, `unique_architectures`
- `param_mean / max`, `depth_mean / max`
- `novelty_mean / max`

Per la co-evoluzione:
- Solver: entropia, architetture uniche, parametri media/std, profondità media.
- Saboteur: distribuzione del tipo di pattern, livello medio di rumore, lunghezza media del pattern.

Esporta tramite:
```bash
python main.py --generations 30 --population 50 --metrics-json run_metrics.json
```

### 🚀 Avvio Rapido
Installa le dipendenze:
```bash
pip install torch numpy matplotlib
```
(Opzionale) Sistema Graphviz per la visualizzazione:
```bash
# macOS
brew install graphviz
# Debian/Ubuntu
sudo apt-get install graphviz
```
Esegui l'evoluzione standard:
```bash
python main.py --population 50 --generations 40
```
Esegui la co-evoluzione competitiva:
```bash
python main.py --coevolution --population 40 --generations 20 --metrics-json coevo.json
```
Flag opzionali:
| Flag | Descrizione |
|------|-------------|
| `--coevolution` | Abilita la corsa agli armamenti solver vs saboteur |
| `--population` | Dimensione della popolazione |
| `--generations` | Numero di generazioni |
| `--metrics-json PATH` | Esporta metriche a livello di generazione |

### 🔍 Struttura del Progetto (Core)
```
evolutionary_discovery/
  evolution.py        # Cicli di evoluzione + co-evoluzione, metriche
  genome.py           # ModelGenome + logica di costruzione PyTorch
  saboteur.py         # Generatore di pattern avversari SaboteurGenome
  grammar.py          # Grammatica + utilità di espansione
  main.py             # Punto di ingresso CLI
  evaluation_suite.py # (Opzionale) Benchmarking esteso
```

### 🧩 Filosofia della Grammatica
Uno spazio di ricerca controllato riduce le architetture non valide e codifica un bias induttivo verso sequenze modulari e favorevoli ai residui.

### 🧭 Roadmap Futura
- Fronti Pareto multi-obiettivo (accuratezza vs parametri vs latenza)
- Archiviazione Hall-of-Fame + tracciamento della discendenza
- Dataset più grandi (IMDB / WikiText / compiti di memoria sintetica)
- Integrazione di primitivi simili all'attenzione o di gating
- Fitness parallela tramite multiprocessing / distribuito
- Dashboard di visualizzazione (Streamlit o Lite web UI)

### 🤝 Contributi
Idee, problemi e PR sono i benvenuti. Apri una discussione per estensioni del design grammaticale o suggerimenti per nuovi primitivi.

---
## License
MIT (add a `LICENSE` file if distributing).

## Citation (Optional)
If you use this framework in research, you can cite it as:
```
@software{notransformers2025,
  title        = {NoTransformers: Grammar-Guided Neuroevolution for Sequence Models},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/your/repo}
}
```

---
Feedback or feature ideas? Open an issue. Happy evolving! 🧬