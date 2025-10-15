# NoTransformers: Grammar-Guided Neuroevolution Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Automatically discover powerful neural architectures without transformers through grammar-guided evolutionary algorithms.

---

## 📚 Documentation

**New to NoTransformers?** Start here:
- **[Getting Started Guide](docs/getting-started.md)** - Installation, quick start, and basic usage
- **[Examples & Tutorials](docs/examples.md)** - Practical examples and step-by-step tutorials

**Understanding the System:**
- **[Architecture Guide](docs/architecture.md)** - System design and core components
- **[Grammar System](docs/grammar.md)** - How grammars define neural architectures
- **[Evolutionary Algorithms](docs/evolution.md)** - Genetic operators and search mechanisms
- **[Co-Evolution](docs/coevolution.md)** - Competitive evolution with adversarial patterns

**Reference:**
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and FAQ

---

## 🎯 Project Vision

**Can we automatically discover powerful neural architectures without relying on Transformers?**

NoTransformers answers this question through **grammar-guided neuroevolution**: instead of manually designing models or fine-tuning pre-trained Transformers, we evolve entire architectures from scratch using evolutionary algorithms constrained by formal grammars.

---

### 🔬 Technical Approach

NoTransformers combines multiple advanced techniques:

- **Grammar-Guided Search**: Context-free grammars ensure valid architectures
- **Genetic Algorithms**: Population-based optimization for discrete structures
- **Multi-Objective Optimization**: Balance performance, complexity, and diversity
- **Competitive Co-Evolution**: Adversarial robustness through evolutionary arms races

**Learn more:**
- [Grammar System Documentation](docs/grammar.md)
- [Evolutionary Algorithms Documentation](docs/evolution.md)
- [Architecture Guide](docs/architecture.md)

---

### ✨ Key Features

#### 🧬 Grammar-Based Architecture Search
- **Valid architectures only**: Production rules guarantee well-formed models
- **Hierarchical composition**: Build complex structures from simple blocks
- **Automatic shape handling**: Dynamic adapters prevent dimension mismatches
- **Extensible grammar**: Easy to add new operations and patterns

**Details:** [Grammar System Documentation](docs/grammar.md)

#### 🛠️ Rich Primitive Library
```python
# Computational Primitives
Conv1D / Conv1DWrapper  # Sequence convolution with shape handling
GRU / GRUWrapper        # Recurrent processing with aggregation
Linear / Dense          # Fully connected layers
LayerNorm / BatchNorm   # Normalization primitives
ReLU / GELU / Tanh      # Activation functions

# Structural Primitives
residual                # Skip connections with adapters
identity               # Passthrough operations
SequenceToVector       # Sequence aggregation
```

**API Reference:** [Primitives Module](docs/api-reference.md#primitives-module)

#### 🥊 Competitive Co-Evolution
Two populations evolve simultaneously in an adversarial arms race:

1. **Solvers** (ModelGenome): Neural architectures solving tasks
2. **Saboteurs** (SaboteurGenome): Adversarial pattern generators

**Saboteur Strategies:**
- `alternating`: Simple 1,0,1,0,... patterns
- `repeating_chunk`: Repeating subsequences
- `mostly_zeros`: Sparse positive examples
- `mostly_ones`: Sparse negative examples
- `edge_ones`: Boundary-focused patterns

**Learn more:** [Co-Evolution Guide](docs/coevolution.md)

#### 📊 Comprehensive Metrics & Analytics

Track evolution progress with detailed metrics:

**Standard Evolution:**
- Fitness statistics (mean, median, best, std)
- Architecture diversity (entropy, unique count)
- Model complexity (parameters, depth)
- Novelty scores (Levenshtein distance)

**Co-Evolution:**
- Solver vs Saboteur fitness
- Pattern type distribution
- Arms race dynamics
- Population diversity

**Example:** [Metrics Analysis](docs/examples.md#example-4-export-and-analyze-metrics)

#### 🎛️ Advanced Evolutionary Mechanisms

1. **Novelty Search**: Levenshtein distance-based architecture diversity
2. **Diversity Pressure**: Penalizes duplicate architectures
3. **Early Stopping**: Configurable fitness threshold with patience
4. **Top-K Reporting**: Tracks best unique architectures
5. **Parameter Caching**: Efficient complexity computation
6. **Multi-Objective**: Balance performance, size, and novelty

**Deep dive:** [Evolutionary Algorithms Documentation](docs/evolution.md)

---

### 🚀 Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/fabriziosalmi/notransformers.git
cd notransformers

# Install dependencies
pip install torch numpy matplotlib

# Optional: Install Graphviz for visualization
pip install graphviz
```

For detailed installation instructions, see the **[Getting Started Guide](docs/getting-started.md)**.

#### Basic Usage

**Standard Evolution:**
```bash
cd evolutionary_discovery
python main.py --population 50 --generations 40
```

**Competitive Co-Evolution:**
```bash
python main.py --coevolution --population 40 --generations 20
```

**Export Metrics:**
```bash
python main.py --population 100 --generations 50 --metrics-json results.json
```

For more examples, see **[Examples & Tutorials](docs/examples.md)**.

---

### 📋 Project Structure

```
notransformers/
├── docs/                           # 📚 Documentation
│   ├── getting-started.md         # Installation and quick start
│   ├── architecture.md            # System architecture and design
│   ├── grammar.md                 # Grammar system deep dive
│   ├── evolution.md               # Evolutionary algorithms
│   ├── coevolution.md             # Competitive co-evolution
│   ├── api-reference.md           # Complete API documentation
│   ├── examples.md                # Examples and tutorials
│   └── troubleshooting.md         # Common issues and FAQ
│
├── evolutionary_discovery/         # 🧬 Core modules
│   ├── main.py                    # CLI entry point
│   ├── evolution.py               # Evolutionary algorithms
│   ├── genome.py                  # Neural architecture genome
│   ├── grammar.py                 # Context-free grammar
│   ├── saboteur.py                # Adversarial patterns
│   ├── primitives.py              # Computational primitives
│   ├── evaluation_suite.py        # Extended benchmarks
│   ├── visualize.py               # Architecture visualization
│   └── test_evolution_exact.py    # Unit tests
│
├── README.md                       # This file
├── LICENSE                         # MIT License
└── metrics_coevo.json             # Example output

```

**For detailed information on each module, see the [API Reference](docs/api-reference.md).**

---

### 🎓 Learning Path

**Beginners:**
1. Read [Getting Started](docs/getting-started.md)
2. Try the [Quick Start](#quick-start)
3. Explore [Examples & Tutorials](docs/examples.md)

**Intermediate:**
1. Understand [Architecture Guide](docs/architecture.md)
2. Study [Grammar System](docs/grammar.md)
3. Learn [Evolutionary Algorithms](docs/evolution.md)

**Advanced:**
1. Master [Co-Evolution](docs/coevolution.md)
2. Review [API Reference](docs/api-reference.md)
3. Extend the system with custom components

---

### 💡 Use Cases

**Research:**
- Neural architecture search (NAS)
- Evolutionary computation studies
- Adversarial robustness research
- Grammar-based generation systems

**Education:**
- Learning about evolutionary algorithms
- Understanding neural architecture design
- Exploring genetic programming
- Studying competitive co-evolution

**Development:**
- Automated model discovery
- Architecture optimization
- Hyperparameter search
- Robust model development

---

### 🤝 Contributing

We welcome contributions! Here's how you can help:

**Bug Reports & Feature Requests:**
- Open an issue on GitHub
- Provide detailed descriptions and examples
- Include error messages and system info

**Code Contributions:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

**Ideas for Contributions:**
- New computational primitives
- Alternative selection mechanisms
- Custom grammar structures
- Additional benchmark tasks
- Documentation improvements
- Performance optimizations

**Discussion:**
- Share experiments and results
- Propose new features
- Ask questions
- Help other users

For more details, see the [Contributing Guidelines](CONTRIBUTING.md) (if available) or open a discussion.

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/fabriziosalmi/notransformers/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fabriziosalmi/notransformers/discussions)
- **Documentation**: [docs/](docs/)

---

### 🌟 Acknowledgments

NoTransformers builds upon research in:
- Grammatical Evolution (O'Neill & Ryan, 2001)
- Neural Architecture Search (Zoph & Le, 2017)
- Competitive Co-Evolution (Hillis, 1990)
- Novelty Search (Lehman & Stanley, 2011)

---

### 📚 Citation

If you use NoTransformers in your research, please cite:

```bibtex
@software{notransformers2024,
  title = {NoTransformers: Grammar-Guided Neuroevolution Framework},
  author = {Salmi, Fabrizio},
  year = {2024},
  url = {https://github.com/fabriziosalmi/notransformers}
}
```

---

**Ready to start?** Head to the [Getting Started Guide](docs/getting-started.md) and begin evolving your first neural architectures!
