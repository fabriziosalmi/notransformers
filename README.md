# NoTransformers: A Grammar-Guided Neuroevolution Framework / Un Framework per la Scoperta Automatica di Architetture Neurali

> Bilingual README (English 🇬🇧 / Italiano 🇮🇹). Scroll for your preferred language.

---
## English Overview 🇬🇧

### ❓ Core Question
Can we automatically discover powerful neural architectures for sequential data **without** relying on the Transformer blueprint? This project explores that question by evolving architectures using a **grammar-guided evolutionary system** instead of manually designing models.

### 🧠 Concept
Rather than evolving raw weight matrices or arbitrary graphs, each individual (genome) is a sequence of integers that—interpreted through a structured grammar—builds a valid neural architecture (sequence + residual topology + hybrid blocks). The evolutionary loop co-optimizes both:
- Model structure ("hardware").
- Learning strategy: optimizer, learning rate, scheduler, activation ("software").

### ✨ Key Features
- **Grammar-Based Neural Architecture Search:** Safe architectural space guided by production rules.
- **Modern Primitive Library:** `Conv1D`, `GRU`, `LayerNorm`, `GELU`, `ReLU`, residual markers, identity, and hybrid stacking.
- **Co-Evolution (Competitive):** Two populations: Solvers (models) vs Saboteurs (adversarial sequence generators) in an arms race.
- **Rich Fitness Metrics:** Per-generation stats (mean/median/std, entropy, novelty, parameter and depth statistics) + coevolution pattern analytics.
- **Automatic Training:** Each genome is instantiated and trained with PyTorch (optionally GPU accelerated).
- **Top-K Unique Reporting & Early Stopping:** Diversity pressure + novelty blending.
- **JSON Metrics Export:** For downstream analysis / plotting.

### 🧪 Example Discovered Winner (Parity Task)
Final best architecture (fitness ≈ `0.9979`):
```
Conv1D -> ReLU -> LayerNorm -> identity
```
Despite exploring deeper hybrid graphs, evolution selected a minimal model—an illustration of **Occam's Razor** under complexity penalty.

### 🔁 Competitive Co-Evolution
The system optionally launches a second population of **SaboteurGenome** entities that generate structured adversarial binary sequences (patterns: alternating, repeating chunks, mostly zeros/ones, edge-active, etc.).
Solvers train briefly on random parity sequences then are evaluated on saboteur batches. Fitness becomes a win ratio; saboteurs maximize solver failure.

### 📊 Metrics (Excerpt)
For each generation (standard):
- `fitness_mean / median / std`
- `arch_entropy`, `unique_architectures`
- `param_mean / max`, `depth_mean / max`
- `novelty_mean / max`

For co-evolution:
- Solver: entropy, unique arches, param mean/std, depth mean.
- Saboteur: pattern type distribution, mean noise level, mean pattern length.

Export via:
```bash
python main.py --generations 30 --population 50 --metrics-json run_metrics.json
```

### 🚀 Quick Start
Install dependencies:
```bash
pip install torch numpy matplotlib
```
(Optional) System Graphviz for visualization:
```bash
# macOS
brew install graphviz
# Debian/Ubuntu
sudo apt-get install graphviz
```
Run standard evolution:
```bash
python main.py --population 50 --generations 40
```
Run competitive co-evolution:
```bash
python main.py --coevolution --population 40 --generations 20 --metrics-json coevo.json
```
Optional flags:
| Flag | Description |
|------|-------------|
| `--coevolution` | Enable solver vs saboteur arms race |
| `--population` | Population size |
| `--generations` | Number of generations |
| `--metrics-json PATH` | Export generation-level metrics |

### 🔍 Project Structure (Core)
```
evolutionary_discovery/
  evolution.py        # Evolution + co-evolution loops, metrics
  genome.py           # ModelGenome + PyTorch build logic
  saboteur.py         # SaboteurGenome adversarial pattern generator
  grammar.py          # Grammar + expansion utilities
  main.py             # CLI entry point
  evaluation_suite.py # (Optional) Extended benchmarking
```

### 🧩 Grammar Philosophy
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
  author       = {F.S.},
  year         = {2025},
  url          = {https://github.com/fabriziosalmi/notransformers}
}
```

---
Feedback or feature ideas? Open an issue. Happy evolving! 🧬
