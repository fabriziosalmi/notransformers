# NoTransformers: Un Framework per la Scoperta Automatica di Architetture Neurali

Questo repository contiene il codice e i risultati di un progetto di ricerca sperimentale volto a rispondere a una domanda fondamentale: **È possibile scoprire automaticamente architetture neurali potenti per dati sequenziali, senza fare affidamento sul design predefinito del Transformer?**

Il progetto implementa un sistema di **Neuroevoluzione Grammaticale** in PyTorch, un algoritmo che evolve "ricette" per costruire reti neurali, invece di manipolare direttamente i loro pesi. Il framework è stato progettato da zero, partendo da semplici esperimenti in NumPy fino a un sistema di co-evoluzione competitiva, capace di forgiare architetture robuste in un ambiente avversario.

Questo non è solo un repository di codice, ma la cronaca di un viaggio scientifico attraverso le sfide e le scoperte dell'Automated Machine Learning (AutoML).

## 🏛️ Filosofia del Progetto

L'idea centrale è quella di passare da "architetti" di modelli a "biologi digitali". Invece di progettare manualmente una rete, abbiamo costruito un ecosistema in cui le architetture neurali competono, si riproducono e si adattano, con l'obiettivo di trovare soluzioni che non solo siano performanti, ma anche efficienti ed eleganti.

Il sistema si basa su tre pilastri:
1.  **Modularità:** Le architetture sono costruite combinando una libreria di primitivi di base (convoluzioni, unità ricorrenti, meccanismi di gating, etc.).
2.  **Evoluzione Strutturata:** Utilizziamo una **grammatica** per garantire che solo architetture sintatticamente corrette e modulari vengano generate, evitando il "caos computazionale".
3.  **Pressione Selettiva Realistica:** I modelli vengono valutati attraverso un **vero ciclo di addestramento in PyTorch**, e la loro robustezza viene testata in un ambiente di **co-evoluzione competitiva**.

## 🚀 Caratteristiche Principali

-   **Neuroevoluzione Grammaticale:** Il genoma di un modello non è un grafo complesso, ma una semplice lista di interi che viene interpretata da una grammatica definita per costruire il modello finale. Questo rende le operazioni di crossover e mutazione semplici ed efficaci.
-   **Libreria di Primitivi Ricca:** Include componenti moderni come `Conv1D`, `GRU`, `LayerNorm`, `GELU`, `GatedLinearUnit (GLU)` e connessioni residue.
-   **Co-evoluzione di Architettura e Ottimizzazione:** L'algoritmo non scopre solo la struttura della rete, ma anche la migliore strategia di apprendimento (learning rate, optimizer, scheduler) per essa.
-   **Modalità Co-Evolutiva Competitiva:** Una modalità avanzata in cui una popolazione di modelli ("Solutori") si evolve contro una popolazione di generatori di dati difficili ("Sabotatori"). Questo costringe i solutori a generalizzare invece di imparare a memoria.
-   **Banco di Prova Integrato (`evaluation_suite.py`):** Uno script per testare rigorosamente le architetture scoperte su una suite di compiti sequenziali standard (parità, copia, rilevamento di pattern).
-   **Visualizzazione Completa:** Generazione automatica di grafici sull'andamento dell'evoluzione e diagrammi delle architetture scoperte tramite Graphviz.

##  perjalanan Il Nostro Viaggio: Dalle Basi al Sistema Finale

Il progetto è stato sviluppato attraverso una serie di fasi iterative, ognuna delle quali ha affrontato una sfida specifica:
1.  **Fase NumPy:** Una prova di concetto per implementare la logica evolutiva di base.
2.  **Fase di Robustezza:** Introduzione di tecniche avanzate come elitismo e isole per stabilizzare l'evoluzione.
3.  **Fase PyTorch:** Il "grande salto" a un framework di deep learning per permettere un addestramento reale su GPU.
4.  **Fase Grammaticale:** Abbandono dei grafi liberi in favore di una grammatica strutturata per generare architetture più sensate.
5.  **Fase Competitiva:** L'introduzione della co-evoluzione per forgiare modelli robusti e capaci di generalizzare.

## 🛠️ Come Usare il Framework

### Prerequisiti

Assicurati di avere Python 3.8+ e le seguenti librerie installate:
```bash
pip install torch numpy matplotlib graphviz
```
Potrebbe essere necessario installare Graphviz a livello di sistema operativo (`brew install graphviz` su macOS, `sudo apt-get install graphviz` su Debian/Ubuntu).

### Eseguire l'Evoluzione Principale

Lo script principale è `main.py`. Puoi lanciarlo in due modalità.

**1. Modalità Standard:**
L'evoluzione cercherà di ottimizzare un'architettura per il compito di parità.
```bash
python main.py --population 50 --generations 100 --epochs 15
```

**2. Modalità Co-Evolutiva Competitiva:**
Lancia la "corsa agli armamenti" tra Solutori e Sabotatori.
```bash
python main.py --coevolution --population 20 --generations 50
```

**Argomenti Comuni:**
-   `--population`: Dimensione della popolazione (per ogni popolazione in co-evoluzione).
-   `--generations`: Numero di generazioni da eseguire.
-   `--epochs`: Numero di epoche di addestramento per la valutazione della fitness.
-   ... (aggiungi altri argomenti che hai reso disponibili).

L'esecuzione produrrà un log dettagliato, un grafico `evolution_plot.png` e un diagramma dell'architettura migliore `best_genome.png`.

### Eseguire il Banco di Prova

Per testare le architetture scoperte o le baseline su compiti multipli, usa `evaluation_suite.py`.
```bash
python evaluation_suite.py
```
Questo script eseguirà tutti i test definiti e stamperà un riassunto comparativo delle performance.

## 🔬 Risultati e Scoperte

Attraverso questo processo, il nostro sistema ha riscoperto autonomamente diversi principi di progettazione delle reti neurali moderne:
-   La necessità di **componenti con memoria** (come le GRU) per compiti sequenziali.
-   L'efficacia dei **blocchi modulari e residui** (`Operazione -> Attivazione -> Norm -> Residual`).
-   La potenza delle **architetture ibride** che combinano elementi ricorrenti e convoluzionali.
-   L'importanza di trovare la **sinergia tra architettura e strategia di apprendimento**.

Il risultato più importante è la dimostrazione che, con i giusti vincoli (la grammatica) e la giusta pressione selettiva (la competizione), un processo evolutivo può essere un potente motore per la scoperta di design neurali originali ed efficaci.

## 🤝 Contributi e Sviluppi Futuri

Questo progetto è una base di partenza. Ci sono innumerevoli direzioni future da esplorare:
-   **Scalare a Compiti Reali:** Adattare il framework per dataset di linguaggio naturale più grandi (es. classificazione di testo con IMDb).
-   **Arricchire la Grammatica:** Aggiungere nuovi primitivi (es. meccanismi di attenzione) e regole grammaticali più complesse per permettere la scoperta di architetture ancora più sofisticate.
-   **Ottimizzare le Prestazioni:** Parallelizzare il calcolo della fitness per eseguire esperimenti su una scala molto più ampia.

I contributi sono i benvenuti. Sentiti libero di aprire una issue per discutere di nuove idee o di inviare una pull request.
