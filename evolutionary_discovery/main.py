from evolution import EvolutionarySearch, plot_evolution
from grammar import GRAMMAR, print_grammar_info
import argparse
import numpy as np
import json


# Configurazione per Neuroevoluzione Grammaticale
POPULATION_SIZE = 100  # Piccola popolazione per test
GENERATIONS = 100  # Poche generazioni per test
MUTATION_RATE = 0.3  # Mutazione più alta per esplorare architetture diverse
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3  # Torneo più piccolo
NUM_EVAL_RUNS = 1  # Singola valutazione per debug
INPUT_DIM = 1
OUTPUT_DIM = 1
SEQUENCE_LENGTH = 8
NUM_SAMPLES = 500  # Dati ridotti per test più veloce
# Nuovi parametri evolutivi avanzati
COMPLEXITY_PENALTY = 1e-6
DIVERSITY_PRESSURE = 0.2  # Penalizza duplicati
NOVELTY_WEIGHT = 0.15     # Mescola fitness e novità
EARLY_STOP_FITNESS = 0.999  # Soglia di early stopping
EARLY_STOP_PATIENCE = 2     # Generazioni consecutive richieste
TOP_K_REPORT = 3            # Report delle migliori architetture uniche


def parse_args():
    parser = argparse.ArgumentParser(description="Grammar-based Neuroevolution with optional Competitive Co-Evolution")
    parser.add_argument('--coevolution', action='store_true', help='Abilita modalità competizione Solvers vs Saboteurs')
    parser.add_argument('--generations', type=int, default=GENERATIONS)
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--metrics-json', type=str, default=None, help='Percorso file JSON per esportare le metriche')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("🧬 NEUROEVOLUZIONE GRAMMATICALE 🧬")
    print("=" * 50)
    mode = 'CO-EVOLUTION' if args.coevolution else 'STANDARD'
    print(f"Modalità: {mode}")
    print(f"Popolazione: {args.population}, Generazioni: {args.generations}")
    print(f"Valutazioni per genoma: {NUM_EVAL_RUNS}, Torneo: {TOURNAMENT_SIZE}")
    print(f"Mutation Rate: {MUTATION_RATE}, Crossover Rate: {CROSSOVER_RATE}")
    print()

    # Mostra info sulla grammatica
    print_grammar_info()
    print()

    search = EvolutionarySearch(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        population_size=args.population,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        num_eval_runs=NUM_EVAL_RUNS,
        complexity_penalty_coef=COMPLEXITY_PENALTY,
        diversity_pressure=DIVERSITY_PRESSURE,
        novelty_weight=NOVELTY_WEIGHT,
        early_stop_fitness=EARLY_STOP_FITNESS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        top_k_report=TOP_K_REPORT
    )

    if args.coevolution:
        solvers, saboteurs, histories = search.run_coevolution(args.generations)
        solver_avg, solver_best, sab_avg, sab_best = histories
        print("\n== CO-EVOLUTION TERMINATA ==")
        print(f"Miglior solver (fitness={max(solver_best):.4f}) arch: {solvers[int(np.argmax(solver_best))].get_architecture_string() if solvers else 'N/A'}")
        print(f"Miglior saboteur (score={max(sab_best):.4f}) pattern: {saboteurs[int(np.argmax(sab_best))].params if saboteurs else 'N/A'}")
        if args.metrics_json:
            with open(args.metrics_json, 'w') as f:
                json.dump(search.export_metrics(), f, indent=2)
            print(f"Metriche esportate in {args.metrics_json}")
        plot_evolution([], [], solver_histories=histories)
    else:
        best_genome, fitness_history, best_fitness_history = search.run(
            args.generations,
            sequence_length=SEQUENCE_LENGTH,
            num_samples=NUM_SAMPLES
        )

        print("\n" + "=" * 60)
        print("🏆 RISULTATI FINALI 🏆")
        print("=" * 60)
        print(f"Miglior genoma: {best_genome}")
        print(f"\nArchitettura scoperta:")
        print(f"  {best_genome.get_architecture_string()}")
        print(f"\nParametri di apprendimento:")
        for param, value in best_genome.learning_params.items():
            print(f"  {param}: {value}")
        print(f"\nGeni che codificano l'architettura:")
        print(f"  {best_genome.genes}")
        print(f"\nNumero totale di geni: {len(best_genome.genes)}")

        # Mostra il modello PyTorch finale
        print(f"\nModello PyTorch generato:")
        final_model = best_genome.build_pytorch_model(INPUT_DIM, OUTPUT_DIM)
        print(final_model)

        print("\n" + "=" * 60)

        # Plot evoluzione
        plot_evolution(fitness_history, best_fitness_history)

        if args.metrics_json:
            with open(args.metrics_json, 'w') as f:
                json.dump(search.export_metrics(), f, indent=2)
            print(f"Metriche esportate in {args.metrics_json}")
