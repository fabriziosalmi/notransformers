from evolution import EvolutionarySearch, plot_evolution
from grammar import GRAMMAR, print_grammar_info
import argparse
import numpy as np
import json


# Grammar-based Neuroevolution configuration
POPULATION_SIZE = 100  # Small population for quick tests
GENERATIONS = 100  # Few generations for quick tests
MUTATION_RATE = 0.3  # Higher mutation to explore diverse architectures
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3  # Smaller tournament
NUM_EVAL_RUNS = 1  # Single evaluation per genome (debug mode)
INPUT_DIM = 1
OUTPUT_DIM = 1
SEQUENCE_LENGTH = 8
NUM_SAMPLES = 500  # Reduced dataset for faster testing
# Advanced evolutionary parameters
COMPLEXITY_PENALTY = 1e-6
DIVERSITY_PRESSURE = 0.2  # Penalize duplicate architectures
NOVELTY_WEIGHT = 0.15     # Blend raw fitness and novelty
EARLY_STOP_FITNESS = 0.999  # Early stopping fitness threshold
EARLY_STOP_PATIENCE = 2     # Required consecutive generations above threshold
TOP_K_REPORT = 3            # Report top-K unique architectures each generation


def parse_args():
    parser = argparse.ArgumentParser(description="Grammar-based Neuroevolution with optional Competitive Co-Evolution")
    parser.add_argument('--coevolution', action='store_true', help='Enable competitive Solvers vs Saboteurs mode')
    parser.add_argument('--generations', type=int, default=GENERATIONS)
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--metrics-json', type=str, default=None, help='Path to JSON file to export metrics')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("🧬 GRAMMAR-BASED NEUROEVOLUTION 🧬")
    print("=" * 50)
    mode = 'CO-EVOLUTION' if args.coevolution else 'STANDARD'
    print(f"Mode: {mode}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Evaluations per genome: {NUM_EVAL_RUNS}, Tournament size: {TOURNAMENT_SIZE}")
    print(f"Mutation Rate: {MUTATION_RATE}, Crossover Rate: {CROSSOVER_RATE}")
    print()

    # Show grammar info
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
        print("\n== CO-EVOLUTION COMPLETED ==")
        print(f"Best solver (fitness={max(solver_best):.4f}) arch: {solvers[int(np.argmax(solver_best))].get_architecture_string() if solvers else 'N/A'}")
        print(f"Best saboteur (score={max(sab_best):.4f}) pattern: {saboteurs[int(np.argmax(sab_best))].params if saboteurs else 'N/A'}")
        if args.metrics_json:
            with open(args.metrics_json, 'w') as f:
                json.dump(search.export_metrics(), f, indent=2)
            print(f"Metrics exported to {args.metrics_json}")
        plot_evolution([], [], solver_histories=histories)
    else:
        best_genome, fitness_history, best_fitness_history = search.run(
            args.generations,
            sequence_length=SEQUENCE_LENGTH,
            num_samples=NUM_SAMPLES
        )

        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS 🏆")
        print("=" * 60)
        print(f"Best genome: {best_genome}")
        print(f"\nDiscovered architecture:")
        print(f"  {best_genome.get_architecture_string()}")
        print(f"\nLearning parameters:")
        for param, value in best_genome.learning_params.items():
            print(f"  {param}: {value}")
        print(f"\nGenes encoding the architecture:")
        print(f"  {best_genome.genes}")
        print(f"\nTotal number of genes: {len(best_genome.genes)}")

        # Show final PyTorch model
        print(f"\nGenerated PyTorch model:")
        final_model = best_genome.build_pytorch_model(INPUT_DIM, OUTPUT_DIM)
        print(final_model)

        print("\n" + "=" * 60)

        # Plot evoluzione
        plot_evolution(fitness_history, best_fitness_history)

        if args.metrics_json:
            with open(args.metrics_json, 'w') as f:
                json.dump(search.export_metrics(), f, indent=2)
            print(f"Metrics exported to {args.metrics_json}")
