#!/usr/bin/env python3
"""Test esatto del setup usato dall'evoluzione."""

import torch
from evolution import EvolutionarySearch, train_and_evaluate_genome
from genome import ModelGenome
import numpy as np

# Setup identico a quello dell'evoluzione
evo_search = EvolutionarySearch(
    input_dim=1, 
    output_dim=1,
    population_size=10,
    num_eval_runs=1
)

# Genera dati esattamente come fa l'evoluzione
X_data, y_data = evo_search._generate_data(sequence_length=10, num_samples=500)

print(f"Dataset shape: X={X_data.shape}, y={y_data.shape}")
print(f"X range: [{X_data.min():.3f}, {X_data.max():.3f}]")
print(f"y range: [{y_data.min():.3f}, {y_data.max():.3f}]")
print(f"y unique values: {torch.unique(y_data)}")

# Crea un genoma random come fa l'evoluzione
genome = ModelGenome.create_random_genome(input_dim=1, output_dim=1)
print(f"\nGenoma generato: {genome.get_architecture_string()}")

# Test la valutazione
fitness = evo_search._evaluate_genome_fitness(genome, X_data, y_data)
print(f"Fitness ottenuta: {fitness}")

# Test un secondo genoma
genome2 = ModelGenome.create_random_genome(input_dim=1, output_dim=1)
print(f"\nGenoma 2: {genome2.get_architecture_string()}")
fitness2 = evo_search._evaluate_genome_fitness(genome2, X_data, y_data)
print(f"Fitness 2: {fitness2}")

# Test un genoma molto semplice
genome_simple = ModelGenome([8, 0], {})  # Solo Linear
print(f"\nGenoma semplice: {genome_simple.get_architecture_string()}")
fitness_simple = evo_search._evaluate_genome_fitness(genome_simple, X_data, y_data)
print(f"Fitness semplice: {fitness_simple}")