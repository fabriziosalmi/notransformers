import random
import numpy as np
import torch
from typing import Dict, Any, List

PATTERN_TYPES = [
    'alternating',      # 1,0,1,0,...
    'repeating_chunk',  # e.g., 1,1,0, 1,1,0, ...
    'mostly_zeros',     # mostly 0 with sparse 1s
    'mostly_ones',      # mostly 1 with sparse 0s
    'edge_ones',        # ones at edges, zeros inside
]

REPEATING_CHUNKS = [
    [1,1,0],
    [1,0,0],
    [1,0,1,0],
    [0,1,1],
]

class SaboteurGenome:
    """Genome that generates adversarial parity sequences."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @staticmethod
    def random_genome():
        return SaboteurGenome({
            'pattern_length': random.randint(2, 8),
            'pattern_type': random.choice(PATTERN_TYPES),
            'noise_level': random.uniform(0.0, 0.25),
            'density': random.uniform(0.05, 0.5),  # for sparse patterns
        })

    def clone(self):
        return SaboteurGenome(self.params.copy())

    def mutate(self, rate=0.2):
        if random.random() < rate:
            self.params['pattern_length'] = max(2, min(32, self.params['pattern_length'] + random.randint(-2, 2)))
        if random.random() < rate:
            self.params['pattern_type'] = random.choice(PATTERN_TYPES)
        if random.random() < rate:
            self.params['noise_level'] = min(0.5, max(0.0, self.params['noise_level'] + random.uniform(-0.05, 0.05)))
        if random.random() < rate:
            self.params['density'] = min(0.9, max(0.01, self.params['density'] + random.uniform(-0.1, 0.1)))

    def crossover(self, other:'SaboteurGenome'):
        child_params = {}
        for k in self.params:
            child_params[k] = random.choice([self.params[k], other.params.get(k, self.params[k])])
        return SaboteurGenome(child_params)

    def generate_sequence(self, seq_len:int):
        t = self.params['pattern_type']
        L = self.params['pattern_length']
        density = self.params['density']
        if t == 'alternating':
            seq = [(i % 2) for i in range(seq_len)]
        elif t == 'repeating_chunk':
            chunk = random.choice(REPEATING_CHUNKS)
            seq = [chunk[i % len(chunk)] for i in range(seq_len)]
        elif t == 'mostly_zeros':
            seq = [1 if random.random() < density else 0 for _ in range(seq_len)]
        elif t == 'mostly_ones':
            seq = [0 if random.random() < density else 1 for _ in range(seq_len)]
        elif t == 'edge_ones':
            seq = [1 if (i < L or i >= seq_len - L) else 0 for i in range(seq_len)]
        else:
            seq = [random.randint(0,1) for _ in range(seq_len)]

        # Introduce pattern length manipulation (flip subsections)
        if L < seq_len and random.random() < 0.5:
            start = random.randint(0, seq_len - L)
            for i in range(start, start + L):
                seq[i] = 1 - seq[i]

        # Apply noise
        noise_level = self.params['noise_level']
        if noise_level > 0:
            for i in range(seq_len):
                if random.random() < noise_level:
                    seq[i] = 1 - seq[i]

        return torch.tensor(seq, dtype=torch.float32).view(seq_len, 1)

    def generate_batch(self, batch_size:int, seq_len:int):
        X = torch.stack([self.generate_sequence(seq_len) for _ in range(batch_size)])  # (batch, seq, 1)
        # Parity labels
        y = (X.sum(dim=1) % 2).float()
        return X, y

    def __repr__(self):
        return f"SaboteurGenome({self.params})"
