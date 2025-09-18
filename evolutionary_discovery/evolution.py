import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from genome import ModelGenome
from saboteur import SaboteurGenome
import matplotlib.pyplot as plt

def train_and_evaluate_genome(genome, X_data, y_data, learning_params, epochs=15, batch_size=32, random_seed=None):
    """Train and evaluate a grammar-based genome.

    Args:
        genome: Grammar-based ModelGenome instance
        X_data, y_data: Training data tensors
        learning_params: Dict of learning configuration
        epochs: Number of training epochs
        batch_size: Mini-batch size
        random_seed: Optional seed for reproducibility
    """
    # Set a distinct seed for each evaluation
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build PyTorch model from grammar
    try:
        model = genome.build_pytorch_model(X_data.shape[-1], 1)
        model = model.to(device)
        model.train()
    except Exception as e:
        print(f"Error while building model: {e}")
        return 0.0
    
    # Check model has trainable parameters
    trainable_params = list(model.parameters())
    if len(trainable_params) == 0:
        # If no trainable parameters, return a small random baseline performance
        return 0.3 + 0.1 * (torch.rand(1).item() - 0.5)
    
    # Ottimizzatore
    if learning_params['optimizer'] == 'adam':
        optimizer = optim.Adam(trainable_params, lr=learning_params['learning_rate'])
    elif learning_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=learning_params['learning_rate'])
    else:
        optimizer = optim.RMSprop(trainable_params, lr=learning_params['learning_rate'])
    
    # Learning rate scheduler
    scheduler = None
    if learning_params.get('lr_scheduler', 'none') != 'none':
        if learning_params['lr_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        elif learning_params['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif learning_params['lr_scheduler'] == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    criterion = nn.BCELoss()
    
    # Training loop
    try:
        for epoch in range(epochs):
            for i in range(0, len(X_data), batch_size):
                batch_idx = slice(i, i + batch_size)
                xb = X_data[batch_idx].to(device)
                yb = y_data[batch_idx].to(device)
                
                # Ensure target dimensions are correct
                if yb.dim() > 1:
                    yb = yb.squeeze()
                    
                optimizer.zero_grad()
                out = model(xb)
                
                # Ensure model output shape is correct
                if out.dim() > 1:
                    out = out.squeeze()
                    
                loss = criterion(torch.sigmoid(out), yb)
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_eval = X_data.to(device)
            y_eval = y_data.to(device)
            out = model(X_eval)
            
            # Ensure correct shape for BCELoss
            if out.dim() > 1:
                out = out.squeeze()  # Remove extra dimensions
            if y_eval.dim() > 1:
                y_eval = y_eval.squeeze()  # Remove extra dimensions
                
            # Apply sigmoid and threshold for binary classification
            pred_prob = torch.sigmoid(out)
            pred = (pred_prob > 0.5).float()
            accuracy = (pred == y_eval).float().mean().item()
        
        return accuracy
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 0.0


class _MetricsExportMixin:
    def export_metrics(self):
        return {
            'standard': getattr(self, 'metrics_history', []),
            'coevolution': getattr(self, 'coevo_metrics_history', []),
        }

class EvolutionarySearch(_MetricsExportMixin):
    """Grammar-based evolutionary search algorithm.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output dimension (e.g. 1 for binary tasks).
        population_size (int): Number of genomes in the population.
        mutation_rate (float): Per-gene mutation probability.
        crossover_rate (float): Probability to apply crossover vs clone+mutate.
        tournament_size (int): Tournament size for selection.
        num_eval_runs (int): Repeated evaluations to average out stochastic noise.
        complexity_penalty_coef (float): Complexity penalty coefficient (num_params * coef).
        diversity_pressure (float): Duplicate architecture penalty factor (0 disables).
        novelty_weight (float): Blend weight for novelty (effective = (1-nw)*fitness + nw*novelty_norm).
        early_stop_fitness (float|None): Early stopping threshold.
        early_stop_patience (int): Required consecutive generations above threshold.
        top_k_report (int): Number of best unique architectures to print per generation.
    """
    def __init__(self, input_dim, output_dim, population_size=30, mutation_rate=0.2, crossover_rate=0.7, tournament_size=7, num_eval_runs=3, complexity_penalty_coef=1e-6, diversity_pressure=0.0, novelty_weight=0.0, early_stop_fitness=None, early_stop_patience=5, top_k_report=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.num_eval_runs = num_eval_runs  # Multiple evaluations to reduce noise
        # Coefficient to penalize overly large models (num_params * coef)
        self.complexity_penalty_coef = complexity_penalty_coef
        # Diversity: fraction penalty applied to repeated architectures (0 = disabled)
        self.diversity_pressure = diversity_pressure
        # Novelty: blending between raw fitness and normalized novelty score
        self.novelty_weight = novelty_weight
        # Early stopping
        self.early_stop_fitness = early_stop_fitness
        self.early_stop_patience = early_stop_patience
        self._early_stop_counter = 0
        # Reporting top-K
        self.top_k_report = top_k_report
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        # Coevolution populations (initialized lazily in run_coevolution)
        self.solvers = []  # List[ModelGenome]
        self.saboteurs = []  # List[SaboteurGenome]
        self.solver_avg_history = []
        self.solver_best_history = []
        self.saboteur_avg_history = []
        self.saboteur_best_history = []

    # ------------------ Novelty Utilities ------------------
    @staticmethod
    def _levenshtein(a_tokens, b_tokens):
        """Compute Levenshtein distance between two token sequences."""
        n, m = len(a_tokens), len(b_tokens)
        if n == 0: return m
        if m == 0: return n
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a_tokens[i-1] == b_tokens[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[n][m]

    def _compute_novelty_scores(self, arch_sequences):
        """Return list of novelty scores (mean Levenshtein distance to others)."""
        if len(arch_sequences) < 2:
            return [0.0]*len(arch_sequences)
        distances = [[0.0]*len(arch_sequences) for _ in range(len(arch_sequences))]
        for i in range(len(arch_sequences)):
            for j in range(i+1, len(arch_sequences)):
                d = self._levenshtein(arch_sequences[i], arch_sequences[j])
                distances[i][j] = d
                distances[j][i] = d
        novelty = []
        for i in range(len(arch_sequences)):
            mean_d = sum(distances[i]) / (len(arch_sequences)-1)
            novelty.append(mean_d)
        # Normalizza 0-1 (evita divisione per zero)
        min_n, max_n = min(novelty), max(novelty)
        if max_n - min_n < 1e-8:
            return [0.0]*len(novelty)
        return [(n - min_n)/(max_n - min_n) for n in novelty]

    # ------------------ Coevolution Methods ------------------
    def _init_coevolution_populations(self):
        if not self.solvers:
            for _ in range(self.population_size):
                self.solvers.append(ModelGenome.create_random_genome(self.input_dim, self.output_dim))
        if not self.saboteurs:
            for _ in range(self.population_size):
                self.saboteurs.append(SaboteurGenome.random_genome())

    def _evaluate_match(self, solver: ModelGenome, saboteur: SaboteurGenome, train_seq_len:int, test_seq_len:int, train_samples:int=128, test_batch:int=64, epochs:int=3):
        """Train solver shortly on random parity data then test on saboteur sequences.
        Returns (solver_win_count, saboteur_win_count)."""
        # Generate simple random parity training data
        X_train = torch.randint(0,2,(train_samples, train_seq_len, self.input_dim)).float()
        y_train = (X_train.sum(dim=1) % 2).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Build and train solver model
        try:
            model = solver.build_pytorch_model(self.input_dim, self.output_dim).to(device)
            model.train()
        except Exception:
            return 0, 1  # saboteur wins by default if model fails
        params = list(model.parameters())
        if not params:
            return 0, 1
        optimizer = optim.Adam(params, lr=solver.learning_params.get('learning_rate', 0.005))
        criterion = nn.BCEWithLogitsLoss()
        batch_size = 32
        for epoch in range(epochs):
            idx = torch.randperm(train_samples)
            for i in range(0, train_samples, batch_size):
                b = idx[i:i+batch_size]
                xb = X_train[b].to(device)
                yb = y_train[b].to(device).view(-1)
                optimizer.zero_grad()
                out = model(xb).view(-1)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        # Test on saboteur sequences
        model.eval()
        X_adv, y_adv = saboteur.generate_batch(test_batch, test_seq_len)
        with torch.no_grad():
            logits = model(X_adv.to(device)).view(-1)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        correct = (preds == y_adv.view(-1)).sum().item()
        solver_wins = correct
        saboteur_wins = test_batch - correct
        return solver_wins, saboteur_wins

    def run_coevolution(self, generations:int, train_seq_len:int=16, test_seq_len:int=32, sample_size:int=10):
        """Run competitive co-evolution between solver and saboteur populations."""
        self._init_coevolution_populations()
        print("Starting Competitive Co-Evolution (Solvers vs Saboteurs)...")
        for gen in range(generations):
            print(f"\n== Generation {gen+1}/{generations} ==")
            # Sample participants
            solver_indices = random.sample(range(len(self.solvers)), min(sample_size, len(self.solvers)))
            saboteur_indices = random.sample(range(len(self.saboteurs)), min(sample_size, len(self.saboteurs)))
            solver_scores = {i:0 for i in solver_indices}
            saboteur_scores = {j:0 for j in saboteur_indices}
            matches = 0
            for si in solver_indices:
                for sj in saboteur_indices:
                    s_win, sab_win = self._evaluate_match(self.solvers[si], self.saboteurs[sj], train_seq_len, test_seq_len)
                    solver_scores[si] += s_win
                    saboteur_scores[sj] += sab_win
                    matches += 1
            # Compute fitness (normalize by total adversarial examples)
            solver_fitness = []
            for idx, genome in enumerate(self.solvers):
                if idx in solver_scores:
                    max_points = len(saboteur_indices) * 64  # test_batch fixed 64
                    solver_fitness.append(solver_scores[idx] / max_points)
                else:
                    solver_fitness.append(0.0)
            saboteur_fitness = []
            for idx, genome in enumerate(self.saboteurs):
                if idx in saboteur_scores:
                    max_points = len(solver_indices) * 64
                    saboteur_fitness.append(saboteur_scores[idx] / max_points)
                else:
                    saboteur_fitness.append(0.0)
            # Record stats
            solver_avg = float(np.mean(solver_fitness))
            solver_best = float(np.max(solver_fitness))
            sab_avg = float(np.mean(saboteur_fitness))
            sab_best = float(np.max(saboteur_fitness))
            self.solver_avg_history.append(solver_avg)
            self.solver_best_history.append(solver_best)
            self.saboteur_avg_history.append(sab_avg)
            self.saboteur_best_history.append(sab_best)
            print(f"Solver Avg: {solver_avg:.4f} | Solver Best: {solver_best:.4f}")
            print(f"Saboteur Avg: {sab_avg:.4f} | Saboteur Best: {sab_best:.4f}")
            # ----- Rich Coevolution Metrics -----
            solver_arch_counts = {}
            solver_param_counts = []
            solver_depths = []
            for s in self.solvers:
                try:
                    s.build_pytorch_model(self.input_dim, self.output_dim)
                    solver_param_counts.append(s.param_count())
                except Exception:
                    solver_param_counts.append(0)
                arch_s = s.get_architecture_string()
                solver_arch_counts[arch_s] = solver_arch_counts.get(arch_s,0)+1
                solver_depths.append(len(arch_s.split(' -> ')))
            pattern_types = {}
            noise_levels = []
            pattern_lengths = []
            for sb in self.saboteurs:
                pt = sb.params.get('pattern_type','?')
                pattern_types[pt] = pattern_types.get(pt,0)+1
                noise_levels.append(sb.params.get('noise_level',0.0))
                pattern_lengths.append(sb.params.get('pattern_length',0))
            total_s = sum(solver_arch_counts.values())
            s_probs = np.array([c/total_s for c in solver_arch_counts.values()]) if total_s>0 else np.array([1.0])
            solver_entropy = float(-np.sum(s_probs * np.log(s_probs + 1e-12)))
            co_metrics = {
                'generation': gen,
                'solver_avg': solver_avg,
                'solver_best': solver_best,
                'saboteur_avg': sab_avg,
                'saboteur_best': sab_best,
                'solver_unique_arch': len(solver_arch_counts),
                'solver_arch_entropy': solver_entropy,
                'solver_param_mean': float(np.mean(solver_param_counts)) if solver_param_counts else 0.0,
                'solver_param_std': float(np.std(solver_param_counts)) if solver_param_counts else 0.0,
                'solver_depth_mean': float(np.mean(solver_depths)) if solver_depths else 0.0,
                'pattern_type_counts': pattern_types,
                'pattern_length_mean': float(np.mean(pattern_lengths)) if pattern_lengths else 0.0,
                'noise_level_mean': float(np.mean(noise_levels)) if noise_levels else 0.0,
            }
            if not hasattr(self, 'coevo_metrics_history'):
                self.coevo_metrics_history = []
            self.coevo_metrics_history.append(co_metrics)
            print(f"  CoMetrics: s_ent={co_metrics['solver_arch_entropy']:.3f} s_arch={co_metrics['solver_unique_arch']} s_depthμ={co_metrics['solver_depth_mean']:.1f} s_paramμ={co_metrics['solver_param_mean']:.0f} sab_noiseμ={co_metrics['noise_level_mean']:.3f}")
            # Reproduce populations separately
            self._reproduce_solvers(solver_fitness)
            self._reproduce_saboteurs(saboteur_fitness)
        return (self.solvers, self.saboteurs,
                (self.solver_avg_history, self.solver_best_history, self.saboteur_avg_history, self.saboteur_best_history))

    def _reproduce_solvers(self, fitness_scores):
        elite_count = max(1, self.population_size // 10)
        elite_idx = np.argsort(fitness_scores)[-elite_count:]
        new_pop = [self.solvers[i] for i in elite_idx]
        while len(new_pop) < self.population_size:
            p1 = self.solvers[self._tournament_index(fitness_scores)]
            if random.random() < self.crossover_rate:
                p2 = self.solvers[self._tournament_index(fitness_scores)]
                child = p1.crossover(p2)
            else:
                child = ModelGenome(p1.genes.copy(), p1.learning_params.copy())
            child.mutate(self.mutation_rate)
            new_pop.append(child)
        self.solvers = new_pop

    def _reproduce_saboteurs(self, fitness_scores):
        elite_count = max(1, self.population_size // 10)
        elite_idx = np.argsort(fitness_scores)[-elite_count:]
        new_pop = [self.saboteurs[i].clone() for i in elite_idx]
        while len(new_pop) < self.population_size:
            p1 = self.saboteurs[self._tournament_index(fitness_scores)]
            if random.random() < self.crossover_rate:
                p2 = self.saboteurs[self._tournament_index(fitness_scores)]
                child = p1.crossover(p2)
            else:
                child = p1.clone()
            child.mutate(self.mutation_rate)
            new_pop.append(child)
        self.saboteurs = new_pop

    def _tournament_index(self, fitness_scores):
        contenders = random.sample(range(len(fitness_scores)), min(self.tournament_size, len(fitness_scores)))
        best = max(contenders, key=lambda i: fitness_scores[i])
        return best

    def _generate_data(self, sequence_length, num_samples):
        np.random.seed(42)
        X = np.random.randn(num_samples, sequence_length, self.input_dim).astype(np.float32)
        y = (np.sum(X[:, :, 0], axis=1) > 0).astype(np.float32).reshape(-1, 1)
        return torch.tensor(X), torch.tensor(y)

    def _evaluate_genome_fitness(self, genome, X_data, y_data):
        """Evaluate genome fitness with multiple runs to reduce stochastic noise."""
        try:
            # Ensure architecture is built before validation.
            # Previously is_valid() was called before grammar expansion, causing
            # built_architecture=None and fitness=0 for all genomes.
            if genome.built_architecture is None:
                genome.build_from_grammar()

            if not genome.is_valid():
                # Minimal debug log (avoid flooding output)
                return 0.0
            
            # Perform num_eval_runs evaluations with different seeds
            fitness_scores = []
            for eval_run in range(self.num_eval_runs):
                # Distinct seed per evaluation
                seed = random.randint(0, 2**31 - 1)
                perf = train_and_evaluate_genome(genome, X_data, y_data, genome.learning_params, epochs=15, random_seed=seed)
                fitness_scores.append(perf)
            
            # Average evaluations for stable fitness estimate
            avg_performance = np.mean(fitness_scores)
            
            # Complexity penalty (small to avoid wiping fitness)
            complexity = sum(p.numel() for p in genome.parameters())
            penalty = self.complexity_penalty_coef * complexity
            
            return max(0.0, avg_performance - penalty)
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return 0.0

    def _tournament_selection(self, fitness_scores):
        """Tournament selection (competitive)."""
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def run(self, generations, sequence_length=10, num_samples=1000):
        print("Starting Grammar-based Neuroevolution...")
        print(f"Population: {self.population_size}, Evaluations per genome: {self.num_eval_runs}")
        X_data, y_data = self._generate_data(sequence_length, num_samples)
        
        # Initialize population with grammar-based genomes
        for _ in range(self.population_size):
            genome = ModelGenome.create_random_genome(self.input_dim, self.output_dim)
            self.population.append(genome)

        for generation in range(generations):
            print(f"\\nGeneration {generation + 1}/{generations}")
            
            # Evaluate fitness with repeated evaluations
            fitness_scores = []
            arch_strings = []
            arch_counts = {}
            arch_token_lists = []
            for i, g in enumerate(self.population):
                fit = self._evaluate_genome_fitness(g, X_data, y_data)
                arch = g.get_architecture_string()
                arch_strings.append(arch)
                arch_token_lists.append(arch.split(' -> '))
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
                fitness_scores.append(fit)
                # Sampled printing + highlight good solutions
                if i % 5 == 0 or fit > 0.6:
                    print(f"  Genome {i}: fitness = {fit:.4f}, arch = {arch[:50]}...")

            # Apply diversity pressure: penalize repeated architectures
            if self.diversity_pressure > 0.0:
                for i, arch in enumerate(arch_strings):
                    count = arch_counts[arch]
                    if count > 1:
                        # Reduction proportional to number of duplicates (more copies -> larger penalty)
                        dup_factor = (count - 1) / count  # in [0,1)
                        penalty_factor = 1.0 - self.diversity_pressure * dup_factor
                        fitness_scores[i] *= max(0.0, penalty_factor)

            # Compute novelty and blend into fitness if enabled
            if self.novelty_weight > 0.0:
                novelty_scores = self._compute_novelty_scores(arch_token_lists)
                for i in range(len(fitness_scores)):
                    fitness_scores[i] = (1 - self.novelty_weight) * fitness_scores[i] + self.novelty_weight * novelty_scores[i]
            else:
                novelty_scores = [0.0]*len(fitness_scores)

            # ----- Rich Metrics (standard evolution) -----
            try:
                fitness_arr = np.array(fitness_scores) if fitness_scores else np.array([0.0])
                param_counts = []
                depths = []
                for g in self.population:
                    try:
                        g.build_pytorch_model(self.input_dim, self.output_dim)
                        param_counts.append(g.param_count())
                    except Exception:
                        param_counts.append(0)
                    depths.append(len(g.get_architecture_string().split(' -> ')))
                param_arr = np.array(param_counts) if param_counts else np.array([0])
                depth_arr = np.array(depths) if depths else np.array([0])
                total_arch = sum(arch_counts.values())
                probs = np.array([c/total_arch for c in arch_counts.values()]) if total_arch>0 else np.array([1.0])
                arch_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                metrics = {
                    'generation': generation,
                    'fitness_mean': float(fitness_arr.mean()),
                    'fitness_median': float(np.median(fitness_arr)),
                    'fitness_std': float(fitness_arr.std()),
                    'fitness_min': float(fitness_arr.min()),
                    'fitness_max': float(fitness_arr.max()),
                    'novelty_mean': float(np.mean(novelty_scores)) if novelty_scores else 0.0,
                    'novelty_max': float(np.max(novelty_scores)) if novelty_scores else 0.0,
                    'arch_entropy': arch_entropy,
                    'unique_architectures': len(arch_counts),
                    'param_mean': float(param_arr.mean()),
                    'param_median': float(np.median(param_arr)),
                    'param_std': float(param_arr.std()),
                    'param_min': int(param_arr.min()),
                    'param_max': int(param_arr.max()),
                    'depth_mean': float(depth_arr.mean()),
                    'depth_median': float(np.median(depth_arr)),
                    'depth_std': float(depth_arr.std()),
                    'depth_min': int(depth_arr.min()),
                    'depth_max': int(depth_arr.max()),
                }
                if not hasattr(self, 'metrics_history'):
                    self.metrics_history = []
                self.metrics_history.append(metrics)
                print(f"  Metrics: med={metrics['fitness_median']:.4f} std={metrics['fitness_std']:.4f} uniq={metrics['unique_architectures']} ent={metrics['arch_entropy']:.3f} depthμ={metrics['depth_mean']:.1f} paramsμ={metrics['param_mean']:.0f}")
            except Exception as _e_metrics:
                print(f"  Metrics collection failed: {_e_metrics}")
            
            # Record statistics
            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.max(fitness_scores)
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(best_fitness)
            
            print(f"  Average fitness: {avg_fitness:.4f}")
            print(f"  Best fitness: {best_fitness:.4f}")
            
            # Show best architecture of the generation
            best_idx = np.argmax(fitness_scores)
            best_genome = self.population[best_idx]
            print(f"  Best architecture: {best_genome.get_architecture_string()}")

            # Top-K unique architectures
            if self.top_k_report > 0:
                unique = {}
                for arch, fit in zip(arch_strings, fitness_scores):
                    if arch not in unique or fit > unique[arch]:
                        unique[arch] = fit
                top_sorted = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:self.top_k_report]
                print("  Top-K unique architectures:")
                for rank, (arch, fit) in enumerate(top_sorted, 1):
                    print(f"    {rank}. {fit:.4f} | {arch}")

            # Early stopping check
            if self.early_stop_fitness is not None:
                if best_fitness >= self.early_stop_fitness:
                    self._early_stop_counter += 1
                else:
                    self._early_stop_counter = 0
                if self._early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered (fitness >= {self.early_stop_fitness} for {self.early_stop_patience} consecutive generations).")
                    break
            
            # Elitism: keep the top 10%
            elite_count = max(1, self.population_size // 10)  # Top 10%
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            new_population = [self.population[i] for i in elite_indices]
            
            # Generate new population via crossover + grammar mutations
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate and len(self.population) > 1:
                    # Crossover between two parents
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)
                    child = parent1.crossover(parent2)
                else:
                    # Tournament-selected parent for mutation
                    parent = self._tournament_selection(fitness_scores)
                    child = ModelGenome(parent.genes.copy(), parent.learning_params.copy())
                
                # Apply mutations
                child.mutate(self.mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population

        # Evaluate final population
        final_fitness = []
        for g in self.population:
            fit = self._evaluate_genome_fitness(g, X_data, y_data)
            final_fitness.append(fit)
        
        best_idx = np.argmax(final_fitness)
        best_genome = self.population[best_idx]
        
        print(f"\\nBest final fitness: {final_fitness[best_idx]:.4f}")
        print(f"Best final architecture: {best_genome.get_architecture_string()}")
        print(f"Best genes: {best_genome.genes}")
        
        return best_genome, self.fitness_history, self.best_fitness_history


def plot_evolution(fitness_history, best_fitness_history, solver_histories=None):
    # Backwards compatible: if solver_histories is None behave as before
    if solver_histories is None:
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, label='Average Fitness')
        plt.plot(best_fitness_history, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    # Otherwise expect tuple: (solver_avg, solver_best, sab_avg, sab_best)
    solver_avg, solver_best, sab_avg, sab_best = solver_histories
    gens = range(1, len(solver_avg)+1)
    plt.figure(figsize=(12,6))
    plt.plot(gens, solver_avg, label='Solver Avg', color='blue', linestyle='-')
    plt.plot(gens, solver_best, label='Solver Best', color='blue', linestyle='--')
    plt.plot(gens, sab_avg, label='Saboteur Avg', color='red', linestyle='-')
    plt.plot(gens, sab_best, label='Saboteur Best', color='red', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Normalized Score')
    plt.title('Competitive Co-Evolution Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class _MetricsExportMixin:
    def export_metrics(self):
        return {
            'standard': getattr(self, 'metrics_history', []),
            'coevolution': getattr(self, 'coevo_metrics_history', []),
        }