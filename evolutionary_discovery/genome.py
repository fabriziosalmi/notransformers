"""
Grammatical Neuroevolution - ModelGenome

This module implements a genome based on a sequence of genes (integers)
that are used to build neural architectures through a predefined grammar.
"""

import torch
import torch.nn as nn
import random
import copy
from typing import List, Dict, Any
from grammar import GRAMMAR, expand_grammar


class ModelGenome:
    """
    Grammar-based genome for neuroevolution.
    
    The genome contains:
    - genes: List of integers guiding grammatical choices
    - learning_params: Evolutionary learning parameters
    - built_architecture: Sequence of terminals built from the grammar
    """
    
    def __init__(self, genes: List[int], learning_params: Dict[str, Any]):
        self.genes = genes.copy()  # List of integers guiding choices
        self.learning_params = learning_params.copy()
        self.built_architecture = None  # Cache of built architecture
        self.model = None  # Cache of PyTorch model
        self.input_dim = None
        self.output_dim = None
        self._param_count_cache = None  # Cache of parameter count
    
    def build_from_grammar(self, grammar=None, max_expansions=50):
        """
        Builds the terminal sequence from the grammar using genes.
        
        Returns:
            List[str]: Sequence of terminals representing the architecture
        """
        if grammar is None:
            grammar = GRAMMAR
        
        # If already built, return cache
        if self.built_architecture is not None:
            return self.built_architecture
        
        # Expand the grammar using genes
        self.built_architecture = expand_grammar(self.genes, grammar, max_expansions)
        return self.built_architecture
    
    def build_pytorch_model(self, input_dim: int, output_dim: int):
        """
        Builds the PyTorch model from the grammatical architecture.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            nn.Module: PyTorch model
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # If already built, return cache
        if self.model is not None:
            return self.model
        
        # Get the terminal sequence
        architecture = self.build_from_grammar()
        
        if not architecture:
            # Empty architecture, create a simple linear model
            self.model = nn.Linear(input_dim, output_dim)
            return self.model
        
        # Build the sequential model
        layers = []
        current_dim = input_dim
        residual_stack = []  # Stack to track residual connections
        
        i = 0
        while i < len(architecture):
            terminal = architecture[i]
            
            if terminal == "Conv1D":
                # Conv1D for sequential data - needs transposition for (batch, channels, seq_len)
                out_channels = max(8, current_dim)  # At least 8 channels
                conv = nn.Conv1d(current_dim, out_channels, kernel_size=3, padding=1)
                layers.append(Conv1DWrapper(conv))
                current_dim = out_channels
                
            elif terminal == "GRU":
                # GRU for sequential processing
                hidden_size = max(8, current_dim)
                gru = nn.GRU(current_dim, hidden_size, batch_first=True)
                layers.append(GRUWrapper(gru))
                current_dim = hidden_size
                
            elif terminal == "Linear":
                # Linear layer - needs 2D input, so take the last timestep
                out_features = max(8, current_dim // 2) if current_dim > 8 else current_dim
                linear = nn.Linear(current_dim, out_features)
                layers.append(SequenceToVector())  # Convert sequence to vector
                layers.append(linear)
                current_dim = out_features
                
            elif terminal == "LayerNorm":
                # Layer normalization
                layers.append(nn.LayerNorm(current_dim))
                
            elif terminal in ["ReLU", "Tanh", "GELU"]:
                # Activation functions
                if terminal == "ReLU":
                    layers.append(nn.ReLU())
                elif terminal == "Tanh":
                    layers.append(nn.Tanh())
                elif terminal == "GELU":
                    layers.append(nn.GELU())
                    
            elif terminal == "residual":
                # Residual connection - save current state
                residual_stack.append(len(layers))
                
            elif terminal == "identity":
                # Identity operation - do nothing
                pass
            
            i += 1
        
        # Add final layer for output
        if current_dim != output_dim:
            layers.append(nn.Linear(current_dim, output_dim))
        
        # Handle residual connections
        if residual_stack:
            self.model = ResidualNetwork(layers, residual_stack)
        else:
            self.model = nn.Sequential(*layers)
        # Invalidate param count cache (new model)
        self._param_count_cache = None
        return self.model
    
    def forward(self, x):
        """Forward pass through the model"""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_pytorch_model first.")
        return self.model(x)
    
    def parameters(self):
        """Returns PyTorch model parameters"""
        if self.model is None:
            return []
        return list(self.model.parameters())

    def param_count(self):
        """Total parameter count (cached)."""
        if self._param_count_cache is None:
            if self.model is None:
                return 0
            self._param_count_cache = sum(p.numel() for p in self.model.parameters())
        return self._param_count_cache
    
    def is_valid(self):
        """Check if genome is valid"""
        return len(self.genes) > 0 and len(self.built_architecture or []) > 0
    
    def mutate(self, mutation_rate=0.1):
        """
        Mutate the genome by modifying some genes and learning parameters.
        
        Args:
            mutation_rate: Mutation probability for each gene
        """
        # Mutate genes
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = random.randint(0, 10)  # New random gene
        
        # Occasionally add or remove genes
        if random.random() < mutation_rate * 0.5:
            if random.random() < 0.5 and len(self.genes) < 50:
                # Add a gene
                self.genes.append(random.randint(0, 10))
            elif len(self.genes) > 5:
                # Remove a gene
                self.genes.pop(random.randint(0, len(self.genes) - 1))
        
        # Mutate learning parameters
        self.mutate_learning_params(mutation_rate)
        
        # Invalidate cache
        self.built_architecture = None
        self.model = None
        self._param_count_cache = None
    
    def mutate_learning_params(self, mutation_rate):
        """Mutate learning parameters"""
        if random.random() < mutation_rate:
            self.learning_params['learning_rate'] = random.uniform(0.0001, 0.1)
        
        if random.random() < mutation_rate:
            self.learning_params['optimizer'] = random.choice(['adam', 'sgd', 'rmsprop'])
        
        if random.random() < mutation_rate:
            self.learning_params['activation_function'] = random.choice(['relu', 'tanh', 'gelu', 'sigmoid'])
        
        if random.random() < mutation_rate:
            self.learning_params['lr_scheduler'] = random.choice(['none', 'step', 'cosine', 'exponential'])
    
    @classmethod
    def create_random_genome(cls, input_dim: int, output_dim: int, gene_length=20):
        """
        Create a random genome.
        
        Args:
            input_dim: Input dimension (not used for now)
            output_dim: Output dimension (not used for now)
            gene_length: Initial gene sequence length
            
        Returns:
            ModelGenome: New random genome
        """
        # Generate a random sequence of genes
        genes = [random.randint(0, 10) for _ in range(gene_length)]
        
        # Random learning parameters
        learning_params = {
            'learning_rate': random.uniform(0.001, 0.01),
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
            'activation_function': random.choice(['relu', 'tanh', 'gelu', 'sigmoid']),
            'lr_scheduler': random.choice(['none', 'step', 'cosine', 'exponential'])
        }
        
        return cls(genes, learning_params)
    
    def crossover(self, other_genome):
        """
        Crossover with another genome.
        
        Args:
            other_genome: The other genome for crossover
            
        Returns:
            ModelGenome: New child genome
        """
        # Gene crossover
        min_length = min(len(self.genes), len(other_genome.genes))
        if min_length < 2:
            # If one genome is too small, simply copy
            child_genes = self.genes.copy()
        else:
            cut_point = random.randint(1, min_length - 1)
            child_genes = self.genes[:cut_point] + other_genome.genes[cut_point:]
        
        # Learning parameter crossover
        child_learning_params = {}
        for key in self.learning_params:
            if random.random() < 0.5:
                child_learning_params[key] = self.learning_params[key]
            else:
                child_learning_params[key] = other_genome.learning_params.get(key, self.learning_params[key])
        
        return ModelGenome(child_genes, child_learning_params)
    
    def get_architecture_string(self):
        """Returns a readable representation of the architecture"""
        architecture = self.build_from_grammar()
        return " -> ".join(architecture) if architecture else "Empty Architecture"
    
    def __str__(self):
        """String representation of the genome"""
        arch = self.get_architecture_string()
        return f"ModelGenome(genes={len(self.genes)}, arch='{arch}', lr={self.learning_params.get('learning_rate', 'N/A'):.4f})"
    
    def __repr__(self):
        return self.__str__()


class SequenceToVector(nn.Module):
    """Convert sequence (batch, seq, features) to vector (batch, features) by taking the last timestep"""
    
    def forward(self, x):
        if x.dim() == 3:  # (batch, seq, features)
            return x[:, -1, :]  # Take last timestep
        else:  # Already in correct format
            return x


class GRUWrapper(nn.Module):
    """Wrapper for GRU that returns only the final output"""
    
    def __init__(self, gru):
        super().__init__()
        self.gru = gru
    
    def forward(self, x):
        # Support 2D input (batch, features) by converting to sequence of length 1
        if x.dim() == 2:  # (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
        elif x.dim() != 3:
            raise ValueError(f"GRUWrapper expected 2D or 3D tensor, got shape={x.shape}")

        output, _ = self.gru(x)
        # Take only last timestep (handles seq_len=1 too)
        return output[:, -1, :]


class Conv1DWrapper(nn.Module):
    """Wrapper for Conv1D that handles dimensional transposition"""
    
    def __init__(self, conv1d):
        super().__init__()
        self.conv1d = conv1d
    
    def forward(self, x):
        # Handle 2D input by expanding to 3D if necessary
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, features) -> (batch, features, 1)
        
        # Input: (batch, seq_len, features) or (batch, features, seq_len)
        # Conv1D expects: (batch, features, seq_len)
        if x.shape[1] != self.conv1d.in_channels:
            x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        x = self.conv1d(x)     # (batch, out_channels, seq_len)
        
        # Return (batch, out_channels) for compatibility with Linear
        if x.shape[-1] == 1:
            return x.squeeze(-1)  # (batch, out_channels)
        else:
            return x.mean(dim=-1)  # Mean over timesteps


class ResidualNetwork(nn.Module):
    """
    Network that handles residual connections based on markers in the sequence.
    """
    
    def __init__(self, layers, residual_positions):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.residual_positions = residual_positions
    
    def forward(self, x):
        residual_values = {}  # Save values for residual connections
        
        for i, layer in enumerate(self.layers):
            # Save value if it's a residual position
            if i in self.residual_positions:
                residual_values[i] = x.clone()
            
            # Apply the layer
            x = layer(x)
            
            # If there's a compatible previous residual connection, add it
            for res_pos in self.residual_positions:
                if res_pos < i and res_pos in residual_values:
                    residual_val = residual_values[res_pos]
                    # Check dimensional compatibility
                    if self._are_compatible(x, residual_val):
                        x = x + residual_val
                        break  # Only one residual connection per layer
        
        return x
    
    def _are_compatible(self, tensor1, tensor2):
        """Check if two tensors are compatible for addition"""
        # Handles both 2D (batch, features) and 3D (batch, seq, features) tensors
        if tensor1.dim() != tensor2.dim():
            return False
        
        if tensor1.dim() == 2:  # (batch, features)
            return tensor1.shape[-1] == tensor2.shape[-1]
        elif tensor1.dim() == 3:  # (batch, seq, features)
            return tensor1.shape[-1] == tensor2.shape[-1] and tensor1.shape[-2] == tensor2.shape[-2]
        else:
            return tensor1.shape == tensor2.shape


if __name__ == "__main__":
    # System test
    print("=== GRAMMATICAL NEUROEVOLUTION TEST ===")
    
    # Create a random genome
    genome = ModelGenome.create_random_genome(input_dim=10, output_dim=1)
    print(f"Genome created: {genome}")
    print(f"Genes: {genome.genes}")
    print(f"Architecture: {genome.get_architecture_string()}")
    
    # Build PyTorch model
    model = genome.build_pytorch_model(input_dim=10, output_dim=1)
    print(f"\nPyTorch model created:")
    print(model)
    
    # Test forward pass
    x = torch.randn(32, 8, 10)  # (batch, seq_len, features)
    try:
        output = model(x)
        print(f"\nOutput shape: {output.shape}")
        print("✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
    
    # Test mutation
    print(f"\nBefore mutation: {genome.genes[:10]}...")
    genome.mutate(mutation_rate=0.3)
    print(f"After mutation: {genome.genes[:10]}...")
    
    # Test crossover
    genome2 = ModelGenome.create_random_genome(input_dim=10, output_dim=1)
    child = genome.crossover(genome2)
    print(f"\nParent 1: {genome.genes[:5]}...")
    print(f"Parent 2: {genome2.genes[:5]}...")
    print(f"Child: {child.genes[:5]}...")
    
    print("\n=== TEST COMPLETED ===")