"""
Neuroevoluzione Grammaticale - ModelGenome

Questo modulo implementa un genoma basato su una sequenza di geni (interi)
che vengono usati per costruire architetture neurali attraverso una grammatica predefinita.
"""

import torch
import torch.nn as nn
import random
import copy
from typing import List, Dict, Any
from grammar import GRAMMAR, expand_grammar


class ModelGenome:
    """
    Genoma basato su grammatica per la neuroevoluzione.
    
    Il genoma contiene:
    - genes: Lista di interi che guidano le scelte grammaticali
    - learning_params: Parametri di apprendimento evolutivi
    - built_architecture: Sequenza di terminali costruita dalla grammatica
    """
    
    def __init__(self, genes: List[int], learning_params: Dict[str, Any]):
        self.genes = genes.copy()  # Lista di interi che guidano le scelte
        self.learning_params = learning_params.copy()
        self.built_architecture = None  # Cache dell'architettura costruita
        self.model = None  # Cache del modello PyTorch
        self.input_dim = None
        self.output_dim = None
        self._param_count_cache = None  # Cache del numero di parametri
    
    def build_from_grammar(self, grammar=None, max_expansions=50):
        """
        Costruisce la sequenza di terminali dalla grammatica usando i geni.
        
        Returns:
            List[str]: Sequenza di terminali che rappresenta l'architettura
        """
        if grammar is None:
            grammar = GRAMMAR
        
        # Se già costruita, restituisci la cache
        if self.built_architecture is not None:
            return self.built_architecture
        
        # Espandi la grammatica usando i geni
        self.built_architecture = expand_grammar(self.genes, grammar, max_expansions)
        return self.built_architecture
    
    def build_pytorch_model(self, input_dim: int, output_dim: int):
        """
        Costruisce il modello PyTorch dall'architettura grammaticale.
        
        Args:
            input_dim: Dimensione dell'input
            output_dim: Dimensione dell'output
            
        Returns:
            nn.Module: Modello PyTorch
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Se già costruito, restituisci la cache
        if self.model is not None:
            return self.model
        
        # Ottieni la sequenza di terminali
        architecture = self.build_from_grammar()
        
        if not architecture:
            # Architettura vuota, crea un modello lineare semplice
            self.model = nn.Linear(input_dim, output_dim)
            return self.model
        
        # Costruisci il modello sequenziale
        layers = []
        current_dim = input_dim
        residual_stack = []  # Stack per tenere traccia delle connessioni residue
        
        i = 0
        while i < len(architecture):
            terminal = architecture[i]
            
            if terminal == "Conv1D":
                # Conv1D per dati sequenziali - necessita trasposizione per (batch, channels, seq_len)
                out_channels = max(8, current_dim)  # Almeno 8 canali
                conv = nn.Conv1d(current_dim, out_channels, kernel_size=3, padding=1)
                layers.append(Conv1DWrapper(conv))
                current_dim = out_channels
                
            elif terminal == "GRU":
                # GRU per elaborazione sequenziale
                hidden_size = max(8, current_dim)
                gru = nn.GRU(current_dim, hidden_size, batch_first=True)
                layers.append(GRUWrapper(gru))
                current_dim = hidden_size
                
            elif terminal == "Linear":
                # Layer lineare - ha bisogno di input 2D, quindi prendiamo l'ultimo timestep
                out_features = max(8, current_dim // 2) if current_dim > 8 else current_dim
                linear = nn.Linear(current_dim, out_features)
                layers.append(SequenceToVector())  # Converte sequenza a vettore
                layers.append(linear)
                current_dim = out_features
                
            elif terminal == "LayerNorm":
                # Normalizzazione layer
                layers.append(nn.LayerNorm(current_dim))
                
            elif terminal in ["ReLU", "Tanh", "GELU"]:
                # Funzioni di attivazione
                if terminal == "ReLU":
                    layers.append(nn.ReLU())
                elif terminal == "Tanh":
                    layers.append(nn.Tanh())
                elif terminal == "GELU":
                    layers.append(nn.GELU())
                    
            elif terminal == "residual":
                # Connessione residua - salva lo stato corrente
                residual_stack.append(len(layers))
                
            elif terminal == "identity":
                # Operazione identità - non fare nulla
                pass
            
            i += 1
        
        # Aggiungi il layer finale per l'output
        if current_dim != output_dim:
            layers.append(nn.Linear(current_dim, output_dim))
        
        # Gestisci le connessioni residue
        if residual_stack:
            self.model = ResidualNetwork(layers, residual_stack)
        else:
            self.model = nn.Sequential(*layers)
        # Invalida param count cache (nuovo modello)
        self._param_count_cache = None
        return self.model
    
    def forward(self, x):
        """Forward pass del modello"""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_pytorch_model first.")
        return self.model(x)
    
    def parameters(self):
        """Restituisce i parametri del modello PyTorch"""
        if self.model is None:
            return []
        return list(self.model.parameters())

    def param_count(self):
        """Numero totale di parametri (cache)."""
        if self._param_count_cache is None:
            if self.model is None:
                return 0
            self._param_count_cache = sum(p.numel() for p in self.model.parameters())
        return self._param_count_cache
    
    def is_valid(self):
        """Verifica se il genoma è valido"""
        return len(self.genes) > 0 and len(self.built_architecture or []) > 0
    
    def mutate(self, mutation_rate=0.1):
        """
        Muta il genoma modificando alcuni geni e parametri di apprendimento.
        
        Args:
            mutation_rate: Probabilità di mutazione per ogni gene
        """
        # Muta i geni
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = random.randint(0, 10)  # Nuovo gene casuale
        
        # Occasionalmente aggiungi o rimuovi geni
        if random.random() < mutation_rate * 0.5:
            if random.random() < 0.5 and len(self.genes) < 50:
                # Aggiungi un gene
                self.genes.append(random.randint(0, 10))
            elif len(self.genes) > 5:
                # Rimuovi un gene
                self.genes.pop(random.randint(0, len(self.genes) - 1))
        
        # Muta i parametri di apprendimento
        self.mutate_learning_params(mutation_rate)
        
        # Invalida la cache
        self.built_architecture = None
        self.model = None
        self._param_count_cache = None
    
    def mutate_learning_params(self, mutation_rate):
        """Muta i parametri di apprendimento"""
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
        Crea un genoma casuale.
        
        Args:
            input_dim: Dimensione input (non usata per ora)
            output_dim: Dimensione output (non usata per ora)
            gene_length: Lunghezza iniziale della sequenza di geni
            
        Returns:
            ModelGenome: Nuovo genoma casuale
        """
        # Genera una sequenza casuale di geni
        genes = [random.randint(0, 10) for _ in range(gene_length)]
        
        # Parametri di apprendimento casuali
        learning_params = {
            'learning_rate': random.uniform(0.001, 0.01),
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
            'activation_function': random.choice(['relu', 'tanh', 'gelu', 'sigmoid']),
            'lr_scheduler': random.choice(['none', 'step', 'cosine', 'exponential'])
        }
        
        return cls(genes, learning_params)
    
    def crossover(self, other_genome):
        """
        Crossover con un altro genoma.
        
        Args:
            other_genome: L'altro genoma per il crossover
            
        Returns:
            ModelGenome: Nuovo genoma figlio
        """
        # Crossover dei geni
        min_length = min(len(self.genes), len(other_genome.genes))
        if min_length < 2:
            # Se uno dei genomi è troppo piccolo, copia semplicemente
            child_genes = self.genes.copy()
        else:
            cut_point = random.randint(1, min_length - 1)
            child_genes = self.genes[:cut_point] + other_genome.genes[cut_point:]
        
        # Crossover dei parametri di apprendimento
        child_learning_params = {}
        for key in self.learning_params:
            if random.random() < 0.5:
                child_learning_params[key] = self.learning_params[key]
            else:
                child_learning_params[key] = other_genome.learning_params.get(key, self.learning_params[key])
        
        return ModelGenome(child_genes, child_learning_params)
    
    def get_architecture_string(self):
        """Restituisce una rappresentazione leggibile dell'architettura"""
        architecture = self.build_from_grammar()
        return " -> ".join(architecture) if architecture else "Empty Architecture"
    
    def __str__(self):
        """Rappresentazione string del genoma"""
        arch = self.get_architecture_string()
        return f"ModelGenome(genes={len(self.genes)}, arch='{arch}', lr={self.learning_params.get('learning_rate', 'N/A'):.4f})"
    
    def __repr__(self):
        return self.__str__()


class SequenceToVector(nn.Module):
    """Converte sequenza (batch, seq, features) a vettore (batch, features) prendendo l'ultimo timestep"""
    
    def forward(self, x):
        if x.dim() == 3:  # (batch, seq, features)
            return x[:, -1, :]  # Prendi ultimo timestep
        else:  # Già in formato corretto
            return x


class GRUWrapper(nn.Module):
    """Wrapper per GRU che restituisce solo l'output finale"""
    
    def __init__(self, gru):
        super().__init__()
        self.gru = gru
    
    def forward(self, x):
        # Supporta input 2D (batch, features) convertendolo in sequenza di lunghezza 1.
        if x.dim() == 2:  # (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
        elif x.dim() != 3:
            raise ValueError(f"GRUWrapper expected 2D or 3D tensor, got shape={x.shape}")

        output, _ = self.gru(x)
        # Prendi solo l'ultimo timestep (gestisce anche seq_len=1)
        return output[:, -1, :]


class Conv1DWrapper(nn.Module):
    """Wrapper per Conv1D che gestisce la trasposizione dimensionale"""
    
    def __init__(self, conv1d):
        super().__init__()
        self.conv1d = conv1d
    
    def forward(self, x):
        # Gestisci input 2D espandendolo a 3D se necessario
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, features) -> (batch, features, 1)
        
        # Input: (batch, seq_len, features) o (batch, features, seq_len)
        # Conv1D expects: (batch, features, seq_len)
        if x.shape[1] != self.conv1d.in_channels:
            x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        x = self.conv1d(x)     # (batch, out_channels, seq_len)
        
        # Ritorna (batch, out_channels) per compatibilità con Linear
        if x.shape[-1] == 1:
            return x.squeeze(-1)  # (batch, out_channels)
        else:
            return x.mean(dim=-1)  # Media sui timestep


class ResidualNetwork(nn.Module):
    """
    Rete che gestisce connessioni residue basate sui marker nella sequenza.
    """
    
    def __init__(self, layers, residual_positions):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.residual_positions = residual_positions
    
    def forward(self, x):
        residual_values = {}  # Salva i valori per le connessioni residue
        
        for i, layer in enumerate(self.layers):
            # Salva il valore se è una posizione residua
            if i in self.residual_positions:
                residual_values[i] = x.clone()
            
            # Applica il layer
            x = layer(x)
            
            # Se c'è una connessione residua precedente compatibile, sommala
            for res_pos in self.residual_positions:
                if res_pos < i and res_pos in residual_values:
                    residual_val = residual_values[res_pos]
                    # Controlla compatibilità dimensionale
                    if self._are_compatible(x, residual_val):
                        x = x + residual_val
                        break  # Una sola connessione residua per layer
        
        return x
    
    def _are_compatible(self, tensor1, tensor2):
        """Verifica se due tensori sono compatibili per la somma"""
        # Gestisce sia tensori 2D (batch, features) che 3D (batch, seq, features)
        if tensor1.dim() != tensor2.dim():
            return False
        
        if tensor1.dim() == 2:  # (batch, features)
            return tensor1.shape[-1] == tensor2.shape[-1]
        elif tensor1.dim() == 3:  # (batch, seq, features)
            return tensor1.shape[-1] == tensor2.shape[-1] and tensor1.shape[-2] == tensor2.shape[-2]
        else:
            return tensor1.shape == tensor2.shape


if __name__ == "__main__":
    # Test del sistema
    print("=== TEST NEUROEVOLUZIONE GRAMMATICALE ===")
    
    # Crea un genoma casuale
    genome = ModelGenome.create_random_genome(input_dim=10, output_dim=1)
    print(f"Genoma creato: {genome}")
    print(f"Geni: {genome.genes}")
    print(f"Architettura: {genome.get_architecture_string()}")
    
    # Costruisci il modello PyTorch
    model = genome.build_pytorch_model(input_dim=10, output_dim=1)
    print(f"\\nModello PyTorch creato:")
    print(model)
    
    # Test forward pass
    x = torch.randn(32, 8, 10)  # (batch, seq_len, features)
    try:
        output = model(x)
        print(f"\\nOutput shape: {output.shape}")
        print("✅ Forward pass riuscito!")
    except Exception as e:
        print(f"❌ Errore nel forward pass: {e}")
    
    # Test mutazione
    print(f"\\nPrima della mutazione: {genome.genes[:10]}...")
    genome.mutate(mutation_rate=0.3)
    print(f"Dopo la mutazione: {genome.genes[:10]}...")
    
    # Test crossover
    genome2 = ModelGenome.create_random_genome(input_dim=10, output_dim=1)
    child = genome.crossover(genome2)
    print(f"\\nGenitore 1: {genome.genes[:5]}...")
    print(f"Genitore 2: {genome2.genes[:5]}...")
    print(f"Figlio: {child.genes[:5]}...")
    
    print("\\n=== TEST COMPLETATO ===")