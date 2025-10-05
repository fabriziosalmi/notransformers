import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputationalPrimitive(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
    def reset(self):
        pass

class InputNode(ComputationalPrimitive):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
    def forward(self, x):
        return x
    def __repr__(self):
        return f"InputNode(dim={self.input_dim})"

class Linear(ComputationalPrimitive):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = None  # Inizializziamo dinamicamente
    
    def _init_linear(self, actual_input_dim):
        """Inizializza il layer lineare con le dimensioni effettive dell'input"""
        if self.linear is None or self.linear.in_features != actual_input_dim:
            self.linear = nn.Linear(actual_input_dim, self.output_dim)
            if hasattr(self, '_device'):
                self.linear = self.linear.to(self._device)
    
    def forward(self, x):
        actual_input_dim = x.shape[-1]
        self._init_linear(actual_input_dim)
        return self.linear(x)
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.linear is not None:
            self.linear = self.linear.to(device)
        return self

class Conv1D(ComputationalPrimitive):
    def __init__(self, input_dim, output_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.conv = None
    
    def _init_conv(self, actual_input_dim):
        """Inizializza Conv1D con le dimensioni effettive dell'input"""
        if self.conv is None or self.conv.in_channels != actual_input_dim:
            self.conv = nn.Conv1d(actual_input_dim, self.output_dim, self.kernel_size, padding=self.kernel_size//2)
            if hasattr(self, '_device'):
                self.conv = self.conv.to(self._device)
    
    def forward(self, x):
        # x: (batch, seq, input_dim) or (batch, input_dim)
        if len(x.shape) == 2:  # (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)
        
        actual_input_dim = x.shape[2]  # channels
        self._init_conv(actual_input_dim)
        
        # x: (batch, seq, input_dim) -> (batch, input_dim, seq)
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.transpose(1, 2)  # (batch, seq, output_dim)
        
        # Se l'input originale era 2D, restituisci 2D
        if out.shape[1] == 1:
            out = out.squeeze(1)  # (batch, output_dim)
        
        return out
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.conv is not None:
            self.conv = self.conv.to(device)
        return self

class ElementwiseAdd(ComputationalPrimitive):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return x + y

class Concatenate(ComputationalPrimitive):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.cat([x, y], dim=-1)

class GatedRecurrentUnit(ComputationalPrimitive):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = None  # Inizializziamo dinamicamente
        self.h = None
    
    def _init_gru(self, actual_input_dim):
        """Inizializza la GRU con le dimensioni effettive dell'input"""
        if self.gru is None or self.gru.input_size != actual_input_dim:
            self.gru = nn.GRU(actual_input_dim, self.output_dim, batch_first=True)
            if hasattr(self, '_device'):
                self.gru = self.gru.to(self._device)
    
    def forward(self, x):
        # x: (batch, seq, input_dim) or (batch, input_dim)
        if len(x.shape) == 2:  # (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)
        
        batch_size, seq_len, actual_input_dim = x.shape
        
        # Inizializza la GRU con le dimensioni corrette
        self._init_gru(actual_input_dim)
        
        # Aggiusta lo stato nascosto per il batch size corrente
        if self.h is None or self.h.shape[1] != batch_size:
            self.h = torch.zeros(1, batch_size, self.output_dim, device=x.device, dtype=x.dtype)
        
        out, h = self.gru(x, self.h)
        self.h = h.detach()  # Detach to avoid backprop through time
        return out[:, -1, :]  # Return last hidden state for each batch
    
    def reset(self):
        self.h = None
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.gru is not None:
            self.gru = self.gru.to(device)
        return self

class ExponentialMovingAverage(ComputationalPrimitive):
    def __init__(self, input_dim, alpha=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.alpha = alpha
        self.state = None
    def forward(self, x):
        # x: (batch, input_dim) or (batch, seq, input_dim)
        if len(x.shape) == 2:  # (batch, input_dim)
            batch, input_dim = x.shape
            if self.state is None or self.state.shape[0] != batch:
                self.state = torch.zeros((batch, self.input_dim), device=x.device, dtype=x.dtype)
            self.state = self.alpha * x + (1 - self.alpha) * self.state
            return self.state
        else:  # (batch, seq, input_dim)
            batch, seq, _ = x.shape
            if self.state is None or self.state.shape[0] != batch:
                self.state = torch.zeros((batch, self.input_dim), device=x.device, dtype=x.dtype)
            for t in range(seq):
                self.state = self.alpha * x[:, t, :] + (1 - self.alpha) * self.state
            return self.state
    def reset(self):
        self.state = None

class LayerNorm(ComputationalPrimitive):
    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.input_dim = feature_dim
        self.output_dim = feature_dim
        self.norm = None
        self.eps = eps
    
    def _init_norm(self, actual_feature_dim):
        """Inizializza LayerNorm con le dimensioni effettive dell'input"""
        if self.norm is None or self.norm.normalized_shape[0] != actual_feature_dim:
            self.norm = nn.LayerNorm(actual_feature_dim, eps=self.eps)
            if hasattr(self, '_device'):
                self.norm = self.norm.to(self._device)
    
    def forward(self, x):
        actual_feature_dim = x.shape[-1]
        self._init_norm(actual_feature_dim)
        return self.norm(x)
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.norm is not None:
            self.norm = self.norm.to(device)
        return self

class GatedLinearUnit(ComputationalPrimitive):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.linear = None
        self.gate = None
    
    def _init_layers(self, actual_input_dim):
        """Inizializza i layer con le dimensioni effettive dell'input"""
        if self.linear is None or self.linear.in_features != actual_input_dim:
            self.linear = nn.Linear(actual_input_dim, self.output_dim)
            self.gate = nn.Linear(actual_input_dim, self.output_dim)
            if hasattr(self, '_device'):
                self.linear = self.linear.to(self._device)
                self.gate = self.gate.to(self._device)
    
    def forward(self, x):
        actual_input_dim = x.shape[-1]
        self._init_layers(actual_input_dim)
        return self.linear(x) * torch.sigmoid(self.gate(x))
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.linear is not None:
            self.linear = self.linear.to(device)
            self.gate = self.gate.to(device)
        return self

class Activation(ComputationalPrimitive):
    def __init__(self, kind="relu"):
        super().__init__()
        self.kind = kind
    def forward(self, x):
        if self.kind == "relu":
            return F.relu(x)
        elif self.kind == "tanh":
            return torch.tanh(x)
        elif self.kind == "sigmoid":
            return torch.sigmoid(x)
        elif self.kind == "gelu":
            return F.gelu(x)
        else:
            raise ValueError(f"Unknown activation: {self.kind}")

class BatchNorm(ComputationalPrimitive):
    def __init__(self, feature_dim):
        super().__init__()
        self.input_dim = feature_dim
        self.output_dim = feature_dim
        self.norm = None
    
    def _init_norm(self, actual_feature_dim):
        """Initialize BatchNorm with actual input dimensions"""
        if self.norm is None or self.norm.num_features != actual_feature_dim:
            self.norm = nn.BatchNorm1d(actual_feature_dim)
            if hasattr(self, '_device'):
                self.norm = self.norm.to(self._device)
    
    def forward(self, x):
        # x: (batch, features) or (batch, seq, features)
        original_shape = x.shape
        if len(x.shape) == 3:  # (batch, seq, features)
            batch_size, seq_len, features = x.shape
            x = x.transpose(1, 2)  # (batch, features, seq)
            self._init_norm(features)
            x = self.norm(x)
            x = x.transpose(1, 2)  # (batch, seq, features)
        else:  # (batch, features)
            actual_feature_dim = x.shape[-1]
            self._init_norm(actual_feature_dim)
            x = self.norm(x)
        return x
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self.norm is not None:
            self.norm = self.norm.to(device)
        return self
