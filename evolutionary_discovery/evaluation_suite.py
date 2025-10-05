import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Callable

# Reuse wrappers from genome module
from genome import Conv1DWrapper, GRUWrapper, SequenceToVector


def train_and_test_model(model: nn.Module,
                         train_loader: DataLoader,
                         test_loader: DataLoader,
                         task_type: str,
                         epochs: int = 20):
    """Train a model and evaluate on test set.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        test_loader: DataLoader for evaluation
        task_type: 'classification' or 'regression'
        epochs: training epochs
    Returns:
        float: final metric (accuracy for classification, MSE for regression)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            # Adjust shapes
            if task_type == 'classification':
                if yb.dim() > 1:
                    yb_t = yb.float().view(-1)
                else:
                    yb_t = yb.float()
                loss = criterion(out.view(-1), yb_t)
            else:  # regression
                # If target is flattened (batch, seq_len), ensure output matches
                if yb.dim() == 2:
                    # Project model output to same shape
                    out_vec = out.view(out.size(0), -1)
                    if out_vec.shape[1] != yb.shape[1]:
                        # Adjust with linear projection created on-the-fly (avoids architecture change)
                        proj = nn.Linear(out_vec.shape[1], yb.shape[1]).to(out_vec.device)
                        out_vec = proj(out_vec)
                    loss = criterion(out_vec, yb)
                else:
                    loss = criterion(out.view_as(yb), yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        total_loss = 0.0
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            if task_type == 'classification':
                logits = out.view(-1)
                yb_t = yb.view(-1).float()
                loss = criterion(logits, yb_t)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()
                preds.append(pred.cpu())
                targets.append(yb_t.cpu())
                total_loss += loss.item() * xb.size(0)
            else:  # regression
                if yb.dim() == 2:
                    out_vec = out.view(out.size(0), -1)
                    if out_vec.shape[1] != yb.shape[1]:
                        proj = nn.Linear(out_vec.shape[1], yb.shape[1]).to(out_vec.device)
                        out_vec = proj(out_vec)
                    loss = criterion(out_vec, yb)
                    total_loss += loss.item() * xb.size(0)
                    preds.append(out_vec.cpu())
                    targets.append(yb.cpu())
                else:
                    loss = criterion(out.view_as(yb), yb)
                    total_loss += loss.item() * xb.size(0)
                    preds.append(out.view_as(yb).cpu())
                    targets.append(yb.cpu())

        if task_type == 'classification':
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            accuracy = (preds == targets).float().mean().item()
            return accuracy
        else:
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            mse = nn.MSELoss()(preds, targets).item()
            return mse


# ---------------- Dataset Generators ----------------

def _split_dataset(X: torch.Tensor, y: torch.Tensor, batch_size: int, train_frac: float = 0.8):
    n = X.shape[0]
    idx = torch.randperm(n)
    train_n = int(n * train_frac)
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))

def get_parity_task(seq_len: int, num_samples: int = 1000, batch_size: int = 32):
    """Generate parity task (binary classification).
    
    Args:
        seq_len: Sequence length
        num_samples: Number of samples to generate
        batch_size: Batch size for DataLoaders
    
    Returns:
        tuple: (train_loader, test_loader) with parity classification data
    """
    X = torch.randint(0, 2, (num_samples, seq_len, 1)).float()
    y = (X.sum(dim=1) % 2).float()  # (num_samples, 1)
    return _split_dataset(X, y, batch_size)

def get_copy_task(seq_len: int, num_samples: int = 1000, batch_size: int = 32):
    """Generate sequence copy task (regression).
    
    Args:
        seq_len: Sequence length
        num_samples: Number of samples to generate
        batch_size: Batch size for DataLoaders
    
    Returns:
        tuple: (train_loader, test_loader) with sequence copy data
    """
    X = torch.rand(num_samples, seq_len, 1)
    # Flatten the sequence copy into a vector target for simpler regression (batch, seq_len)
    y = X.view(num_samples, seq_len)  # target shape (batch, seq_len)
    return _split_dataset(X, y, batch_size)

def get_pattern_task(seq_len: int = 40, num_samples: int = 1000, batch_size: int = 32):
    """Generate pattern detection task.
    
    Args:
        seq_len: Sequence length
        num_samples: Number of samples to generate
        batch_size: Batch size for DataLoaders
    
    Returns:
        tuple: (train_loader, test_loader) with pattern detection data
    """
    X = torch.zeros(num_samples, seq_len, 1)
    y = torch.zeros(num_samples, 1)
    pattern = torch.tensor([0.5, 1.0, 1.0, 0.5]).view(4, 1)
    half = num_samples // 2
    for i in range(half):
        start = np.random.randint(0, seq_len - len(pattern))
        X[i, start:start+len(pattern), 0] = pattern.view(-1)
        y[i] = 1.0
    return _split_dataset(X, y, batch_size)


# ---------------- Architectures ----------------

class MinimalistArch(nn.Module):
    """Conv1D -> Tanh -> LayerNorm -> Linear(1) with fixed hidden channels."""
    def __init__(self, in_channels: int = 1, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.conv = Conv1DWrapper(torch.nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1))
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (batch, seq, feat)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != 1:  # expect feature dim at last for wrapper; conv wrapper transposes if needed
            pass
        h = self.conv(x)  # (batch, hidden)
        h = self.act(h)
        h = self.norm(h)
        return self.linear(h)

class _GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        h = self.act(h)
        h = self.norm(h)
        return h

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv1DWrapper(torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        h = self.conv(x)
        h = self.act(h)
        h = self.norm(h)
        return h

class HybridDeepArch(nn.Module):
    """4 GRU residual blocks then 2 Conv1D residual blocks then Linear(1)."""
    def __init__(self, input_dim: int = 1, hidden: int = 32, conv_hidden: int = 32, out_dim: int = 1):
        super().__init__()
        self.gru_blocks = nn.ModuleList([_GRUBlock(input_dim if i == 0 else hidden, hidden) for i in range(4)])
        self.conv_blocks = nn.ModuleList([_ConvBlock(1 if i == 0 else conv_hidden, conv_hidden) for i in range(2)])
        self.linear = nn.Linear(conv_hidden, out_dim)

    def forward(self, x):
        # Ensure 3D input for GRU blocks
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Pass through GRU residual blocks
        h = x
        for block in self.gru_blocks:
            block_input = h
            out = block(h)  # (batch, hidden)
            # Expand residual input to match hidden if necessary
            if block_input.dim() == 3:
                block_input_vec = block_input[:, -1, :]
            else:
                block_input_vec = block_input
            if block_input_vec.shape[-1] != out.shape[-1]:
                # Project if mismatch
                proj = nn.Linear(block_input_vec.shape[-1], out.shape[-1]).to(out.device)
                block_input_vec = proj(block_input_vec)
            h = out + block_input_vec
        # After GRU blocks we have (batch, hidden) vector -> expand to (batch, hidden, 1) for conv wrapper
        h_seq = h.unsqueeze(-1)  # (batch, hidden, 1)
        # Reinterpret as (batch, seq_len=1, hidden) for conv blocks: conv wrapper will transpose if needed
        for block in self.conv_blocks:
            block_in = h_seq
            out = block(h_seq)  # returns (batch, conv_hidden)
            if block_in.shape[-2] == 1:  # block_in shape (batch, hidden, 1)
                base_vec = block_in.squeeze(-1)
            else:
                base_vec = block_in.mean(dim=-1)
            if base_vec.shape[-1] != out.shape[-1]:
                proj = nn.Linear(base_vec.shape[-1], out.shape[-1]).to(out.device)
                base_vec = proj(base_vec)
            h_seq = (out + base_vec).unsqueeze(-1)
        h_final = h_seq.squeeze(-1)
        return self.linear(h_final)

class GRUOnlyArch(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 32, out_dim: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.linear = nn.Linear(hidden, out_dim)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.linear(h)

class MLPOnlyArch(nn.Module):
    def __init__(self, input_dim: int = 1, seq_len: int = 12, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(seq_len * input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        if x.dim() == 3:
            b, s, f = x.shape
            x = x.view(b, s * f)
        return self.net(x)


if __name__ == "__main__":
    experiments = [
        {"name": "Parity (Short Seq)", "task_func": get_parity_task, "task_args": {"seq_len": 12}, "type": "classification"},
        {"name": "Parity (Long Seq)", "task_func": get_parity_task, "task_args": {"seq_len": 50}, "type": "classification"},
        {"name": "Copy Task", "task_func": get_copy_task, "task_args": {"seq_len": 12}, "type": "regression"},
        {"name": "Pattern Detection", "task_func": get_pattern_task, "task_args": {}, "type": "classification"},
    ]

    architectures = {
        "Evolved-Minimal": MinimalistArch,
        "Evolved-Hybrid": HybridDeepArch,
        "Baseline-GRU": GRUOnlyArch,
        "Baseline-MLP": MLPOnlyArch
    }

    results: Dict[str, Dict[str, float]] = {}

    for exp in experiments:
        print(f"\n--- Task: {exp['name']} ---")
        train_loader, test_loader = exp["task_func"](**exp["task_args"])  # returns loaders
        task_type = exp['type']
        results[exp['name']] = {}
        for arch_name, ArchClass in architectures.items():
            # Provide seq_len to MLP if needed
            if ArchClass is MLPOnlyArch:
                seq_len = exp['task_args'].get('seq_len', 12 if 'Parity' in exp['name'] else 40)
                model = ArchClass(seq_len=seq_len)
            else:
                model = ArchClass()
            metric = train_and_test_model(model, train_loader, test_loader, task_type=task_type, epochs=10)
            if task_type == 'classification':
                print(f"  - {arch_name}: Accuracy = {metric:.4f}")
            else:
                print(f"  - {arch_name}: MSE = {metric:.6f}")
            results[exp['name']][arch_name] = metric

    print("\n=== SUMMARY ===")
    for task_name, arch_metrics in results.items():
        print(f"\nTask: {task_name}")
        for arch_name, metric in arch_metrics.items():
            label = 'Acc' if 'Parity' in task_name or 'Pattern' in task_name else 'MSE'
            if label == 'Acc':
                print(f"  {arch_name:<18} : {metric:.4f}")
            else:
                print(f"  {arch_name:<18} : {metric:.6f}")
