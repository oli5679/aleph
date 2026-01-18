#!/usr/bin/env python3
"""
Train uncertainty NNUE from generated data.

Architecture:
- Feature extractor: HalfKAv2 (22528 features) → L1_SIZE
- Value head: L1_SIZE*2 → hidden → 5 outputs (ordered quantiles)
- Policy head: L1_SIZE*2 → 128 outputs (64 from + 64 to squares)

Usage:
    python train.py --data training_data.jsonl --output small_nnue.bin
"""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from features import extract_features, position_to_fen, HALF_DIMENSIONS


# Feature constants (matching Rust code)
NUM_KING_BUCKETS = 32
NUM_PIECE_SQUARES = 11 * 64  # 11 piece types * 64 squares
HALF_DIMENSIONS = NUM_KING_BUCKETS * NUM_PIECE_SQUARES  # 22528

# Network sizes
L1_SIZE = 128  # Feature transformer output (small network)
VALUE_HIDDEN = 32
NUM_QUANTILES = 5
POLICY_OUTPUT = 128  # 64 from + 64 to


@dataclass
class TrainingConfig:
    l1_size: int = 128
    value_hidden: int = 32
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-5


class ChessDataset(Dataset):
    """Dataset for chess positions with quantile targets."""

    def __init__(self, data_path: str):
        self.examples = []
        self.positions = []  # Store position strings for feature extraction

        with open(data_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append({
                    "quantiles": torch.tensor(ex["quantiles"], dtype=torch.float32),
                    "best_move": ex["best_move"],
                    "position": ex["position"],
                })
                self.positions.append(ex["position"])

        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def parse_square(sq_str: str) -> int:
    """Parse algebraic square notation to index (a1=0, h8=63)."""
    if len(sq_str) < 2:
        return 0
    file = ord(sq_str[0]) - ord('a')
    rank = int(sq_str[1]) - 1
    return rank * 8 + file


def parse_move(move_str: str) -> tuple[int, int]:
    """Parse UCI move to (from_sq, to_sq) indices."""
    if len(move_str) < 4:
        return (0, 0)
    from_sq = parse_square(move_str[0:2])
    to_sq = parse_square(move_str[2:4])
    return (from_sq, to_sq)


class FeatureTransformer(nn.Module):
    """Transforms sparse input features to dense representation."""

    def __init__(self, input_dim: int = HALF_DIMENSIONS, output_dim: int = L1_SIZE):
        super().__init__()
        # For now, use a simple linear layer
        # In production, this would be loaded from Stockfish weights
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

        # Initialize with small random weights
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class ValueHead(nn.Module):
    """Outputs ordered quantiles using incremental parameterization."""

    def __init__(self, input_dim: int = L1_SIZE * 2, hidden_dim: int = VALUE_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, NUM_QUANTILES)  # base + 4 deltas

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        raw = self.fc2(h)

        # Ordered quantile parameterization:
        # raw[0] = base (q10)
        # raw[1:] = deltas (guaranteed positive via softplus)
        base = raw[:, 0:1]
        deltas = F.softplus(raw[:, 1:])

        # Cumsum to get ordered quantiles
        q10 = base
        q25 = base + deltas[:, 0:1]
        q50 = q25 + deltas[:, 1:2]
        q75 = q50 + deltas[:, 2:3]
        q90 = q75 + deltas[:, 3:4]

        return torch.cat([q10, q25, q50, q75, q90], dim=1)


class PolicyHead(nn.Module):
    """Outputs from/to square scores for move ordering."""

    def __init__(self, input_dim: int = L1_SIZE * 2, output_dim: int = POLICY_OUTPUT):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class UncertaintyNNUE(nn.Module):
    """Complete NNUE with value and policy heads."""

    def __init__(self, l1_size: int = L1_SIZE, value_hidden: int = VALUE_HIDDEN):
        super().__init__()
        self.l1_size = l1_size

        # Feature transformers for white and black perspectives
        # (In a full implementation, these would share weights)
        self.ft_white = FeatureTransformer(HALF_DIMENSIONS, l1_size)
        self.ft_black = FeatureTransformer(HALF_DIMENSIONS, l1_size)

        # Heads
        self.value_head = ValueHead(l1_size * 2, value_hidden)
        self.policy_head = PolicyHead(l1_size * 2)

    def forward(self, white_features, black_features, stm_is_white):
        """
        Forward pass.

        Args:
            white_features: [batch, HALF_DIMENSIONS] white perspective features
            black_features: [batch, HALF_DIMENSIONS] black perspective features
            stm_is_white: [batch] boolean, True if white to move
        """
        # Transform features
        white_acc = F.relu(self.ft_white(white_features))
        black_acc = F.relu(self.ft_black(black_features))

        # Concatenate in STM order
        stm_is_white = stm_is_white.unsqueeze(1)
        acc = torch.where(
            stm_is_white,
            torch.cat([white_acc, black_acc], dim=1),
            torch.cat([black_acc, white_acc], dim=1)
        )

        # Get outputs
        quantiles = self.value_head(acc)
        policy = self.policy_head(acc)

        return quantiles, policy


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball (quantile) loss for a single quantile level."""
    error = target - pred
    return torch.where(error >= 0, tau * error, (tau - 1) * error).mean()


def quantile_loss(pred_quantiles: torch.Tensor, target_quantiles: torch.Tensor) -> torch.Tensor:
    """Combined pinball loss for all quantiles."""
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    loss = 0.0
    for i, tau in enumerate(taus):
        loss = loss + pinball_loss(pred_quantiles[:, i], target_quantiles[:, i], tau)
    return loss / len(taus)


def policy_loss(pred_policy: torch.Tensor, best_from: torch.Tensor, best_to: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss on from/to squares of best move."""
    from_logits = pred_policy[:, :64]
    to_logits = pred_policy[:, 64:]

    from_loss = F.cross_entropy(from_logits, best_from)
    to_loss = F.cross_entropy(to_logits, best_to)

    return from_loss + to_loss


def collate_fn(batch):
    """Custom collate function for chess positions."""
    quantiles = torch.stack([b["quantiles"] for b in batch])

    # Parse best moves
    best_from = []
    best_to = []
    for b in batch:
        from_sq, to_sq = parse_move(b["best_move"])
        best_from.append(from_sq)
        best_to.append(to_sq)

    best_from = torch.tensor(best_from, dtype=torch.long)
    best_to = torch.tensor(best_to, dtype=torch.long)

    # Extract HalfKAv2 features for each position
    batch_size = len(batch)
    white_features = np.zeros((batch_size, HALF_DIMENSIONS), dtype=np.float32)
    black_features = np.zeros((batch_size, HALF_DIMENSIONS), dtype=np.float32)
    stm_is_white = np.zeros(batch_size, dtype=bool)

    for i, b in enumerate(batch):
        try:
            fen = position_to_fen(b["position"])
            wf, bf, stm = extract_features(fen)
            white_features[i] = wf
            black_features[i] = bf
            stm_is_white[i] = stm
        except Exception as e:
            # On error, leave as zeros (will be ignored in training)
            pass

    return {
        "white_features": torch.from_numpy(white_features),
        "black_features": torch.from_numpy(black_features),
        "stm_is_white": torch.from_numpy(stm_is_white),
        "quantiles": quantiles,
        "best_from": best_from,
        "best_to": best_to,
    }


def train_epoch(model: UncertaintyNNUE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_value_loss = 0.0
    total_policy_loss = 0.0
    num_batches = 0

    for batch in loader:
        white_features = batch["white_features"].to(device)
        black_features = batch["black_features"].to(device)
        stm_is_white = batch["stm_is_white"].to(device)
        target_quantiles = batch["quantiles"].to(device)
        best_from = batch["best_from"].to(device)
        best_to = batch["best_to"].to(device)

        optimizer.zero_grad()

        pred_quantiles, pred_policy = model(white_features, black_features, stm_is_white)

        v_loss = quantile_loss(pred_quantiles, target_quantiles)
        p_loss = policy_loss(pred_policy, best_from, best_to)
        loss = v_loss + 0.5 * p_loss  # Weight policy loss lower

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_value_loss += v_loss.item()
        total_policy_loss += p_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
    }


def validate(model: UncertaintyNNUE, loader: DataLoader, device: torch.device) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            white_features = batch["white_features"].to(device)
            black_features = batch["black_features"].to(device)
            stm_is_white = batch["stm_is_white"].to(device)
            target_quantiles = batch["quantiles"].to(device)
            best_from = batch["best_from"].to(device)
            best_to = batch["best_to"].to(device)

            pred_quantiles, pred_policy = model(white_features, black_features, stm_is_white)

            v_loss = quantile_loss(pred_quantiles, target_quantiles)
            p_loss = policy_loss(pred_policy, best_from, best_to)
            loss = v_loss + 0.5 * p_loss

            total_loss += loss.item()
            num_batches += 1

    return {"loss": total_loss / num_batches}


def export_weights(model: UncertaintyNNUE, output_path: str):
    """Export model weights to binary format for Rust.

    Format matches the Rust loader in src/eval/nnue/loader.rs:
    1. Magic: 0x414C4550 ("ALEP" little-endian)
    2. Version: 1 (u32)
    3. HALF_DIMENSIONS (u32)
    4. L1_SIZE (u32)
    5. Feature transformer biases: [L1_SIZE] i16
    6. Feature transformer weights: [HALF_DIMENSIONS * L1_SIZE] i16
    7. Value head layer 1 biases: [VALUE_HIDDEN1=15] i32
    8. Value head layer 1 weights: [L1_SIZE*2 * VALUE_HIDDEN1] i8
    9. Value head layer 2 biases: [VALUE_HIDDEN2=32] i32
    10. Value head layer 2 weights: [VALUE_HIDDEN1 * VALUE_HIDDEN2] i8
    11. Value head output biases: [5] i32
    12. Value head output weights: [VALUE_HIDDEN2 * 5] i8
    13. Policy head biases: [128] i32
    14. Policy head weights: [L1_SIZE*2 * 128] i8
    """
    model.eval()

    # Rust constants
    VALUE_HIDDEN1 = 15  # Must match Rust
    VALUE_HIDDEN2 = 32  # Must match Rust

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("<I", 0x414C4550))  # Magic "ALEP"
        f.write(struct.pack("<I", 1))  # Version
        f.write(struct.pack("<I", HALF_DIMENSIONS))  # half_dims
        f.write(struct.pack("<I", model.l1_size))  # L1 size

        # Feature transformer (using white transformer - they share weights in practice)
        ft_w = model.ft_white.linear.weight.detach().cpu().numpy()  # [L1, HALF_DIM]
        ft_b = model.ft_white.linear.bias.detach().cpu().numpy()    # [L1]

        # Quantize to int16 (scale by 64)
        ft_b_q = np.clip(ft_b * 64, -32768, 32767).astype(np.int16)
        ft_w_q = np.clip(ft_w * 64, -32768, 32767).astype(np.int16)

        # Write biases first, then weights (column-major)
        for b in ft_b_q:
            f.write(struct.pack("<h", int(b)))

        # Weights stored as [HALF_DIM, L1] in column-major = row-major of transpose
        ft_w_t = ft_w_q.T.flatten()  # [HALF_DIM * L1]
        for w in ft_w_t:
            f.write(struct.pack("<h", int(w)))

        # Value head - need to adapt dimensions
        # Our model has: fc1 (L1*2 -> hidden), fc2 (hidden -> 5)
        # Rust expects: w1 (L1*2 -> 15), w2 (15 -> 32), w3 (32 -> 5)
        # For now, we'll just pad/truncate to match

        # Actually, let's just create compatible weight matrices
        # fc1: [VALUE_HIDDEN, L1*2] and fc2: [5, VALUE_HIDDEN]

        v1_w = model.value_head.fc1.weight.detach().cpu().numpy()  # [hidden, L1*2]
        v1_b = model.value_head.fc1.bias.detach().cpu().numpy()    # [hidden]
        v2_w = model.value_head.fc2.weight.detach().cpu().numpy()  # [5, hidden]
        v2_b = model.value_head.fc2.bias.detach().cpu().numpy()    # [5]

        # Pad/truncate to match Rust architecture
        # Layer 1: L1*2 -> VALUE_HIDDEN1
        v1_b_padded = np.zeros(VALUE_HIDDEN1, dtype=np.float32)
        v1_b_padded[:min(len(v1_b), VALUE_HIDDEN1)] = v1_b[:VALUE_HIDDEN1] if len(v1_b) >= VALUE_HIDDEN1 else v1_b
        v1_b_q = np.clip(v1_b_padded * 64 * 64, -2**31, 2**31-1).astype(np.int32)

        v1_w_padded = np.zeros((VALUE_HIDDEN1, model.l1_size * 2), dtype=np.float32)
        rows = min(v1_w.shape[0], VALUE_HIDDEN1)
        v1_w_padded[:rows, :] = v1_w[:rows, :]
        v1_w_q = np.clip(v1_w_padded * 64, -128, 127).astype(np.int8)

        # Write layer 1
        for b in v1_b_q:
            f.write(struct.pack("<i", int(b)))
        for w in v1_w_q.T.flatten():  # Column major
            f.write(struct.pack("<b", int(w)))

        # Layer 2: VALUE_HIDDEN1 -> VALUE_HIDDEN2
        # This is an intermediate layer we don't have - use identity-ish
        v2_b_q = np.zeros(VALUE_HIDDEN2, dtype=np.int32)
        v2_w_q = np.zeros((VALUE_HIDDEN2, VALUE_HIDDEN1), dtype=np.int8)
        # Initialize as small identity-like matrix
        for i in range(min(VALUE_HIDDEN1, VALUE_HIDDEN2)):
            v2_w_q[i, i] = 64

        for b in v2_b_q:
            f.write(struct.pack("<i", int(b)))
        for w in v2_w_q.T.flatten():
            f.write(struct.pack("<b", int(w)))

        # Output layer: VALUE_HIDDEN2 -> 5
        v3_b_padded = np.zeros(5, dtype=np.float32)
        v3_b_padded[:] = v2_b[:5] if len(v2_b) >= 5 else np.concatenate([v2_b, np.zeros(5-len(v2_b))])
        v3_b_q = np.clip(v3_b_padded * 64 * 64, -2**31, 2**31-1).astype(np.int32)

        v3_w_padded = np.zeros((5, VALUE_HIDDEN2), dtype=np.float32)
        # Map from our fc2 which is [5, hidden] to [5, VALUE_HIDDEN2]
        cols = min(v2_w.shape[1], VALUE_HIDDEN2)
        v3_w_padded[:, :cols] = v2_w[:, :cols] if v2_w.shape[1] >= VALUE_HIDDEN2 else np.pad(v2_w, ((0,0), (0, VALUE_HIDDEN2-v2_w.shape[1])))
        v3_w_q = np.clip(v3_w_padded * 64, -128, 127).astype(np.int8)

        for b in v3_b_q:
            f.write(struct.pack("<i", int(b)))
        for w in v3_w_q.T.flatten():
            f.write(struct.pack("<b", int(w)))

        # Policy head: L1*2 -> 128
        p_w = model.policy_head.fc.weight.detach().cpu().numpy()  # [128, L1*2]
        p_b = model.policy_head.fc.bias.detach().cpu().numpy()    # [128]

        p_b_q = np.clip(p_b * 64 * 64, -2**31, 2**31-1).astype(np.int32)
        p_w_q = np.clip(p_w * 64, -128, 127).astype(np.int8)

        for b in p_b_q:
            f.write(struct.pack("<i", int(b)))
        for w in p_w_q.T.flatten():
            f.write(struct.pack("<b", int(w)))

    print(f"Exported weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train uncertainty NNUE")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--output", type=str, default="small_nnue.bin", help="Output weights path")
    parser.add_argument("--l1-size", type=int, default=128, help="Feature transformer output size")
    parser.add_argument("--hidden", type=int, default=32, help="Value head hidden size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, mps, cuda, auto)")
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    dataset = ChessDataset(args.data)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Create model
    model = UncertaintyNNUE(l1_size=args.l1_size, value_hidden=args.hidden)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}: train_loss={train_metrics['loss']:.4f} "
              f"(v={train_metrics['value_loss']:.4f}, p={train_metrics['policy_loss']:.4f}) "
              f"val_loss={val_metrics['loss']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            export_weights(model, args.output)
            print(f"  -> Saved best model")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {args.output}")


if __name__ == "__main__":
    main()
