# Uncertainty-Based NNUE: Implementation Plan

## Core Idea

Replace hardcoded search heuristics (LMR, killer moves, futility pruning, etc.) with **learned uncertainty estimates** from a neural network. The network outputs a distribution of values, and pruning decisions emerge from the bounds of that distribution.

## Architecture

### Dual-NNUE with Uncertainty Gating

```
Position
    │
    ▼
┌─────────────────────┐
│  Small NNUE         │  Fast (~10μs)
│  22528 → 128 → heads│
│  Outputs:           │
│  - 5 quantiles      │
│  - policy scores    │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ uncertainty  │───── LOW ────► Use small NNUE result
    │  q90-q10 > θ │
    └──────┬───────┘
           │
          HIGH
           │
           ▼
    ┌─────────────────────┐
    │  Large NNUE         │  Slower (~100μs) but accurate
    │  22528 → 512 → heads│
    │  Outputs:           │
    │  - 5 quantiles      │
    │  - policy scores    │
    └─────────────────────┘
```

### Ordered Quantile Outputs

To ensure quantiles are properly ordered (q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90):

```
Network raw outputs: [base, δ1, δ2, δ3, δ4]

q10 = base
q25 = base + softplus(δ1)
q50 = base + softplus(δ1) + softplus(δ2)
q75 = base + softplus(δ1) + softplus(δ2) + softplus(δ3)
q90 = base + softplus(δ1) + softplus(δ2) + softplus(δ3) + softplus(δ4)
```

This guarantees ordering by construction.

### Policy Head

Outputs from/to square scores. Move score = from[from_sq] + to[to_sq].
Replaces MVV-LVA, killer moves, history heuristic, etc.

## Training

### Data Generation

For each position P:
1. Run search at depths 4, 8, 12
2. Record eval at each depth
3. Compute target quantiles from the distribution of evals
4. Record best move from deepest search for policy training

```python
# Pseudo-code for data generation
for game in games:
    for position in game.positions:
        evals = [search(position, depth=d) for d in [4, 8, 12]]
        quantiles = compute_quantiles(evals, [0.1, 0.25, 0.5, 0.75, 0.9])
        best_move = search(position, depth=12).best_move
        save(position.fen, quantiles, best_move)
```

### Loss Functions

**Value head (Pinball loss for quantile regression):**
```python
def pinball_loss(pred, target, tau):
    error = target - pred
    return torch.where(error >= 0, tau * error, (tau - 1) * error)

loss = sum(pinball_loss(q[i], target[i], tau[i]) for i, tau in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]))
```

**Policy head (Cross-entropy on best move):**
```python
from_logits = policy[:64]
to_logits = policy[64:]
loss = F.cross_entropy(from_logits, best_from) + F.cross_entropy(to_logits, best_to)
```

### Training Procedure

1. Load Stockfish feature transformer weights (frozen initially)
2. Train value + policy heads on generated data
3. Optionally fine-tune feature transformer later

## Search Modifications

### Uncertainty-Based Pruning

```rust
fn alpha_beta(pos: &Position, depth: i32, alpha: i32, beta: i32) -> i32 {
    let q = evaluator.evaluate(pos);

    // Hard cutoffs from bounds
    if q.q10 >= beta {
        return q.q10;  // Even pessimistic bound beats beta
    }
    if q.q90 <= alpha {
        return q.q90;  // Even optimistic bound can't reach alpha
    }

    // Uncertainty-based depth adjustment (replaces LMR)
    let uncertainty = q.q90 - q.q10;
    let adjusted_depth = match uncertainty {
        u if u < 50 => depth - 1,   // Clear position, reduce
        u if u > 200 => depth + 1,  // Complex position, extend
        _ => depth,
    };

    // ... rest of search
}
```

### Policy-Based Move Ordering

```rust
fn order_moves(moves: &mut MoveList, policy: &PolicyOutput) {
    for mv in moves {
        mv.score = policy.from[mv.from()] + policy.to[mv.to()];
    }
    moves.sort_by_score();
}
```

No MVV-LVA, no killers, no history - just learned policy.

## POC Scope (MacBook)

### Phase 1: Data Generation (~1 hour)
- 50K positions from self-play
- Depths 4, 8, 12
- Output: training.jsonl (~50MB)

### Phase 2: Training (~30 min)
- Small network: 22528 → 128 → 32 → 5 (value), 128 → 128 (policy)
- ~100 epochs on 50K positions
- Output: small_nnue.bin (~1MB)

### Phase 3: Integration
- Load trained weights
- Modify search for quantile pruning
- Benchmark uncertainty correlation
- Benchmark Elo

### Phase 4: Scale Up (if POC works)
- Large network: 22528 → 512 → 128 → 5 (value)
- Uncertainty gating between small/large
- More training data (1M+ positions)

## Files to Create/Modify

### New Files
- `training/generate_data.py` - Data generation script
- `training/train.py` - PyTorch training script
- `training/export_weights.py` - Export to Rust format
- `training/requirements.txt` - Python dependencies

### Modified Files
- `src/eval/nnue/network.rs` - Ordered quantile outputs
- `src/eval/nnue/mod.rs` - Dual evaluator with gating
- `src/search.rs` - Uncertainty-based pruning, policy ordering

## Success Criteria

1. **Uncertainty correlation**: High uncertainty should correlate with eval volatility across depths
2. **Policy quality**: Policy ordering should produce similar or better move ordering than MVV-LVA
3. **Elo improvement**: Should match or exceed classical eval (~2100 ELO baseline)
4. **Inference speed**: Small NNUE should be <50μs per eval
