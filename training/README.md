# Uncertainty NNUE Training - POC

This directory contains scripts to train an uncertainty-aware NNUE that replaces hardcoded search heuristics with learned quantile predictions.

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Build the Engine

```bash
cd ..
cargo build --release
```

### 3. Generate Training Data

This plays self-play games and evaluates positions at multiple depths to create training targets.

```bash
# Generate ~10K positions for a quick test (takes ~10-15 minutes)
python generate_data.py --engine ../target/release/aleph --output training_data.jsonl --games 100 --positions 10000

# For full POC, generate 50K positions (takes ~1-2 hours)
python generate_data.py --engine ../target/release/aleph --output training_data.jsonl --games 500 --positions 50000
```

### 4. Train the Network

```bash
# Quick training (10 epochs, ~5 minutes)
python train.py --data training_data.jsonl --output small_nnue.bin --epochs 10

# Full POC training (100 epochs, ~30 minutes)
python train.py --data training_data.jsonl --output small_nnue.bin --epochs 100
```

### 5. Test the Trained Network

The trained network is saved in the Rust-compatible format. To use it:

```rust
use aleph::eval::nnue::{loader, DualNnueEvaluator};

let network = loader::load_network("training/small_nnue.bin")?;
let evaluator = DualNnueEvaluator::new(network);
```

## Architecture

### Uncertainty Quantiles

The network outputs 5 quantiles instead of a single score:
- **q10**: 10th percentile (pessimistic bound)
- **q25**: 25th percentile
- **q50**: Median (the "score")
- **q75**: 75th percentile
- **q90**: 90th percentile (optimistic bound)

These are **guaranteed ordered** via incremental parameterization:
```
q10 = base
q25 = base + softplus(δ1)
q50 = base + softplus(δ1) + softplus(δ2)
...
```

### Uncertainty-Based Pruning

The search uses quantiles instead of hardcoded heuristics:

```rust
// Hard cutoffs from bounds
if q10 >= beta { return q10; }  // Pessimistic beats beta
if q90 <= alpha { return q90; } // Optimistic can't reach alpha

// Depth adjustment based on uncertainty
let uncertainty = q90 - q10;
if uncertainty < 30 { depth -= 1; }  // Clear position
if uncertainty > 150 { depth += 1; } // Complex position
```

### Policy Head

Move ordering is learned from the best moves found during search:
- Outputs 128 values: 64 from-square scores + 64 to-square scores
- Move score = from[from_sq] + to[to_sq]
- Replaces MVV-LVA, killer moves, history heuristic

## Files

| File | Description |
|------|-------------|
| `generate_data.py` | Generates training data via self-play + multi-depth search |
| `train.py` | PyTorch training script with pinball loss |
| `features.py` | HalfKAv2 feature extraction (mirrors Rust code) |
| `requirements.txt` | Python dependencies |

## Expected Results

After training on 50K positions:

1. **Uncertainty correlation**: Positions with higher uncertainty should have more volatile evals across depths
2. **Policy quality**: Policy should order good moves first (comparable to MVV-LVA)
3. **Elo**: Should maintain ~2100+ Elo (baseline with classical eval)

## Next Steps

1. **Train larger network**: Increase L1 from 128 to 512 or 1024
2. **Dual-NNUE**: Use small network for most positions, large for uncertain ones
3. **More training data**: 1M+ positions for production quality
4. **SIMD optimization**: Add AVX2/NEON for faster inference
