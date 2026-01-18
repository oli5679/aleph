# NNUE Integration Analysis for Aleph Chess Engine

## Executive Summary

This document analyzes how to integrate NNUE neural network evaluation into Aleph, with a focus on:
1. Loading and using Stockfish's pre-trained NNUE
2. Transfer learning to predict evaluation quantiles
3. Integration with the existing architecture
4. Efficiency considerations for multiple evaluations

**Key insight**: Aleph's board representation is already largely compatible with Stockfish's NNUE feature encoding. With minor additions, we can directly load Stockfish networks and use transfer learning to fine-tune them for quantile prediction.

---

## 1. Loading Stockfish's NNUE

### 1.1 Stockfish's Current Architecture (SFNNv10)

Stockfish uses a **dual network** architecture:
- **Big Network** (1024 L1): For balanced positions
- **Small Network** (128 L1): For positions with large material imbalances

Feature set: **HalfKAv2_hm** (with optional Full_Threats)
- 11 piece types × 64 squares × 32 king buckets = 22,528 features per perspective
- Board horizontally mirrored so king is always on files e-h

### 1.2 Feature Index Computation

From `half_ka_v2_hm.cpp`, the feature index formula:

```cpp
IndexType make_index(Color perspective, Square s, Piece pc, Square ksq) {
    const IndexType flip = 56 * perspective;  // Rank flip for black
    return (s ^ OrientTBL[ksq] ^ flip)         // Mirrored square
         + PieceSquareIndex[perspective][pc]   // Piece type offset
         + KingBuckets[ksq ^ flip];            // King bucket offset
}
```

**Mapping to Aleph types**:

| Stockfish | Aleph | Compatibility |
|-----------|-------|---------------|
| `Square` (0-63, a1=0) | `Square(u8)` (0-63, a1=0) | ✅ Identical |
| `Color` (WHITE=0, BLACK=1) | `Color::White=0, Black=1` | ✅ Identical |
| `Piece` (PAWN..KING) | `Piece::Pawn=0..King=5` | ⚠️ Need mapping |

**Piece mapping**: Stockfish uses 16-entry piece enum with color embedded. Aleph uses 6 piece types + separate color. The `PieceSquareIndex` table handles the conversion.

### 1.3 Recommended Feature Extractor

```rust
// src/nnue/features.rs

/// HalfKAv2_hm feature set compatible with Stockfish
pub struct HalfKAv2 {
    // Constants from Stockfish
    const PS_NB: usize = 11 * 64;  // 704
    const DIMENSIONS: usize = 64 * Self::PS_NB / 2;  // 22,528
}

impl HalfKAv2 {
    /// Piece-square index offsets (relative to perspective)
    const PIECE_SQUARE_INDEX: [[usize; 6]; 2] = [
        // White perspective: our pieces use W_*, their pieces use B_*
        [0, 2*64, 4*64, 6*64, 8*64, 10*64],   // Our P,N,B,R,Q,K
        [1*64, 3*64, 5*64, 7*64, 9*64, 10*64], // Their P,N,B,R,Q,K
    ];

    /// King bucket table (32 buckets, mirrored horizontally)
    const KING_BUCKETS: [usize; 64] = [
        28*704, 29*704, 30*704, 31*704, 31*704, 30*704, 29*704, 28*704,
        24*704, 25*704, 26*704, 27*704, 27*704, 26*704, 25*704, 24*704,
        20*704, 21*704, 22*704, 23*704, 23*704, 22*704, 21*704, 20*704,
        16*704, 17*704, 18*704, 19*704, 19*704, 18*704, 17*704, 16*704,
        12*704, 13*704, 14*704, 15*704, 15*704, 14*704, 13*704, 12*704,
         8*704,  9*704, 10*704, 11*704, 11*704, 10*704,  9*704,  8*704,
         4*704,  5*704,  6*704,  7*704,  7*704,  6*704,  5*704,  4*704,
         0*704,  1*704,  2*704,  3*704,  3*704,  2*704,  1*704,  0*704,
    ];

    /// Orient table: XOR value to mirror square horizontally if king on a-d files
    const ORIENT_TBL: [u8; 64] = [
        7, 7, 7, 7, 0, 0, 0, 0,  // Rank 1
        7, 7, 7, 7, 0, 0, 0, 0,  // Rank 2
        7, 7, 7, 7, 0, 0, 0, 0,  // ...
        7, 7, 7, 7, 0, 0, 0, 0,
        7, 7, 7, 7, 0, 0, 0, 0,
        7, 7, 7, 7, 0, 0, 0, 0,
        7, 7, 7, 7, 0, 0, 0, 0,
        7, 7, 7, 7, 0, 0, 0, 0,
    ];

    /// Compute feature index for a piece
    pub fn make_index(
        perspective: Color,
        piece_sq: Square,
        piece: Piece,
        piece_color: Color,
        king_sq: Square,
    ) -> usize {
        let flip = if perspective == Color::Black { 56 } else { 0 };
        let ksq = king_sq.0 as usize ^ flip;

        // Is this piece ours or theirs?
        let is_ours = piece_color == perspective;
        let piece_idx = if is_ours { 0 } else { 1 };

        let sq = (piece_sq.0 as usize) ^ (Self::ORIENT_TBL[ksq] as usize) ^ flip;

        sq + Self::PIECE_SQUARE_INDEX[piece_idx][piece.index()]
           + Self::KING_BUCKETS[ksq]
    }

    /// Get all active feature indices for a position
    pub fn get_active_features(pos: &Position, perspective: Color) -> Vec<usize> {
        let king_sq = pos.king_sq(perspective);
        let mut features = Vec::with_capacity(32);

        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                for sq in pos.pieces(color, piece) {
                    features.push(Self::make_index(
                        perspective, sq, piece, color, king_sq
                    ));
                }
            }
        }
        features
    }
}
```

### 1.4 Network Loading (.nnue format)

The `.nnue` file format:

```
[Header]
  - Version: u32 (0x7AF32F20)
  - Hash: u32 (architecture hash)
  - Description: length-prefixed UTF-8 string

[Feature Transformer]
  - Feature set hash: u32
  - Biases: i16[L1] (LEB128 compressed)
  - Weights: i16[features × L1] (LEB128 compressed)
  - PSQT weights: i32[features × 8] (for quick material eval)

[Layer Stack × 8 buckets]
  - FC_0: bias i32[L2], weights i8[L1 × L2]
  - FC_1: bias i32[L3], weights i8[L2×2 × L3]
  - FC_2: bias i32[1], weights i8[L3 × 1]
```

**LEB128 decoding** is required for the feature transformer. See `nnue_feature_transformer.h:read_leb_128()`.

---

## 2. Training Quantile NNUE via Transfer Learning

### 2.1 The Quantile Prediction Approach

**Current Stockfish NNUE**: Predicts a single evaluation E (mean/point estimate)

**Quantile NNUE for Aleph**: Predicts 5 quantiles [q10, q25, q50, q75, q90]

The key insight: The feature transformer (sparse input → L1) contains most of the chess knowledge. We can:
1. **Freeze** the feature transformer weights from Stockfish
2. **Replace** the output layers to predict 5 quantiles instead of 1 value
3. **Fine-tune** on data with quantile labels

### 2.2 Data Generation Strategy

Generate training data with Stockfish at multiple depths:

```bash
# Position → evaluate at depths 1, 4, 8, 12, 16, 20
# The distribution of evaluations across depths approximates true outcome uncertainty
```

For each position P:
1. Run Stockfish `go depth 1` → score s1
2. Run Stockfish `go depth 4` → score s4
3. ...
4. Run Stockfish `go depth 20` → score s20

The empirical distribution {s1, s4, s8, s12, s16, s20} provides quantile targets.

**Alternative**: Use game outcomes (WDL) combined with shallow search scores to learn uncertainty.

### 2.3 Modified Network Architecture

```
                        Stockfish NNUE             Quantile NNUE
                        ─────────────              ─────────────
Input Features    →     [sparse, ~22K]       →     [sparse, ~22K]
                              ↓                          ↓
Feature Transform →     [L1 = 1024] ←────────────→ [L1 = 1024]  (FROZEN)
                              ↓                          ↓
FC_0              →     [L2 = 15]            →     [L2 = 32] (retrained)
                              ↓                          ↓
FC_1              →     [L3 = 32]            →     [L3 = 64] (retrained)
                              ↓                          ↓
Output            →     [1 value]            →     [5 quantiles]
```

### 2.4 Loss Function: Quantile Regression

Standard quantile regression loss (pinball loss):

```python
def quantile_loss(pred, target, tau):
    """
    pred: predicted quantile value
    target: actual value
    tau: quantile level (e.g., 0.1 for q10)
    """
    error = target - pred
    return torch.max(tau * error, (tau - 1) * error)

def multi_quantile_loss(preds, targets, taus=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    preds: [batch, 5] predicted quantiles
    targets: [batch] actual deep search values
    """
    loss = 0
    for i, tau in enumerate(taus):
        loss += quantile_loss(preds[:, i], targets, tau)
    return loss.mean()
```

### 2.5 Training Pipeline (using nnue-pytorch as base)

```python
# In nnue-pytorch, modify model.py:

class QuantileNNUE(nn.Module):
    def __init__(self, stockfish_nnue_path):
        super().__init__()

        # Load Stockfish feature transformer (freeze it)
        self.feature_transformer = load_stockfish_ft(stockfish_nnue_path)
        for param in self.feature_transformer.parameters():
            param.requires_grad = False

        # New output layers for quantile prediction
        self.fc0 = nn.Linear(1024, 32)
        self.fc1 = nn.Linear(64, 64)  # 64 = 32 white + 32 black perspective
        self.fc_out = nn.Linear(64, 5)  # 5 quantiles

    def forward(self, white_features, black_features):
        # Accumulator from frozen feature transformer
        white_acc = self.feature_transformer(white_features)
        black_acc = self.feature_transformer(black_features)

        # Perspective combination
        x = torch.cat([
            self.fc0(white_acc),
            self.fc0(black_acc)
        ], dim=-1)

        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        quantiles = self.fc_out(x)

        # Ensure monotonicity: q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
        # Use cumulative softplus
        base = quantiles[:, 0:1]
        deltas = F.softplus(quantiles[:, 1:])
        quantiles = torch.cat([base, base + deltas.cumsum(dim=-1)], dim=-1)

        return quantiles
```

### 2.6 Data Format for Training

```python
# Training data record:
{
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "scores": [45, 42, 38, 35, 33, 32],  # depth 1, 4, 8, 12, 16, 20
    "outcome": 0.5,  # game result: 1=white win, 0=black win, 0.5=draw
}

# Compute quantiles from scores distribution:
q10 = np.percentile(scores, 10)
q25 = np.percentile(scores, 25)
# ... etc
```

---

## 3. Integration with Aleph Architecture

### 3.1 Current Evaluation Trait

Aleph already has the perfect abstraction:

```rust
// src/eval/mod.rs
pub struct Quantiles {
    pub q10: i16,
    pub q25: i16,
    pub q50: i16,
    pub q75: i16,
    pub q90: i16,
}

pub trait Evaluator {
    fn evaluate(&self, pos: &Position) -> Quantiles;
}
```

### 3.2 NNUE Evaluator Implementation

```rust
// src/eval/nnue.rs

pub struct NNUEEvaluator {
    // Feature transformer weights (frozen from Stockfish)
    ft_weights: Box<[[i16; L1]; FEATURES]>,  // ~45MB for L1=1024
    ft_biases: Box<[i16; L1]>,

    // Quantile output layers
    fc0_weights: Box<[[i8; 32]; L1]>,
    fc0_biases: Box<[i32; 32]>,
    fc1_weights: Box<[[i8; 64]; 64]>,
    fc1_biases: Box<[i32; 64]>,
    fc_out_weights: Box<[[i8; 5]; 64]>,
    fc_out_biases: Box<[i32; 5]>,
}

impl Evaluator for NNUEEvaluator {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        // Get active features for both perspectives
        let white_features = HalfKAv2::get_active_features(pos, Color::White);
        let black_features = HalfKAv2::get_active_features(pos, Color::Black);

        // Compute accumulators (sparse → dense)
        let white_acc = self.compute_accumulator(&white_features);
        let black_acc = self.compute_accumulator(&black_features);

        // Forward through quantile layers
        let quantiles = self.forward_quantile_layers(&white_acc, &black_acc);

        // Return from side-to-move perspective
        if pos.side_to_move() == Color::White {
            quantiles
        } else {
            quantiles.negate()
        }
    }
}
```

### 3.3 Incremental Accumulator Updates

For efficiency during search, maintain accumulators on the search stack:

```rust
// In search.rs
struct SearchStack {
    accumulator: [Accumulator; 2],  // [white, black]
    // ...
}

impl NNUEEvaluator {
    /// Update accumulator incrementally after a move
    fn update_accumulator(
        &self,
        acc: &mut Accumulator,
        perspective: Color,
        removed: &[usize],  // features removed
        added: &[usize],    // features added
    ) {
        for &idx in removed {
            for i in 0..L1 {
                acc.values[i] -= self.ft_weights[idx][i];
            }
        }
        for &idx in added {
            for i in 0..L1 {
                acc.values[i] += self.ft_weights[idx][i];
            }
        }
    }

    /// Refresh accumulator completely (after king moves)
    fn refresh_accumulator(
        &self,
        acc: &mut Accumulator,
        features: &[usize],
    ) {
        acc.values.copy_from_slice(&self.ft_biases);
        for &idx in features {
            for i in 0..L1 {
                acc.values[i] += self.ft_weights[idx][i];
            }
        }
    }
}
```

### 3.4 Zero Code Changes to Search

The beauty of Aleph's design: the search code doesn't change at all!

```rust
// In search.rs - this already works
let q = self.eval.evaluate(&self.pos);

if q.pessimistic_cutoff(beta) {
    return beta;  // Safe cutoff: even pessimistic outcome beats beta
}

if q.optimistic_cutoff(alpha) {
    return alpha;  // Safe cutoff: even optimistic outcome below alpha
}

// Use median as primary score
let score = q.q50;
```

---

## 4. Efficiency Considerations

### 4.1 Single Evaluation Performance

| Component | Operations | Time (approx) |
|-----------|-----------|---------------|
| Feature extraction | O(32) sparse lookups | 50ns |
| Accumulator (full) | 22K × 1024 additions | 5μs |
| Accumulator (incremental) | 2-4 × 1024 additions | 100ns |
| Output layers | 1024×32 + 64×64 + 64×5 | 500ns |
| **Total (incremental)** | | **~1μs** |

### 4.2 Multiple Quantile Outputs

**Key optimization**: The feature transformer (expensive part) is shared across all quantile outputs. Computing 5 quantiles costs almost the same as computing 1 value.

```
Stockfish single eval:   FT → L1 → L2 → L3 → 1 output     ~1μs
Aleph 5 quantiles:       FT → L1 → L2 → L3 → 5 outputs    ~1.1μs (only 10% more)
```

### 4.3 SIMD Optimization

Critical for performance. From Stockfish's SIMD code:

```rust
// Use portable_simd or explicit intrinsics
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Vectorized accumulator update (AVX2)
#[target_feature(enable = "avx2")]
unsafe fn update_accumulator_avx2(
    acc: &mut [i16; 1024],
    weights: &[i16; 1024],
    add: bool,
) {
    for i in (0..1024).step_by(16) {
        let a = _mm256_loadu_si256(acc[i..].as_ptr() as *const __m256i);
        let w = _mm256_loadu_si256(weights[i..].as_ptr() as *const __m256i);
        let result = if add {
            _mm256_add_epi16(a, w)
        } else {
            _mm256_sub_epi16(a, w)
        };
        _mm256_storeu_si256(acc[i..].as_mut_ptr() as *mut __m256i, result);
    }
}
```

### 4.4 Memory Layout

For cache efficiency:

```rust
/// Cache-aligned accumulator
#[repr(C, align(64))]
pub struct Accumulator {
    pub values: [i16; 1024],
}

/// Feature transformer with optimal memory layout
#[repr(C, align(64))]
pub struct FeatureTransformer {
    // Biases first (small, frequently accessed)
    biases: [i16; 1024],
    // Weights column-major for accumulator updates
    weights: [[i16; 1024]; 22528],
}
```

### 4.5 Lazy Evaluation

Not all positions need full quantile computation:

```rust
impl NNUEEvaluator {
    /// Quick material-only estimate using PSQT weights
    fn quick_eval(&self, pos: &Position) -> i16 {
        // Use the PSQT (piece-square table) weights from feature transformer
        // This is ~10x faster than full eval
        let mut score = 0i32;
        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                for sq in pos.pieces(color, piece) {
                    score += self.psqt_weights[feature_index];
                }
            }
        }
        (score / 8) as i16  // Divide by bucket count
    }

    /// Full evaluation only when needed
    fn evaluate_lazy(&self, pos: &Position, alpha: i16, beta: i16) -> Quantiles {
        // Quick check: if material alone is way outside window, skip full eval
        let quick = self.quick_eval(pos);
        if quick > beta + 500 || quick < alpha - 500 {
            return Quantiles::certain(quick);
        }

        // Full evaluation for close positions
        self.evaluate(pos)
    }
}
```

---

## 5. Board Representation Compatibility

### 5.1 Current Aleph vs Stockfish

| Aspect | Aleph | Stockfish | Status |
|--------|-------|-----------|--------|
| Square encoding | a1=0, rank×8+file | a1=0, rank×8+file | ✅ Compatible |
| Bitboard | `Bitboard(u64)` | `Bitboard` typedef | ✅ Compatible |
| Piece types | Pawn=0..King=5 | Similar ordering | ✅ Compatible |
| Color | White=0, Black=1 | WHITE=0, BLACK=1 | ✅ Compatible |
| Position storage | `pieces[color][piece]` | `pieces[color][piece]` | ✅ Compatible |

### 5.2 Recommended Minor Additions

To fully support NNUE feature extraction:

```rust
// In types.rs - add piece-color combo for NNUE
impl Piece {
    /// Combined piece value for NNUE (color embedded)
    /// Matches Stockfish's Piece enum: PAWN, KNIGHT, ..., KING (no color)
    pub const fn nnue_index(self) -> usize {
        self as usize
    }
}

// In position.rs - efficient piece iteration
impl Position {
    /// Iterator over all pieces with their squares (for NNUE)
    pub fn all_pieces(&self) -> impl Iterator<Item = (Square, Color, Piece)> + '_ {
        [Color::White, Color::Black].into_iter().flat_map(|color| {
            Piece::ALL.into_iter().flat_map(move |piece| {
                self.pieces(color, piece).map(move |sq| (sq, color, piece))
            })
        })
    }
}
```

---

## 6. Implementation Roadmap

### Phase 1: Basic NNUE Loading (No Training)
1. Implement `.nnue` file parser (LEB128 decoding)
2. Implement HalfKAv2 feature extraction
3. Implement forward pass (no SIMD yet)
4. Load Stockfish network, return `Quantiles::certain(score)`
5. Verify: evaluation matches Stockfish on test positions

### Phase 2: SIMD Optimization
1. Add AVX2 accumulator operations
2. Add incremental update tracking in `make_move`
3. Benchmark: target <2μs per evaluation

### Phase 3: Quantile Training
1. Build data generation pipeline (Stockfish at multiple depths)
2. Modify nnue-pytorch for quantile outputs
3. Transfer learning: freeze FT, train output layers
4. Export trained network to custom `.qnnue` format

### Phase 4: Full Integration
1. Replace output layers in Aleph's NNUE code
2. Verify quantile predictions make sense
3. Benchmark pruning effectiveness
4. Fine-tune on more data if needed

---

## 7. File Size and Memory Estimates

| Component | Size |
|-----------|------|
| Feature transformer (Stockfish) | ~45 MB |
| Quantile output layers | ~200 KB |
| Accumulator (per thread) | ~4 KB |
| **Total runtime memory** | ~46 MB + 4KB/thread |

For comparison:
- Stockfish full network: ~60 MB
- Leela Chess Zero network: 100-700 MB

---

## 8. References

- Stockfish NNUE source: `../Stockfish/src/nnue/`
- nnue-pytorch trainer: `../nnue-pytorch/`
- [NNUE Documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [Chessprogramming Wiki - NNUE](https://www.chessprogramming.org/NNUE)
