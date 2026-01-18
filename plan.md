# Aleph Chess Engine - Implementation Plan

## Overview

High-performance chess engine in Rust with **distributional evaluation**.

**Key Innovation**: Neural network outputs quantiles [q10, q25, q50, q75, q90] instead of single score. Pruning emerges from uncertainty bounds - zero hardcoded magic constants.

## Current Phase: Phase 1 - Playable UCI Engine (MVP)

### Phase 1 Tasks

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1 | Core types (Color, Piece, Square, Move) | `src/types.rs` | [x] |
| 2 | Bitboard type and operations | `src/bitboard.rs` | [x] |
| 3 | Position (state, make/unmake) | `src/position.rs` | [x] |
| 4 | FEN parsing | `src/position.rs` | [x] |
| 5 | Magic bitboards for sliders | `src/magic.rs` | [x] |
| 6 | Attack tables (knight, king, pawn) | `src/magic.rs` | [x] |
| 7 | Legal move generation | `src/movegen.rs` | [x] |
| 8 | Quantiles type, Evaluator trait | `src/eval/mod.rs` | [x] |
| 9 | Classical evaluation (material + PST) | `src/eval/classical.rs` | [x] |
| 10 | Basic alpha-beta with quiescence | `src/search.rs` | [x] |
| 11 | Minimal UCI protocol | `src/uci.rs` | [x] |

**Milestone**: Play test games vs Stockfish using cutechess-cli

### Phase 2 Tasks (after Phase 1)

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 12 | Transposition table | `src/search/tt.rs` | [ ] |
| 13 | Iterative deepening + aspiration | `src/search/mod.rs` | [ ] |
| 14 | PVS (Principal Variation Search) | `src/search/mod.rs` | [ ] |
| 15 | Move ordering (MVV-LVA, killers, history) | `src/search/ordering.rs` | [ ] |
| 16 | Time management | `src/search/time.rs` | [ ] |
| 17 | Enhanced eval (pawn structure, king safety) | `src/eval/classical.rs` | [ ] |

**Milestone**: ~1800-2000 Elo with classical eval

### Phase 3 Tasks (Distributional NNUE)

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 18 | HalfKAv2-style feature extraction | `src/eval/features.rs` | [ ] |
| 19 | Distributional network (5 quantile outputs) | `src/eval/nnue.rs` | [ ] |
| 20 | Incremental accumulator | `src/eval/nnue.rs` | [ ] |
| 21 | SIMD optimization (AVX2/NEON) | `src/eval/simd.rs` | [ ] |

**Milestone**: NN-informed pruning active, measurable Elo gain

### Phase 4 Tasks (Training)

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 22 | Data generation from self-play | `training/data_gen.py` | [ ] |
| 23 | Quantile regression training | `training/train.py` | [ ] |
| 24 | A/B testing framework | `training/eval.py` | [ ] |

---

## Architecture

```
src/
├── lib.rs            # Module exports
├── main.rs           # CLI entry point
├── types.rs          # Square, Piece, Color, Move (~280 lines)
├── bitboard.rs       # Bitboard(u64) with ops (~210 lines)
├── position.rs       # Game state, make/unmake, FEN (~380 lines)
├── magic.rs          # Magic bitboards, attack tables (~490 lines)
├── movegen.rs        # Legal move generation, perft (~400 lines)
├── search.rs         # Alpha-beta, quiescence (~300 lines)
├── uci.rs            # UCI protocol (~200 lines)
└── eval/
    ├── mod.rs        # Quantiles, Evaluator trait (~120 lines)
    └── classical.rs  # Material + PST (~200 lines)
```

---

## Key Design Decisions

### Quantile-Based Pruning

Traditional engines have 12+ pruning heuristics with hundreds of magic constants.

We use ONE mechanism:
- `q10 >= beta` → prune (pessimistic bound beats beta)
- `q90 <= alpha` → prune (optimistic bound can't reach alpha)
- `uncertainty = q90 - q10` → informs reduction depth

### Classical Eval Compatibility

Classical eval returns `Quantiles::certain(score)` where all quantiles equal.
This means `q10 == q90 == score`, so pruning only triggers on normal cutoffs.

When we swap in NNUE with real quantile spread, aggressive pruning activates automatically - no search code changes needed.

---

## Verification

```bash
# Build
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Test
cargo test

# Perft (move generation correctness)
./target/release/aleph bench perft

# Play vs Stockfish
cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph \
  -engine name=Stockfish cmd=stockfish option.Depth=6 \
  -each proto=uci tc=inf \
  -games 10
```

---

## Reference

- Detailed component designs: see appendices in `.claude/plans/` or `docs/` when created
- Stockfish reference: `../Stockfish/src/`
- Pleco reference: `../pleco/`
