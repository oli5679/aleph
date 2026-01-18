# Aleph Chess Engine - Feature Ideas & Research

This document captures research on features needed for competitive chess engine play, with the goal of achieving parity with advanced engines while using neural networks for both search and evaluation.

## Current State

Aleph has a functional UCI engine with:
- [x] Bitboard move generation
- [x] Alpha-beta search with quiescence
- [x] Classical evaluation (material + PST)
- [x] Basic time management
- [x] UCI protocol

Estimated ELO: ~2100-2150 (at depth 4)

---

## High Priority: Search Improvements

### 1. Transposition Table (TT)

**What it is:** Hash table storing previously evaluated positions to avoid redundant search.

**Why it matters:** Can provide 4x+ speedup by avoiding re-searching transpositions.

**Implementation:**
- Zobrist hashing (incrementally updated)
- Store: hash key, best move, score, depth, bound type (exact/lower/upper)
- Replacement strategy: depth-preferred or always-replace with aging

**Resources:**
- [Chessprogramming Wiki: Transposition Table](https://www.chessprogramming.org/Transposition_Table)

### 2. Move Ordering

**Current:** MVV-LVA for captures only.

**Missing:**
| Technique | Description | ELO Gain |
|-----------|-------------|----------|
| TT Move | Try hash move first | +50-100 |
| Killer Moves | Store 2 quiet moves per ply that caused cutoffs | +30-50 |
| History Heuristic | Track [from][to] or [piece][to] success rates | +20-40 |
| Countermove | Reply that refuted opponent's last move | +10-20 |
| SEE | Static Exchange Evaluation for capture ordering | +10-20 |

**Order:** TT move → Good captures (SEE >= 0) → Killers → History → Bad captures

**Resources:**
- [History Heuristic](https://www.chessprogramming.org/History_Heuristic)
- [Killer Moves](https://rustic-chess.org/search/ordering/killers.html)

### 3. Late Move Reductions (LMR)

**What it is:** Reduce search depth for moves that are unlikely to be good (late in move ordering).

**Implementation:**
```
reduction = log(depth) * log(move_number) / C
if move not interesting: search at depth - 1 - reduction
if reduced search beats alpha: re-search at full depth
```

**Why it matters:** Can reduce effective branching factor to < 2. Used by all top engines.

**Resources:**
- [Late Move Reductions](https://www.chessprogramming.org/Late_Move_Reductions)

### 4. Null Move Pruning (NMP)

**What it is:** If passing a turn (null move) still fails high, skip this branch.

**Implementation:**
```
if not in_check and has_pieces and not pv_node:
    make_null_move()
    score = -search(depth - R - 1, -beta, -beta + 1)  // R = 3-4
    unmake_null_move()
    if score >= beta: return beta
```

**Caveats:** Disable in zugzwang positions (pawn-only endgames).

**Resources:**
- [Null Move Pruning](https://www.chessprogramming.org/Null_Move_Pruning)

### 5. Futility Pruning

**What it is:** Skip moves that can't possibly raise alpha even with a large bonus.

**Implementation:**
```
if depth <= 3 and not in_check:
    margin = [0, 200, 350, 500][depth]
    if static_eval + margin < alpha:
        skip this move (or just search captures)
```

**Resources:**
- [Futility Pruning](https://www.chessprogramming.org/Futility_Pruning)

### 6. Principal Variation Search (PVS)

**What it is:** Search first move with full window, rest with null window.

**Implementation:**
```
for i, move in moves:
    if i == 0:
        score = -search(depth - 1, -beta, -alpha)
    else:
        score = -search(depth - 1, -alpha - 1, -alpha)  // null window
        if alpha < score < beta:
            score = -search(depth - 1, -beta, -alpha)  // re-search
```

---

## High Priority: Opening & Endgame

### 7. Opening Book (Polyglot)

**What it is:** Pre-computed database of opening moves.

**Why it matters:**
- Saves time in openings
- Avoids early blunders
- Provides variety in play

**Implementation:**
- Polyglot format is standard (.bin files)
- Binary search by Zobrist hash
- Weight-based move selection

**Resources:**
- [Polyglot Format](http://hgm.nubati.net/book_format.html)
- [Chessprogramming: Opening Book](https://www.chessprogramming.org/Opening_Book)

**Recommended books:**
- `gm2001.bin` - Strong GM games
- `performance.bin` - Optimized for engine play
- Custom book from high-quality games

### 8. Syzygy Endgame Tablebases

**What it is:** Perfect play database for positions with ≤7 pieces.

**Why it matters:**
- Perfect endgame play
- WDL (win/draw/loss) for search cutoffs
- DTZ (distance to zero) for optimal moves

**File sizes:**
- 5-piece: ~1 GB
- 6-piece: ~150 GB
- 7-piece: ~18 TB (impractical for cloud)

**Implementation for Rust:**
- [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) - Rust library

**Cloud deployment:** Include 5-piece or 6-piece WDL tables only (~1-150GB).

---

## Medium Priority: UCI Features

### 9. Pondering

**What it is:** Think during opponent's time by predicting their move.

**UCI Protocol:**
- `go ponder` - Start pondering on predicted position
- `ponderhit` - Opponent played predicted move, continue searching
- `stop` - Opponent played different move, abort

**Implementation complexity:** Requires threading/async to handle commands during search.

### 10. Multi-threading (Lazy SMP)

**What it is:** Parallel search using shared transposition table.

**How it works:**
- Multiple threads search the same tree
- Shared TT means they explore different branches
- No explicit work splitting needed

**Speedup:** ~1.5-2x with 4 threads (diminishing returns after ~30 cores)

**Implementation:**
- Shared TT with atomic operations
- Each thread has different depth offset at root
- Thread voting for best move

**Resources:**
- [Lazy SMP](https://www.chessprogramming.org/Lazy_SMP)

---

## Neural Network Search (Your Innovation Goal)

### Current Approach: NNUE (CPU-based)

**Architecture:**
- Input: HalfKP/HalfKAv2 features (king position + piece positions)
- Layers: 768→256→32→32→1 (or similar)
- Output: Single score (or quantiles in our case)

**Key feature:** Incremental update - only changed features recomputed.

**Performance:** 50M+ nodes/second on CPU (100x faster than deep NN).

**Training:** PyTorch or Rust-based Bullet trainer.

### Alternative: MCTS + Policy Network (GPU-based)

**Architecture (AlphaZero-style):**
- Input: Board representation (8x8x17 or similar)
- Network: ResNet with ~20-40 blocks
- Output: Policy (move probabilities) + Value (position score)

**Search:** Monte Carlo Tree Search guided by policy network.

**Performance:** ~100K nodes/second (but smarter node selection).

**Tradeoffs:**
| Aspect | NNUE + Alpha-Beta | MCTS + Policy |
|--------|-------------------|---------------|
| Speed | 50M nps | 100K nps |
| Hardware | CPU | GPU required |
| Training data | Stockfish positions | Self-play |
| Strength | ~3500 ELO | ~3500 ELO |
| Implementation | Medium | Complex |

### Hybrid Approach: Policy-Guided Alpha-Beta

**Idea:** Use policy network to order moves, but keep alpha-beta search.

**Implementation:**
```
policy = network.policy(position)  // Get move probabilities
moves.sort_by(|m| policy[m])       // Order by policy score
// Then normal alpha-beta with LMR/NMP etc.
```

**Benefits:**
- Better move ordering than heuristics
- Keeps fast alpha-beta search
- Can run policy network on GPU, search on CPU

---

## GPU vs CPU Considerations

### NNUE: CPU is Optimal

NNUE was specifically designed for CPU inference:
- Incremental updates (only changed features)
- SIMD vectorization (AVX2/AVX512/NEON)
- Sub-microsecond inference time

**GPU would be slower** due to:
- Kernel launch overhead
- Memory transfer latency
- Network too small to benefit from parallelism

### Deep Networks (MCTS): GPU Required

For AlphaZero-style networks (ResNet with 20+ blocks):
- GPU is 10-100x faster than CPU
- Batch inference amortizes overhead
- Modern GPUs: RTX 4090 can do ~40K inferences/second

### Rust vs Python for Inference

| Aspect | Rust | Python (PyTorch) |
|--------|------|------------------|
| NNUE inference | ✅ Best (native SIMD) | ❌ Overhead too high |
| Deep NN (CPU) | ✅ Good (burn/candle) | ⚠️ Acceptable |
| Deep NN (GPU) | ⚠️ Emerging (burn/candle) | ✅ Best (PyTorch/CUDA) |
| Training | ⚠️ Bullet (limited) | ✅ Best (PyTorch) |

**Recommendation:**
- **NNUE:** Keep everything in Rust
- **Deep NN with GPU:** Consider Python/PyTorch for inference, or use Rust bindings to ONNX Runtime

---

## Implementation Priority

### Phase 2: Core Search (No NN changes)
1. Transposition Table
2. Killer moves + History heuristic
3. LMR + NMP + Futility
4. PVS
5. Time management improvements

### Phase 3: Books & Tables
1. Polyglot opening book support
2. Syzygy 5/6-piece WDL probing

### Phase 4: Neural Network
1. NNUE with quantile outputs (CPU)
2. Optional: Policy network for move ordering (GPU)

### Phase 5: Parallelization
1. Lazy SMP (4-8 threads)
2. Pondering

---

## Resources

### Chessprogramming Wiki
- [Main Page](https://www.chessprogramming.org)
- [Search](https://www.chessprogramming.org/Search)
- [Evaluation](https://www.chessprogramming.org/Evaluation)

### Reference Engines (Open Source)
- [Stockfish](https://github.com/official-stockfish/Stockfish) - NNUE, Lazy SMP
- [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) - MCTS, GPU
- [Ethereal](https://github.com/AndyGrant/Ethereal) - Clean NNUE implementation
- [Berserk](https://github.com/jhonnold/berserk) - Strong Rust-friendly ideas

### Rust Libraries
- [shakmaty](https://github.com/niklasf/shakmaty) - Chess library
- [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) - Tablebase probing
- [burn](https://github.com/tracel-ai/burn) - Deep learning in Rust
- [candle](https://github.com/huggingface/candle) - ML framework from HuggingFace

### Training
- [Bullet](https://github.com/jw1912/bullet) - Rust NNUE trainer
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) - Stockfish trainer
