# Aleph Chess Engine - Technical Specification

A high-performance chess engine in Rust with distributional evaluation.

---

## Design Principles

### 1. Performance First

- **Zero-cost abstractions**: Traits resolve at compile time
- **No allocations in hot paths**: Stack-allocated move lists, pre-sized buffers
- **Cache-friendly layouts**: Bitboards fit in registers, structs aligned
- **SIMD from day one**: Design data layouts for vectorization
- **Inline aggressively**: `#[inline]` on all hot functions

### 2. Clean, Not Over-Engineered

- **Minimal abstraction layers**: Position → Search → Eval, that's it
- **No premature generics**: Concrete types until proven otherwise
- **Small files**: Each module < 500 lines, single responsibility
- **Obvious code**: Prefer clarity over cleverness
- **No dead code**: If it's not used, delete it

### 3. Original Design

- **Not a port**: Fresh architecture, learn from but don't copy Stockfish/pleco
- **Different trade-offs**: Optimize for our use case (distributional eval)
- **Rust-idiomatic**: Use ownership, enums, pattern matching naturally

### 4. Seamless Transition Path

The architecture supports progressive enhancement without rewrites:

```
Phase 1: Position + Search + ClassicalEval
Phase 2: Position + Search + DistributionalNNUE (drop-in replacement)
```

Key design: `Evaluator` trait with single method that returns quantiles.
Classical eval returns `[score, score, score, score, score]` (no uncertainty).
NNUE returns true distribution. Search code doesn't change.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│                    UCI                        │
│              (stdin/stdout)                   │
└──────────────────────────────────────────────┘
                     │
┌──────────────────────────────────────────────┐
│                  Engine                       │
│         (orchestration, time mgmt)            │
└──────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐        ┌───────────────────┐
│    Search     │◄──────►│    Evaluator      │
│  (alpha-beta) │        │  (trait object)   │
└───────────────┘        └───────────────────┘
        │                         │
        ▼                         │
┌───────────────┐                 │
│   Position    │◄────────────────┘
│  (bitboards)  │
└───────────────┘
```

---

## Module Structure

```
src/
├── main.rs           # UCI loop only (~100 lines)
├── lib.rs            # Public API
│
├── types.rs          # Color, Piece, Square, Move (~150 lines)
├── bitboard.rs       # Bitboard operations (~200 lines)
├── position.rs       # Board state, make/unmake (~400 lines)
├── movegen.rs        # Move generation (~300 lines)
├── magic.rs          # Magic bitboards (~200 lines)
│
├── search.rs         # Alpha-beta search (~400 lines)
├── tt.rs             # Transposition table (~150 lines)
├── ordering.rs       # Move ordering (~150 lines)
│
├── eval/
│   ├── mod.rs        # Evaluator trait (~50 lines)
│   ├── classical.rs  # Material + PSQT (~200 lines)
│   └── nnue.rs       # Distributional NNUE (~300 lines, Phase 3)
│
└── uci.rs            # UCI protocol (~200 lines)
```

**Total Phase 1**: ~1800 lines
**Total Phase 3**: ~2400 lines

---

## Core Types

```rust
// types.rs

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Color { White = 0, Black = 1 }

impl Color {
    #[inline] pub fn flip(self) -> Self {
        unsafe { std::mem::transmute(self as u8 ^ 1) }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Piece { Pawn = 0, Knight = 1, Bishop = 2, Rook = 3, Queen = 4, King = 5 }

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct Square(pub u8);  // 0-63, a1=0, h8=63

impl Square {
    #[inline] pub fn new(file: u8, rank: u8) -> Self { Self(rank * 8 + file) }
    #[inline] pub fn file(self) -> u8 { self.0 & 7 }
    #[inline] pub fn rank(self) -> u8 { self.0 >> 3 }
    #[inline] pub fn flip(self) -> Self { Self(self.0 ^ 56) }  // Vertical flip
}

/// 16-bit move encoding
/// Bits 0-5:   from square
/// Bits 6-11:  to square
/// Bits 12-13: promotion (N=0, B=1, R=2, Q=3)
/// Bits 14-15: flags (normal=0, promo=1, ep=2, castle=3)
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Self = Self(0);

    #[inline] pub fn new(from: Square, to: Square) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6))
    }
    #[inline] pub fn from(self) -> Square { Square((self.0 & 63) as u8) }
    #[inline] pub fn to(self) -> Square { Square(((self.0 >> 6) & 63) as u8) }
    #[inline] pub fn is_null(self) -> bool { self.0 == 0 }
}

/// Castling rights as 4-bit flags
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const WK: u8 = 1;
    pub const WQ: u8 = 2;
    pub const BK: u8 = 4;
    pub const BQ: u8 = 8;
}
```

---

## Bitboard

```rust
// bitboard.rs

#[derive(Copy, Clone, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(!0);

    // Files and ranks
    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);
    pub const RANK_1: Self = Self(0x0000_0000_0000_00FF);
    pub const RANK_8: Self = Self(0xFF00_0000_0000_0000);

    #[inline] pub fn from_sq(sq: Square) -> Self { Self(1 << sq.0) }
    #[inline] pub fn contains(self, sq: Square) -> bool { self.0 & (1 << sq.0) != 0 }
    #[inline] pub fn is_empty(self) -> bool { self.0 == 0 }
    #[inline] pub fn count(self) -> u32 { self.0.count_ones() }

    #[inline] pub fn lsb(self) -> Square {
        debug_assert!(!self.is_empty());
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline] pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }

    // Shifts (with wrap prevention)
    #[inline] pub fn north(self) -> Self { Self(self.0 << 8) }
    #[inline] pub fn south(self) -> Self { Self(self.0 >> 8) }
    #[inline] pub fn east(self) -> Self { Self((self.0 << 1) & !Self::FILE_A.0) }
    #[inline] pub fn west(self) -> Self { Self((self.0 >> 1) & !Self::FILE_H.0) }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;
    #[inline] fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Self;
    #[inline] fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;
    #[inline] fn bitxor(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }
}

impl std::ops::Not for Bitboard {
    type Output = Self;
    #[inline] fn not(self) -> Self { Self(!self.0) }
}

// Iterator over set bits
impl Iterator for Bitboard {
    type Item = Square;
    #[inline] fn next(&mut self) -> Option<Square> {
        if self.is_empty() { None } else { Some(self.pop_lsb()) }
    }
}
```

---

## Position

```rust
// position.rs

pub struct Position {
    // Piece bitboards: pieces[color][piece_type]
    pieces: [[Bitboard; 6]; 2],

    // Occupancy (derived, kept in sync)
    occupied: [Bitboard; 2],  // Per color
    all: Bitboard,            // All pieces

    // State
    side: Color,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove: u8,
    fullmove: u16,

    // Zobrist hash (incrementally updated)
    hash: u64,

    // Undo stack
    history: Vec<Undo>,
}

struct Undo {
    mv: Move,
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove: u8,
    hash: u64,
}

impl Position {
    pub fn startpos() -> Self { Self::from_fen(STARTPOS).unwrap() }

    pub fn from_fen(fen: &str) -> Result<Self, &'static str> { /* ... */ }
    pub fn to_fen(&self) -> String { /* ... */ }

    #[inline] pub fn side_to_move(&self) -> Color { self.side }
    #[inline] pub fn hash(&self) -> u64 { self.hash }

    #[inline] pub fn pieces(&self, color: Color, piece: Piece) -> Bitboard {
        self.pieces[color as usize][piece as usize]
    }

    #[inline] pub fn occupied_by(&self, color: Color) -> Bitboard {
        self.occupied[color as usize]
    }

    #[inline] pub fn piece_at(&self, sq: Square) -> Option<(Color, Piece)> {
        let bb = Bitboard::from_sq(sq);
        for color in [Color::White, Color::Black] {
            if !(self.occupied[color as usize] & bb).is_empty() {
                for piece in 0..6 {
                    if !(self.pieces[color as usize][piece] & bb).is_empty() {
                        return Some((color, unsafe { std::mem::transmute(piece as u8) }));
                    }
                }
            }
        }
        None
    }

    pub fn make_move(&mut self, mv: Move) {
        // Save undo info
        self.history.push(Undo {
            mv,
            captured: self.piece_at(mv.to()).map(|(_, p)| p),
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            hash: self.hash,
        });

        // Update bitboards, hash, state...
        // (Implementation details omitted for brevity)
    }

    pub fn unmake_move(&mut self) {
        let undo = self.history.pop().expect("no move to unmake");
        // Restore state from undo...
    }

    pub fn in_check(&self) -> bool {
        let king_sq = self.pieces(self.side, Piece::King).lsb();
        self.is_attacked(king_sq, self.side.flip())
    }

    fn is_attacked(&self, sq: Square, by: Color) -> bool {
        // Check each piece type for attacks to sq
        // Use magic bitboards for sliding pieces
        false // TODO
    }
}

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
```

---

## Move Generation

```rust
// movegen.rs

/// Stack-allocated move list (no heap allocation)
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    #[inline] pub fn new() -> Self {
        Self { moves: [Move::NULL; 256], len: 0 }
    }

    #[inline] pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline] pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().copied()
    }
}

impl Position {
    /// Generate all legal moves
    pub fn generate_moves(&self, moves: &mut MoveList) {
        let us = self.side;
        let them = us.flip();
        let our_pieces = self.occupied_by(us);
        let their_pieces = self.occupied_by(them);
        let empty = !self.all;

        // Pawns
        self.gen_pawn_moves(moves, us, empty, their_pieces);

        // Knights
        for from in self.pieces(us, Piece::Knight) {
            let attacks = KNIGHT_ATTACKS[from.0 as usize] & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }

        // Bishops
        for from in self.pieces(us, Piece::Bishop) {
            let attacks = bishop_attacks(from, self.all) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }

        // Rooks
        for from in self.pieces(us, Piece::Rook) {
            let attacks = rook_attacks(from, self.all) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }

        // Queens
        for from in self.pieces(us, Piece::Queen) {
            let attacks = queen_attacks(from, self.all) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }

        // King
        let king_sq = self.pieces(us, Piece::King).lsb();
        let attacks = KING_ATTACKS[king_sq.0 as usize] & !our_pieces;
        for to in attacks {
            moves.push(Move::new(king_sq, to));
        }

        // Castling
        self.gen_castling_moves(moves, us);

        // Filter illegal moves (leaves king in check)
        self.filter_legal(moves);
    }

    fn filter_legal(&self, moves: &mut MoveList) {
        let mut i = 0;
        while i < moves.len {
            let mv = moves.moves[i];
            let mut copy = self.clone();
            copy.make_move(mv);
            if copy.in_check() {
                // Remove illegal move by swapping with last
                moves.len -= 1;
                moves.moves[i] = moves.moves[moves.len];
            } else {
                i += 1;
            }
        }
    }
}

// Pre-computed attack tables
static KNIGHT_ATTACKS: [Bitboard; 64] = /* ... */;
static KING_ATTACKS: [Bitboard; 64] = /* ... */;
```

---

## Magic Bitboards

```rust
// magic.rs

struct Magic {
    mask: Bitboard,
    magic: u64,
    shift: u8,
    offset: u32,
}

static ROOK_MAGICS: [Magic; 64] = /* ... */;
static BISHOP_MAGICS: [Magic; 64] = /* ... */;
static ATTACK_TABLE: [Bitboard; 88772] = /* ... */;  // ~700KB

#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let m = &ROOK_MAGICS[sq.0 as usize];
    let idx = ((occupied & m.mask).0.wrapping_mul(m.magic) >> m.shift) as usize;
    ATTACK_TABLE[m.offset as usize + idx]
}

#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let m = &BISHOP_MAGICS[sq.0 as usize];
    let idx = ((occupied & m.mask).0.wrapping_mul(m.magic) >> m.shift) as usize;
    ATTACK_TABLE[m.offset as usize + idx]
}

#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}
```

---

## Evaluator Trait (The Transition Point)

```rust
// eval/mod.rs

/// Evaluation quantiles - the core abstraction
///
/// Phase 1 (Classical): Returns [score, score, score, score, score]
/// Phase 3 (NNUE): Returns true distribution [q10, q25, q50, q75, q90]
///
/// Search code never changes - it just uses the quantiles.
#[derive(Copy, Clone)]
pub struct Quantiles {
    pub q10: i16,  // 10th percentile (pessimistic)
    pub q25: i16,  // 25th percentile
    pub q50: i16,  // Median (the "score")
    pub q75: i16,  // 75th percentile
    pub q90: i16,  // 90th percentile (optimistic)
}

impl Quantiles {
    /// Classical eval: no uncertainty
    #[inline]
    pub fn certain(score: i16) -> Self {
        Self { q10: score, q25: score, q50: score, q75: score, q90: score }
    }

    /// Uncertainty spread
    #[inline]
    pub fn uncertainty(&self) -> i16 {
        self.q90 - self.q10
    }

    /// Should we prune this node?
    #[inline]
    pub fn should_prune(&self, alpha: i16, beta: i16) -> bool {
        self.q10 >= beta || self.q90 <= alpha
    }

    /// Depth reduction based on uncertainty
    #[inline]
    pub fn reduction(&self, depth: i32, move_idx: usize) -> i32 {
        let u = self.uncertainty();
        let base = if u < 50 { 2 } else if u < 150 { 1 } else { 0 };
        let late_move = (move_idx / 6).min(3) as i32;
        (base + late_move).min(depth - 1).max(0)
    }
}

/// Evaluator trait - implement for classical or NNUE
pub trait Evaluator {
    fn evaluate(&self, pos: &Position) -> Quantiles;
}
```

---

## Classical Evaluation (Phase 1)

```rust
// eval/classical.rs

pub struct ClassicalEval;

impl Evaluator for ClassicalEval {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        let score = self.score(pos);
        Quantiles::certain(score)
    }
}

impl ClassicalEval {
    fn score(&self, pos: &Position) -> i16 {
        let mut score = 0i16;

        // Material
        score += self.material(pos, Color::White);
        score -= self.material(pos, Color::Black);

        // Piece-square tables
        score += self.psqt(pos, Color::White);
        score -= self.psqt(pos, Color::Black);

        // Return from side-to-move perspective
        if pos.side_to_move() == Color::Black { -score } else { score }
    }

    fn material(&self, pos: &Position, color: Color) -> i16 {
        let p = pos.pieces(color, Piece::Pawn).count() as i16;
        let n = pos.pieces(color, Piece::Knight).count() as i16;
        let b = pos.pieces(color, Piece::Bishop).count() as i16;
        let r = pos.pieces(color, Piece::Rook).count() as i16;
        let q = pos.pieces(color, Piece::Queen).count() as i16;

        p * 100 + n * 320 + b * 330 + r * 500 + q * 900
    }

    fn psqt(&self, pos: &Position, color: Color) -> i16 {
        let mut score = 0i16;
        for piece in 0..6 {
            let pt = unsafe { std::mem::transmute::<u8, Piece>(piece) };
            for sq in pos.pieces(color, pt) {
                let idx = if color == Color::White { sq.0 } else { sq.flip().0 };
                score += PSQT[piece as usize][idx as usize];
            }
        }
        score
    }
}

// Piece-square tables (simplified)
static PSQT: [[i16; 64]; 6] = [
    // Pawn
    [0; 64],  // TODO: Fill with actual values
    // Knight - prefer center
    [0; 64],
    // Bishop
    [0; 64],
    // Rook
    [0; 64],
    // Queen
    [0; 64],
    // King - prefer corners in middlegame
    [0; 64],
];
```

---

## Distributional NNUE (Phase 3)

```rust
// eval/nnue.rs

/// NNUE that outputs quantiles instead of single score
pub struct DistributionalNNUE {
    weights: NNUEWeights,
}

impl Evaluator for DistributionalNNUE {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        let features = self.extract_features(pos);
        let raw = self.forward(&features);

        Quantiles {
            q10: raw[0],
            q25: raw[1],
            q50: raw[2],
            q75: raw[3],
            q90: raw[4],
        }
    }
}

impl DistributionalNNUE {
    fn extract_features(&self, pos: &Position) -> [i16; 512] {
        // HalfKP feature extraction
        // For each perspective: king_square * 64 + piece_square * 10 + piece_type
        todo!()
    }

    fn forward(&self, features: &[i16; 512]) -> [i16; 5] {
        // Simple feedforward:
        // 512 -> 64 (ClippedReLU) -> 32 (ClippedReLU) -> 5 (quantiles)
        todo!()
    }
}
```

---

## Search

```rust
// search.rs

pub struct Searcher<E: Evaluator> {
    pos: Position,
    eval: E,
    tt: TranspositionTable,
    history: [[i32; 64]; 64],  // from-to history
    nodes: u64,
    stop: bool,
}

impl<E: Evaluator> Searcher<E> {
    pub fn new(pos: Position, eval: E, tt_mb: usize) -> Self {
        Self {
            pos,
            eval,
            tt: TranspositionTable::new(tt_mb),
            history: [[0; 64]; 64],
            nodes: 0,
            stop: false,
        }
    }

    /// Iterative deepening search
    pub fn search(&mut self, max_depth: i32) -> (Move, i16) {
        let mut best_move = Move::NULL;
        let mut best_score = 0;

        for depth in 1..=max_depth {
            if self.stop { break; }

            let score = self.negamax(depth, 0, -30000, 30000, true);

            if !self.stop {
                best_score = score;
                if let Some(entry) = self.tt.probe(self.pos.hash()) {
                    best_move = entry.mv;
                }
            }
        }

        (best_move, best_score)
    }

    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i16, beta: i16, pv: bool) -> i16 {
        if depth <= 0 {
            return self.quiesce(alpha, beta);
        }

        self.nodes += 1;

        // TT probe
        if let Some(entry) = self.tt.probe(self.pos.hash()) {
            if !pv && entry.depth >= depth as i8 {
                match entry.bound {
                    Bound::Exact => return entry.score,
                    Bound::Lower if entry.score >= beta => return entry.score,
                    Bound::Upper if entry.score <= alpha => return entry.score,
                    _ => {}
                }
            }
        }

        // Evaluate for pruning decisions
        let quantiles = self.eval.evaluate(&self.pos);

        // Uncertainty-based pruning (replaces NMP, futility, etc.)
        if !pv && !self.pos.in_check() && depth >= 2 {
            if quantiles.should_prune(alpha, beta) {
                // Verification search
                let v = -self.negamax(depth / 2, ply + 1, -beta, -beta + 1, false);
                if v >= beta {
                    return v;
                }
            }
        }

        // Generate and search moves
        let mut moves = MoveList::new();
        self.pos.generate_moves(&mut moves);

        if moves.is_empty() {
            return if self.pos.in_check() { -30000 + ply as i16 } else { 0 };
        }

        // Move ordering
        self.order_moves(&mut moves);

        let mut best_score = -30000i16;
        let mut best_move = Move::NULL;

        for (i, mv) in moves.iter().enumerate() {
            self.pos.make_move(mv);

            // Reduction based on uncertainty
            let r = if i > 3 && depth >= 3 && !pv {
                quantiles.reduction(depth, i)
            } else {
                0
            };

            // PVS
            let score = if i == 0 {
                -self.negamax(depth - 1, ply + 1, -beta, -alpha, pv)
            } else {
                let mut s = -self.negamax(depth - 1 - r, ply + 1, -alpha - 1, -alpha, false);
                if s > alpha && (r > 0 || pv) {
                    s = -self.negamax(depth - 1, ply + 1, -beta, -alpha, pv);
                }
                s
            };

            self.pos.unmake_move();

            if score > best_score {
                best_score = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    if score >= beta {
                        self.history[mv.from().0 as usize][mv.to().0 as usize] += depth as i32 * depth as i32;
                        break;
                    }
                }
            }
        }

        // TT store
        let bound = if best_score >= beta { Bound::Lower }
                   else if best_score > alpha { Bound::Exact }
                   else { Bound::Upper };
        self.tt.store(self.pos.hash(), best_move, best_score, depth as i8, bound);

        best_score
    }

    fn quiesce(&mut self, mut alpha: i16, beta: i16) -> i16 {
        let stand_pat = self.eval.evaluate(&self.pos).q50;

        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }

        // Generate captures only
        let mut captures = MoveList::new();
        self.pos.generate_captures(&mut captures);

        for mv in captures.iter() {
            self.pos.make_move(mv);
            let score = -self.quiesce(-beta, -alpha);
            self.pos.unmake_move();

            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }

        alpha
    }

    fn order_moves(&self, moves: &mut MoveList) {
        // Simple ordering: captures first (MVV-LVA), then by history
        // TODO: TT move first, killers, etc.
    }
}
```

---

## Transposition Table

```rust
// tt.rs

#[derive(Copy, Clone)]
pub enum Bound { Exact, Lower, Upper }

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u64,
    pub mv: Move,
    pub score: i16,
    pub depth: i8,
    pub bound: Bound,
}

pub struct TranspositionTable {
    entries: Box<[Option<TTEntry>]>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let size = (mb * 1024 * 1024 / std::mem::size_of::<Option<TTEntry>>()).next_power_of_two();
        Self {
            entries: vec![None; size].into_boxed_slice(),
            mask: size - 1,
        }
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = (key as usize) & self.mask;
        self.entries[idx].filter(|e| e.key == key)
    }

    #[inline]
    pub fn store(&mut self, key: u64, mv: Move, score: i16, depth: i8, bound: Bound) {
        let idx = (key as usize) & self.mask;
        self.entries[idx] = Some(TTEntry { key, mv, score, depth, bound });
    }
}
```

---

## UCI Protocol

```rust
// uci.rs

pub fn uci_loop() {
    let mut pos = Position::startpos();
    let mut tt = TranspositionTable::new(64);

    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        match tokens[0] {
            "uci" => {
                println!("id name Aleph");
                println!("id author [Author]");
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => tt = TranspositionTable::new(64),
            "position" => pos = parse_position(&tokens[1..]),
            "go" => {
                let depth = parse_go_depth(&tokens[1..]);
                let eval = ClassicalEval;
                let mut searcher = Searcher::new(pos.clone(), eval, 64);
                let (mv, _) = searcher.search(depth);
                println!("bestmove {}", format_move(mv));
            }
            "quit" => break,
            _ => {}
        }
    }
}

fn parse_position(tokens: &[&str]) -> Position { /* ... */ }
fn parse_go_depth(tokens: &[&str]) -> i32 { /* ... */ }
fn format_move(mv: Move) -> String { /* ... */ }
```

---

## Training Pipeline (Phase 4)

### Data Collection

```rust
// In engine, add data collection mode

struct Sample {
    fen: String,
    deep_score: i16,
}

impl Searcher<ClassicalEval> {
    fn collect_samples(&mut self, num_games: usize) -> Vec<Sample> {
        let mut samples = Vec::new();

        for _ in 0..num_games {
            self.pos = Position::startpos();

            while !self.is_game_over() {
                let (mv, score) = self.search(12);  // Deep search
                samples.push(Sample {
                    fen: self.pos.to_fen(),
                    deep_score: score,
                });
                self.pos.make_move(mv);
            }
        }

        samples
    }
}
```

### Python Training

```python
# training/train.py

import torch
import torch.nn as nn

class DistributionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(768 * 2, 256)  # Feature transformer
        self.l1 = nn.Linear(512, 64)
        self.l2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 5)  # 5 quantiles

    def forward(self, white_features, black_features):
        w = torch.clamp(self.ft(white_features), 0, 127)
        b = torch.clamp(self.ft(black_features), 0, 127)
        x = torch.cat([w, b], dim=1)
        x = torch.clamp(self.l1(x), 0, 127)
        x = torch.clamp(self.l2(x), 0, 127)
        return self.out(x)

def quantile_loss(pred, target, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    loss = 0
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        loss += torch.mean(torch.max(q * err, (q - 1) * err))
    return loss / len(quantiles)
```

---

## Implementation Phases

### Phase 1: Playable Engine (MVP)

- [x] Types, Bitboard, Position
- [ ] Move generation (legal)
- [ ] Magic bitboards
- [ ] Basic search (alpha-beta, no pruning)
- [ ] Classical evaluation (material + PSQT)
- [ ] UCI (minimal)

**Test**: Play 10 games vs Stockfish depth 4.

### Phase 2: Search Enhancement

- [ ] Transposition table
- [ ] Iterative deepening
- [ ] PVS
- [ ] Quiescence search
- [ ] Move ordering (history)
- [ ] Time management

**Test**: Reach ~2000 Elo.

### Phase 3: Distributional NNUE

- [ ] Feature extraction
- [ ] Network inference
- [ ] Quantile-based pruning
- [ ] SIMD optimization

**Test**: Verify uncertainty correlates with tactical complexity.

### Phase 4: Training

- [ ] Data collection
- [ ] Quantile regression
- [ ] Export weights
- [ ] Iterate

**Test**: A/B test vs classical pruning.

---

## Testing

```bash
# Build
cargo build --release

# Perft (move generation correctness)
./target/release/aleph perft 5

# Benchmark
./target/release/aleph bench

# Play vs Stockfish
cutechess-cli -engine cmd=./target/release/aleph -engine cmd=stockfish \
  -each proto=uci tc=10+0.1 -games 100
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Perft(6) speed | < 2s |
| Nodes/second (search) | > 2M |
| Phase 1 strength | ~1500 Elo |
| Phase 2 strength | ~2000 Elo |
| Phase 3 strength | ~2500 Elo |
