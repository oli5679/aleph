use crate::eval::{Evaluator, Quantiles};
use crate::position::Position;
use crate::types::{Color, Piece};
use crate::values::piece_value;

/// Classical evaluation using material counting and piece-square tables.
/// Returns Quantiles::certain(score) since there's no uncertainty.
pub struct ClassicalEval;

impl ClassicalEval {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ClassicalEval {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for ClassicalEval {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        let score = evaluate_classical(pos);
        Quantiles::certain(score)
    }
}

/// Evaluate position from side-to-move perspective.
fn evaluate_classical(pos: &Position) -> i16 {
    let white_score = evaluate_side(pos, Color::White);
    let black_score = evaluate_side(pos, Color::Black);
    let score = white_score - black_score;

    if pos.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

/// Pieces that count for material (excludes King).
const MATERIAL_PIECES: [Piece; 5] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
];

/// Evaluate one side's material and positional factors.
fn evaluate_side(pos: &Position, color: Color) -> i16 {
    let mut score = 0i16;

    // Material (King has no material value)
    for &piece in &MATERIAL_PIECES {
        let count = pos.pieces(color, piece).count() as i16;
        score += count * piece_value(piece);
    }

    // Piece-square tables
    score += psqt_score(pos, color);

    score
}

/// Piece-square table bonus for a side.
fn psqt_score(pos: &Position, color: Color) -> i16 {
    let mut score = 0i16;

    for sq in pos.pieces(color, Piece::Pawn) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += PAWN_PST[idx];
    }

    for sq in pos.pieces(color, Piece::Knight) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += KNIGHT_PST[idx];
    }

    for sq in pos.pieces(color, Piece::Bishop) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += BISHOP_PST[idx];
    }

    for sq in pos.pieces(color, Piece::Rook) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += ROOK_PST[idx];
    }

    for sq in pos.pieces(color, Piece::Queen) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += QUEEN_PST[idx];
    }

    for sq in pos.pieces(color, Piece::King) {
        let idx = if color == Color::White {
            sq.index()
        } else {
            sq.flip_rank().index()
        };
        score += KING_PST[idx];
    }

    score
}

// Piece-square tables (from White's perspective, a1=0, h8=63)
// Values are bonuses/penalties in centipawns

#[rustfmt::skip]
const PAWN_PST: [i16; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,  // rank 1 (pawns can't be here)
     5, 10, 10,-20,-20, 10, 10,  5,  // rank 2
     5, -5,-10,  0,  0,-10, -5,  5,  // rank 3
     0,  0,  0, 20, 20,  0,  0,  0,  // rank 4
     5,  5, 10, 25, 25, 10,  5,  5,  // rank 5
    10, 10, 20, 30, 30, 20, 10, 10,  // rank 6
    50, 50, 50, 50, 50, 50, 50, 50,  // rank 7
     0,  0,  0,  0,  0,  0,  0,  0,  // rank 8 (pawns promote)
];

#[rustfmt::skip]
const KNIGHT_PST: [i16; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [i16; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_PST: [i16; 64] = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_PST: [i16; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_PST: [i16; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20,  // Prefer castled positions
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_startpos_eval() {
        let pos = Position::startpos();
        let eval = ClassicalEval::new();
        let q = eval.evaluate(&pos);
        // Starting position should be roughly equal
        assert!(q.score().abs() < 50);
    }

    #[test]
    fn test_material_advantage() {
        // White up a queen
        let pos = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let eval = ClassicalEval::new();
        let q = eval.evaluate(&pos);
        // White should be winning by roughly a queen's value
        assert!(q.score() > 800);
    }

    #[test]
    fn test_symmetry() {
        // Symmetric position should evaluate to ~0
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let eval = ClassicalEval::new();
        let q = eval.evaluate(&pos);
        assert!(q.score().abs() < 10);
    }
}
