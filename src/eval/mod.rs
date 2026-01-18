pub mod classical;

use crate::position::Position;
use crate::types::Piece;

/// Evaluation quantiles - the core abstraction for distributional evaluation.
///
/// In Phase 1 (classical eval), all quantiles equal the score (no uncertainty).
/// In Phase 3 (NNUE), quantiles represent the true distribution of outcomes.
///
/// The search code uses quantiles uniformly - it doesn't need to know
/// whether they come from classical or neural evaluation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Quantiles {
    pub q10: i16, // 10th percentile (pessimistic bound)
    pub q25: i16, // 25th percentile
    pub q50: i16, // Median (the "score")
    pub q75: i16, // 75th percentile
    pub q90: i16, // 90th percentile (optimistic bound)
}

impl Quantiles {
    /// Create quantiles with no uncertainty (all equal to score).
    /// Used by classical evaluation.
    #[inline]
    pub const fn certain(score: i16) -> Self {
        Self {
            q10: score,
            q25: score,
            q50: score,
            q75: score,
            q90: score,
        }
    }

    /// Create quantiles with a spread (for future NNUE use).
    #[inline]
    pub const fn new(q10: i16, q25: i16, q50: i16, q75: i16, q90: i16) -> Self {
        Self { q10, q25, q50, q75, q90 }
    }

    /// The median score (main evaluation value).
    #[inline]
    pub const fn score(self) -> i16 {
        self.q50
    }

    /// Uncertainty spread (q90 - q10).
    /// Zero for classical eval, positive for distributional eval.
    #[inline]
    pub const fn uncertainty(self) -> i16 {
        self.q90 - self.q10
    }

    /// Check if pessimistic bound beats beta (safe to prune).
    #[inline]
    pub const fn pessimistic_cutoff(self, beta: i16) -> bool {
        self.q10 >= beta
    }

    /// Check if optimistic bound can't reach alpha (safe to prune).
    #[inline]
    pub const fn optimistic_cutoff(self, alpha: i16) -> bool {
        self.q90 <= alpha
    }

    /// Negate quantiles (for negamax - swap perspective).
    #[inline]
    pub const fn negate(self) -> Self {
        Self {
            q10: -self.q90,
            q25: -self.q75,
            q50: -self.q50,
            q75: -self.q25,
            q90: -self.q10,
        }
    }
}

/// Evaluator trait - implement for different evaluation strategies.
///
/// The search uses this trait to evaluate positions without knowing
/// whether it's classical material counting or neural network inference.
pub trait Evaluator {
    /// Evaluate the position from the side-to-move's perspective.
    /// Returns quantiles representing the distribution of expected outcomes.
    fn evaluate(&self, pos: &Position) -> Quantiles;
}

/// Special score values
pub const MATE_SCORE: i16 = 30000;
pub const DRAW_SCORE: i16 = 0;

/// Convert mate score to distance-to-mate.
/// Positive = we're mating, negative = we're getting mated.
#[inline]
pub const fn mate_in(ply: i16) -> i16 {
    MATE_SCORE - ply
}

#[inline]
pub const fn mated_in(ply: i16) -> i16 {
    -MATE_SCORE + ply
}

#[inline]
pub const fn is_mate_score(score: i16) -> bool {
    score.abs() > MATE_SCORE - 1000
}

/// Material values in centipawns
#[inline]
pub const fn piece_value(piece: Piece) -> i16 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certain_quantiles() {
        let q = Quantiles::certain(100);
        assert_eq!(q.score(), 100);
        assert_eq!(q.uncertainty(), 0);
        assert_eq!(q.q10, 100);
        assert_eq!(q.q90, 100);
    }

    #[test]
    fn test_quantile_negate() {
        let q = Quantiles::new(-50, 0, 100, 150, 200);
        let neg = q.negate();
        assert_eq!(neg.q10, -200);
        assert_eq!(neg.q50, -100);
        assert_eq!(neg.q90, 50);
    }

    #[test]
    fn test_cutoffs() {
        let q = Quantiles::new(80, 90, 100, 110, 120);
        assert!(q.pessimistic_cutoff(80));
        assert!(!q.pessimistic_cutoff(81));
        assert!(q.optimistic_cutoff(120));
        assert!(!q.optimistic_cutoff(119));
    }
}
