//! NNUE accumulator management.
//!
//! The accumulator stores the result of applying the feature transformer
//! to the active features. It can be updated incrementally when pieces
//! move, which is much faster than recomputing from scratch.

use crate::position::Position;
use crate::types::{Color, Piece, Square};

use super::features::{HalfKAv2, HALF_DIMENSIONS};
use super::network::L1_SIZE;

/// Accumulator holding transformed features for both perspectives.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; L1_SIZE],
    pub black: [i16; L1_SIZE],
    /// Whether the accumulator is computed (vs needing refresh).
    pub computed: bool,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            white: [0; L1_SIZE],
            black: [0; L1_SIZE],
            computed: false,
        }
    }
}

impl Accumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute accumulator from scratch for a position.
    /// This is used when:
    /// 1. Starting a new search
    /// 2. King moves to a different bucket (requires full refresh)
    pub fn from_position(pos: &Position) -> Self {
        // For now, just return a simple accumulator
        // In a real implementation, this would apply the feature transformer
        let mut acc = Self::new();

        let white_king = pos.king_sq(Color::White);
        let black_king = pos.king_sq(Color::Black);

        // Collect all pieces
        let pieces: Vec<_> = pos.all_pieces().collect();

        // Compute white perspective
        for (sq, color, piece) in &pieces {
            let idx = HalfKAv2::make_index(Color::White, *sq, *piece, *color, white_king);
            if idx != usize::MAX && idx < HALF_DIMENSIONS {
                // In a real implementation, this would add the feature weights
                // For now, just mark that this feature is active
                let hash_idx = idx % L1_SIZE;
                acc.white[hash_idx] = acc.white[hash_idx].saturating_add(1);
            }
        }

        // Compute black perspective
        for (sq, color, piece) in &pieces {
            let idx = HalfKAv2::make_index(Color::Black, *sq, *piece, *color, black_king);
            if idx != usize::MAX && idx < HALF_DIMENSIONS {
                let hash_idx = idx % L1_SIZE;
                acc.black[hash_idx] = acc.black[hash_idx].saturating_add(1);
            }
        }

        acc.computed = true;
        acc
    }

    /// Refresh the accumulator for a specific perspective.
    /// Called when the king moves to a different bucket.
    pub fn refresh_perspective(
        &mut self,
        perspective: Color,
        king_sq: Square,
        pieces: impl Iterator<Item = (Square, Color, Piece)>,
    ) {
        let acc = if perspective == Color::White {
            &mut self.white
        } else {
            &mut self.black
        };

        // Zero out
        for v in acc.iter_mut() {
            *v = 0;
        }

        // Add all features
        for (sq, color, piece) in pieces {
            let idx = HalfKAv2::make_index(perspective, sq, piece, color, king_sq);
            if idx != usize::MAX && idx < HALF_DIMENSIONS {
                let hash_idx = idx % L1_SIZE;
                acc[hash_idx] = acc[hash_idx].saturating_add(1);
            }
        }
    }

    /// Add a feature (piece placed on board).
    #[inline]
    pub fn add_feature(&mut self, perspective: Color, feature_idx: usize) {
        if feature_idx >= HALF_DIMENSIONS {
            return;
        }
        let acc = if perspective == Color::White {
            &mut self.white
        } else {
            &mut self.black
        };
        let hash_idx = feature_idx % L1_SIZE;
        acc[hash_idx] = acc[hash_idx].saturating_add(1);
    }

    /// Remove a feature (piece removed from board).
    #[inline]
    pub fn sub_feature(&mut self, perspective: Color, feature_idx: usize) {
        if feature_idx >= HALF_DIMENSIONS {
            return;
        }
        let acc = if perspective == Color::White {
            &mut self.white
        } else {
            &mut self.black
        };
        let hash_idx = feature_idx % L1_SIZE;
        acc[hash_idx] = acc[hash_idx].saturating_sub(1);
    }

    /// Check if king bucket changed (requires full refresh).
    #[inline]
    pub fn king_bucket_changed(old_sq: Square, new_sq: Square) -> bool {
        HalfKAv2::king_bucket(old_sq) != HalfKAv2::king_bucket(new_sq)
    }
}

/// Accumulator stack for search - stores accumulators at each ply.
pub struct AccumulatorStack {
    stack: Vec<Accumulator>,
    ply: usize,
}

impl AccumulatorStack {
    /// Create a new accumulator stack with initial accumulator.
    pub fn new(initial: Accumulator) -> Self {
        let mut stack = Vec::with_capacity(128);
        stack.push(initial);
        Self { stack, ply: 0 }
    }

    /// Get the current accumulator.
    #[inline]
    pub fn current(&self) -> &Accumulator {
        &self.stack[self.ply]
    }

    /// Get the current accumulator mutably.
    #[inline]
    pub fn current_mut(&mut self) -> &mut Accumulator {
        &mut self.stack[self.ply]
    }

    /// Push a new ply (copy current accumulator).
    pub fn push(&mut self) {
        self.ply += 1;
        if self.ply >= self.stack.len() {
            self.stack.push(self.stack[self.ply - 1].clone());
        } else {
            self.stack[self.ply] = self.stack[self.ply - 1].clone();
        }
    }

    /// Pop back to previous ply.
    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.ply > 0);
        self.ply -= 1;
    }

    /// Reset to root position.
    pub fn reset(&mut self, initial: Accumulator) {
        self.stack[0] = initial;
        self.ply = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_from_position() {
        let pos = Position::startpos();
        let acc = Accumulator::from_position(&pos);

        // Should have some non-zero values from the 32 pieces
        let white_sum: i16 = acc.white.iter().sum();
        let black_sum: i16 = acc.black.iter().sum();

        assert!(white_sum > 0, "White accumulator should have values");
        assert!(black_sum > 0, "Black accumulator should have values");
    }

    #[test]
    fn test_accumulator_stack() {
        let pos = Position::startpos();
        let initial = Accumulator::from_position(&pos);
        let mut stack = AccumulatorStack::new(initial);

        // Push a few plies
        stack.push();
        stack.current_mut().white[0] = 100;
        stack.push();
        stack.current_mut().white[0] = 200;

        assert_eq!(stack.current().white[0], 200);

        stack.pop();
        assert_eq!(stack.current().white[0], 100);

        stack.pop();
        assert_ne!(stack.current().white[0], 100);
    }

    #[test]
    fn test_king_bucket_changed() {
        // Same bucket after mirroring (A1 mirrors to file 0, H1 mirrors to file 0)
        assert!(!Accumulator::king_bucket_changed(Square::A1, Square::H1));
        // Same bucket (B1 mirrors to file 1, G1 mirrors to file 1)
        assert!(!Accumulator::king_bucket_changed(Square::B1, Square::G1));
        // Same bucket (D1 and E1 both mirror to file 3)
        assert!(!Accumulator::king_bucket_changed(Square::D1, Square::E1));
        // Different bucket (A1 is file 0, B1 is file 1)
        assert!(Accumulator::king_bucket_changed(Square::A1, Square::B1));
        // Different bucket (different rank)
        assert!(Accumulator::king_bucket_changed(Square::A1, Square::A2));
    }
}
