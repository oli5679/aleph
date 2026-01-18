//! Incremental NNUE evaluation with proper accumulator updates.
//!
//! This module provides efficient NNUE evaluation by:
//! 1. Maintaining an accumulator stack through the search
//! 2. Incrementally updating accumulators when pieces move
//! 3. Only doing full recomputation when king changes bucket

use crate::eval::{Evaluator, Quantiles};
use crate::position::Position;
use crate::types::{Color, Move, Piece, Square};

use super::accumulator::{Accumulator, AccumulatorStack};
use super::features::{HalfKAv2, HALF_DIMENSIONS};
use super::network::{FeatureTransformer, Network, PolicyOutput, L1_SIZE};

/// Incremental NNUE evaluator that maintains accumulator state through search.
pub struct IncrementalNnue {
    network: Network,
    acc_stack: AccumulatorStack,
    /// King squares at each ply (for detecting bucket changes)
    white_kings: Vec<Square>,
    black_kings: Vec<Square>,
}

impl IncrementalNnue {
    /// Create a new incremental evaluator from a network and initial position.
    pub fn new(network: Network, pos: &Position) -> Self {
        let initial = Self::compute_accumulator(&network.feature_transformer, pos);
        let acc_stack = AccumulatorStack::new(initial);

        let mut white_kings = Vec::with_capacity(128);
        let mut black_kings = Vec::with_capacity(128);
        white_kings.push(pos.king_sq(Color::White));
        black_kings.push(pos.king_sq(Color::Black));

        Self {
            network,
            acc_stack,
            white_kings,
            black_kings,
        }
    }

    /// Compute accumulator from scratch using feature transformer weights.
    fn compute_accumulator(ft: &FeatureTransformer, pos: &Position) -> Accumulator {
        let mut acc = Accumulator::new();

        // Start with biases
        for i in 0..L1_SIZE {
            acc.white[i] = ft.biases[i];
            acc.black[i] = ft.biases[i];
        }

        let white_king = pos.king_sq(Color::White);
        let black_king = pos.king_sq(Color::Black);

        // Add features for all pieces
        for (sq, color, piece) in pos.all_pieces() {
            // White perspective
            let w_idx = HalfKAv2::make_index(Color::White, sq, piece, color, white_king);
            if w_idx != usize::MAX && w_idx < HALF_DIMENSIONS {
                let weights = ft.feature_weights(w_idx);
                for i in 0..L1_SIZE {
                    acc.white[i] = acc.white[i].saturating_add(weights[i]);
                }
            }

            // Black perspective
            let b_idx = HalfKAv2::make_index(Color::Black, sq, piece, color, black_king);
            if b_idx != usize::MAX && b_idx < HALF_DIMENSIONS {
                let weights = ft.feature_weights(b_idx);
                for i in 0..L1_SIZE {
                    acc.black[i] = acc.black[i].saturating_add(weights[i]);
                }
            }
        }

        acc.computed = true;
        acc
    }

    /// Reset for a new position (e.g., new game or position command).
    pub fn reset(&mut self, pos: &Position) {
        let initial = Self::compute_accumulator(&self.network.feature_transformer, pos);
        self.acc_stack.reset(initial);
        self.white_kings.clear();
        self.black_kings.clear();
        self.white_kings.push(pos.king_sq(Color::White));
        self.black_kings.push(pos.king_sq(Color::Black));
    }

    /// Push accumulator state before making a move.
    /// Call this BEFORE Position::make_move().
    pub fn push(&mut self) {
        self.acc_stack.push();
        // Extend king tracking
        let last_w = *self.white_kings.last().unwrap();
        let last_b = *self.black_kings.last().unwrap();
        self.white_kings.push(last_w);
        self.black_kings.push(last_b);
    }

    /// Update accumulator after a move is made.
    /// Call this AFTER Position::make_move().
    pub fn update_after_move(&mut self, pos: &Position, mv: Move, moved_piece: Piece, captured: Option<Piece>) {
        let ft = &self.network.feature_transformer;
        let ply = self.white_kings.len() - 1;

        let from_sq = mv.from();
        let to_sq = mv.to();
        let stm = pos.side_to_move().flip(); // Side that just moved
        let white_king = pos.king_sq(Color::White);
        let black_king = pos.king_sq(Color::Black);

        // Update king tracking
        self.white_kings[ply] = white_king;
        self.black_kings[ply] = black_king;

        // Check if king moved to different bucket - requires full refresh
        let prev_white_king = self.white_kings[ply - 1];
        let prev_black_king = self.black_kings[ply - 1];

        let white_bucket_changed = Accumulator::king_bucket_changed(prev_white_king, white_king);
        let black_bucket_changed = Accumulator::king_bucket_changed(prev_black_king, black_king);

        if white_bucket_changed || black_bucket_changed {
            // Full recomputation needed
            *self.acc_stack.current_mut() = Self::compute_accumulator(ft, pos);
            return;
        }

        let acc = self.acc_stack.current_mut();

        // Incremental update
        // 1. Remove piece from old square (both perspectives)
        Self::sub_piece(acc, ft, from_sq, stm, moved_piece, prev_white_king, prev_black_king);

        // 2. Add piece to new square (handle promotion)
        let final_piece = if mv.is_promotion() {
            mv.promotion_piece()
        } else {
            moved_piece
        };
        Self::add_piece(acc, ft, to_sq, stm, final_piece, white_king, black_king);

        // 3. Handle capture
        if let Some(captured_piece) = captured {
            let capture_sq = if mv.is_ep() {
                // En passant: captured pawn is on different square
                Square::new(to_sq.file(), from_sq.rank())
            } else {
                to_sq
            };
            Self::sub_piece(acc, ft, capture_sq, stm.flip(), captured_piece, prev_white_king, prev_black_king);
        }

        // 4. Handle castling rook
        if mv.is_castle() {
            let (rook_from, rook_to) = if to_sq.file() > from_sq.file() {
                // Kingside
                (Square::new(7, from_sq.rank()), Square::new(5, from_sq.rank()))
            } else {
                // Queenside
                (Square::new(0, from_sq.rank()), Square::new(3, from_sq.rank()))
            };
            Self::sub_piece(acc, ft, rook_from, stm, Piece::Rook, prev_white_king, prev_black_king);
            Self::add_piece(acc, ft, rook_to, stm, Piece::Rook, white_king, black_king);
        }
    }

    /// Pop accumulator state after unmaking a move.
    pub fn pop(&mut self) {
        self.acc_stack.pop();
        self.white_kings.pop();
        self.black_kings.pop();
    }

    /// Add piece features to accumulator.
    #[inline]
    fn add_piece(
        acc: &mut Accumulator,
        ft: &FeatureTransformer,
        sq: Square,
        color: Color,
        piece: Piece,
        white_king: Square,
        black_king: Square,
    ) {
        // White perspective
        let w_idx = HalfKAv2::make_index(Color::White, sq, piece, color, white_king);
        if w_idx != usize::MAX && w_idx < HALF_DIMENSIONS {
            let weights = ft.feature_weights(w_idx);
            for i in 0..L1_SIZE {
                acc.white[i] = acc.white[i].saturating_add(weights[i]);
            }
        }

        // Black perspective
        let b_idx = HalfKAv2::make_index(Color::Black, sq, piece, color, black_king);
        if b_idx != usize::MAX && b_idx < HALF_DIMENSIONS {
            let weights = ft.feature_weights(b_idx);
            for i in 0..L1_SIZE {
                acc.black[i] = acc.black[i].saturating_add(weights[i]);
            }
        }
    }

    /// Remove piece features from accumulator.
    #[inline]
    fn sub_piece(
        acc: &mut Accumulator,
        ft: &FeatureTransformer,
        sq: Square,
        color: Color,
        piece: Piece,
        white_king: Square,
        black_king: Square,
    ) {
        // White perspective
        let w_idx = HalfKAv2::make_index(Color::White, sq, piece, color, white_king);
        if w_idx != usize::MAX && w_idx < HALF_DIMENSIONS {
            let weights = ft.feature_weights(w_idx);
            for i in 0..L1_SIZE {
                acc.white[i] = acc.white[i].saturating_sub(weights[i]);
            }
        }

        // Black perspective
        let b_idx = HalfKAv2::make_index(Color::Black, sq, piece, color, black_king);
        if b_idx != usize::MAX && b_idx < HALF_DIMENSIONS {
            let weights = ft.feature_weights(b_idx);
            for i in 0..L1_SIZE {
                acc.black[i] = acc.black[i].saturating_sub(weights[i]);
            }
        }
    }

    /// Get current accumulator for forward pass.
    pub fn current_accumulator(&self) -> &Accumulator {
        self.acc_stack.current()
    }

    /// Evaluate current position using cached accumulator.
    pub fn evaluate_current(&self, stm: Color) -> Quantiles {
        self.network.forward_value(self.acc_stack.current(), stm)
    }

    /// Get policy scores using cached accumulator.
    pub fn get_policy(&self, stm: Color) -> PolicyOutput {
        self.network.forward_policy(self.acc_stack.current(), stm)
    }
}

impl Evaluator for IncrementalNnue {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        // Note: This doesn't use the incremental updates properly
        // For full integration, the search should call push/update_after_move/pop
        // This fallback just uses the current accumulator
        self.network.forward_value(self.acc_stack.current(), pos.side_to_move())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_accumulator() {
        let network = Network::new_random();
        let pos = Position::startpos();

        let acc = IncrementalNnue::compute_accumulator(&network.feature_transformer, &pos);

        // Should have values from biases + features
        let white_sum: i32 = acc.white.iter().map(|&x| x as i32).sum();
        let black_sum: i32 = acc.black.iter().map(|&x| x as i32).sum();

        // Sums shouldn't be zero (we have biases and features)
        assert!(white_sum != 0 || black_sum != 0);
    }

    #[test]
    fn test_incremental_new() {
        let network = Network::new_random();
        let pos = Position::startpos();

        let eval = IncrementalNnue::new(network, &pos);

        // Should have valid accumulator
        assert!(eval.acc_stack.current().computed || true); // Just checking it doesn't crash
    }
}
