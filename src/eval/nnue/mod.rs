//! NNUE (Efficiently Updatable Neural Network) evaluation.
//!
//! This module implements a dual-head neural network:
//! - Value head: 5 quantile outputs (q10, q25, q50, q75, q90) for learned pruning
//! - Policy head: Move ordering scores for learned move ordering
//!
//! The architecture uses transfer learning from Stockfish's NNUE feature transformer.

pub mod accumulator;
pub mod features;
pub mod loader;
pub mod network;
pub mod stockfish;

use crate::eval::{Evaluator, Quantiles};
use crate::position::Position;
use crate::types::Move;

pub use accumulator::Accumulator;
pub use features::HalfKAv2;
pub use network::{Network, PolicyOutput};

/// NNUE Evaluator implementing the Evaluator trait.
/// Uses a neural network to evaluate positions and provide policy hints.
pub struct NnueEvaluator {
    network: Network,
}

impl NnueEvaluator {
    /// Create a new NNUE evaluator with the given network.
    pub fn new(network: Network) -> Self {
        Self { network }
    }

    /// Load an NNUE evaluator from a file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let network = loader::load_network(path)?;
        Ok(Self::new(network))
    }

    /// Get policy scores for move ordering.
    /// Returns scores for each from/to square that can be used to order moves.
    pub fn get_policy(&self, pos: &Position) -> PolicyOutput {
        let acc = Accumulator::from_position(pos);
        self.network.forward_policy(&acc, pos.side_to_move())
    }

    /// Get both value and policy in one forward pass.
    pub fn evaluate_full(&self, pos: &Position) -> (Quantiles, PolicyOutput) {
        let acc = Accumulator::from_position(pos);
        let stm = pos.side_to_move();
        let quantiles = self.network.forward_value(&acc, stm);
        let policy = self.network.forward_policy(&acc, stm);
        (quantiles, policy)
    }
}

impl Evaluator for NnueEvaluator {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        let acc = Accumulator::from_position(pos);
        self.network.forward_value(&acc, pos.side_to_move())
    }
}

/// Score a move using policy network output.
/// Higher scores = better moves (should be searched first).
pub fn policy_score(mv: Move, policy: &PolicyOutput) -> i32 {
    // Combine from-square and to-square scores
    policy.from[mv.from().index()] + policy.to[mv.to().index()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_from_position() {
        let pos = Position::startpos();
        let acc = Accumulator::from_position(&pos);
        // Just verify it doesn't crash and has reasonable values
        assert!(acc.white[0] != 0 || acc.white[1] != 0);
    }
}
