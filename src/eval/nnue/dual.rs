//! Dual-NNUE evaluator with uncertainty-based gating.
//!
//! Uses a small, fast NNUE for most positions, but falls back to a
//! larger, more accurate NNUE when the small one reports high uncertainty.
//!
//! This approach balances speed and accuracy:
//! - Most positions are "clear" (one side winning) - small NNUE handles them
//! - Complex positions (sacrifices, compensation) trigger the large NNUE

use crate::eval::{Evaluator, Quantiles};
use crate::position::Position;

use super::network::{Network, PolicyOutput};
use super::Accumulator;

/// Threshold for uncertainty gating.
/// When small NNUE's (q90 - q10) exceeds this, use large NNUE.
const UNCERTAINTY_THRESHOLD: i16 = 100;

/// Dual-NNUE evaluator with uncertainty-based selection.
pub struct DualNnueEvaluator {
    /// Small network - fast, always runs first
    small: Network,
    /// Large network - slower but more accurate, runs when small is uncertain
    large: Option<Network>,
    /// Statistics for monitoring
    stats: DualNnueStats,
}

/// Statistics for monitoring dual-NNUE performance.
#[derive(Default, Clone)]
pub struct DualNnueStats {
    /// Number of evaluations using small NNUE only
    pub small_only: u64,
    /// Number of evaluations that triggered large NNUE
    pub large_triggered: u64,
}

impl DualNnueEvaluator {
    /// Create a new dual evaluator with just the small network.
    /// Large network can be added later.
    pub fn new(small: Network) -> Self {
        Self {
            small,
            large: None,
            stats: DualNnueStats::default(),
        }
    }

    /// Create a dual evaluator with both networks.
    pub fn with_large(small: Network, large: Network) -> Self {
        Self {
            small,
            large: Some(large),
            stats: DualNnueStats::default(),
        }
    }

    /// Get evaluation statistics.
    pub fn stats(&self) -> &DualNnueStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DualNnueStats::default();
    }

    /// Get policy scores from the small network.
    pub fn get_policy(&self, pos: &Position) -> PolicyOutput {
        let acc = Accumulator::from_position(pos);
        self.small.forward_policy(&acc, pos.side_to_move())
    }

    /// Evaluate with uncertainty gating.
    fn evaluate_gated(&mut self, pos: &Position) -> Quantiles {
        let acc = Accumulator::from_position(pos);
        let stm = pos.side_to_move();

        // Always run small NNUE first
        let small_quantiles = self.small.forward_value(&acc, stm);

        // Check if we should use large NNUE
        if let Some(ref large) = self.large {
            let uncertainty = small_quantiles.uncertainty();
            if uncertainty > UNCERTAINTY_THRESHOLD {
                self.stats.large_triggered += 1;
                // Use large NNUE for complex positions
                return large.forward_value(&acc, stm);
            }
        }

        self.stats.small_only += 1;
        small_quantiles
    }
}

impl Evaluator for DualNnueEvaluator {
    fn evaluate(&self, pos: &Position) -> Quantiles {
        let acc = Accumulator::from_position(pos);
        let stm = pos.side_to_move();

        // Always run small NNUE first
        let small_quantiles = self.small.forward_value(&acc, stm);

        // Check if we should use large NNUE
        // Note: Using immutable reference, so can't update stats here
        // For stats tracking, use evaluate_gated on &mut self
        if let Some(ref large) = self.large {
            let uncertainty = small_quantiles.uncertainty();
            if uncertainty > UNCERTAINTY_THRESHOLD {
                return large.forward_value(&acc, stm);
            }
        }

        small_quantiles
    }
}

/// Statistics display
impl std::fmt::Display for DualNnueStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.small_only + self.large_triggered;
        if total == 0 {
            return write!(f, "No evaluations yet");
        }
        let large_pct = (self.large_triggered as f64 / total as f64) * 100.0;
        write!(
            f,
            "Dual-NNUE stats: {} total evals, {} small-only ({:.1}%), {} large-triggered ({:.1}%)",
            total,
            self.small_only,
            100.0 - large_pct,
            self.large_triggered,
            large_pct
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_evaluator_small_only() {
        let small = Network::new_random();
        let eval = DualNnueEvaluator::new(small);

        let pos = Position::startpos();
        let _ = eval.evaluate(&pos);

        // Should work without crashing
    }

    #[test]
    fn test_dual_evaluator_with_large() {
        let small = Network::new_random();
        let large = Network::new_random();
        let eval = DualNnueEvaluator::with_large(small, large);

        let pos = Position::startpos();
        let _ = eval.evaluate(&pos);

        // Should work without crashing
    }
}
