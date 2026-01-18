//! NNUE network architecture.
//!
//! Architecture:
//! ```text
//! Feature Transformer: 22,528 → L1_SIZE (shared, from Stockfish)
//!                          ↓
//!                     [white_acc, black_acc] = L1_SIZE * 2
//!                          ↓
//!          ┌───────────────┴───────────────┐
//!          ↓                               ↓
//!     Value Head                      Policy Head
//!     L1*2 → 15 → 32 → 5              L1*2 → 64 (from) + 64 (to)
//!     (quantiles)                     (move scores)
//! ```

use crate::eval::Quantiles;
use crate::types::Color;

use super::accumulator::Accumulator;
use super::features::HALF_DIMENSIONS;

/// Feature transformer output size (accumulator size per side).
pub const L1_SIZE: usize = 128;

/// Value head hidden layer 1 size.
pub const VALUE_HIDDEN1: usize = 15;

/// Value head hidden layer 2 size.
pub const VALUE_HIDDEN2: usize = 32;

/// Number of quantile outputs.
pub const NUM_QUANTILES: usize = 5;

/// Policy head output size (from squares + to squares).
pub const POLICY_OUTPUT: usize = 128; // 64 from + 64 to

/// Output from the policy head - scores for from/to squares.
#[derive(Clone)]
pub struct PolicyOutput {
    pub from: [i32; 64],
    pub to: [i32; 64],
}

impl Default for PolicyOutput {
    fn default() -> Self {
        Self {
            from: [0; 64],
            to: [0; 64],
        }
    }
}

/// Feature transformer weights and biases.
pub struct FeatureTransformer {
    /// Weights: HALF_DIMENSIONS x L1_SIZE (stored column-major for cache efficiency)
    pub weights: Vec<i16>,
    /// Biases: L1_SIZE
    pub biases: Vec<i16>,
}

impl Default for FeatureTransformer {
    fn default() -> Self {
        Self {
            weights: vec![0; HALF_DIMENSIONS * L1_SIZE],
            biases: vec![0; L1_SIZE],
        }
    }
}

impl FeatureTransformer {
    /// Create a new feature transformer with random weights for testing.
    pub fn new_random() -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut weights = vec![0i16; HALF_DIMENSIONS * L1_SIZE];
        let mut biases = vec![0i16; L1_SIZE];

        let mut seed = 0x12345678u64;
        let mut next = || {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            ((seed >> 32) as i16) >> 8 // Small values
        };

        for w in &mut weights {
            *w = next();
        }
        for b in &mut biases {
            *b = next();
        }

        Self { weights, biases }
    }

    /// Get weight for feature index and output index.
    #[inline]
    pub fn weight(&self, feature_idx: usize, output_idx: usize) -> i16 {
        self.weights[feature_idx * L1_SIZE + output_idx]
    }

    /// Get the weights for a specific feature as a slice.
    #[inline]
    pub fn feature_weights(&self, feature_idx: usize) -> &[i16] {
        let start = feature_idx * L1_SIZE;
        &self.weights[start..start + L1_SIZE]
    }
}

/// Value head: takes concatenated accumulators → quantiles.
pub struct ValueHead {
    /// First layer: (L1_SIZE * 2) → VALUE_HIDDEN1
    pub w1: Vec<i8>,
    pub b1: Vec<i32>,
    /// Second layer: VALUE_HIDDEN1 → VALUE_HIDDEN2
    pub w2: Vec<i8>,
    pub b2: Vec<i32>,
    /// Output layer: VALUE_HIDDEN2 → NUM_QUANTILES
    pub w3: Vec<i8>,
    pub b3: Vec<i32>,
}

impl Default for ValueHead {
    fn default() -> Self {
        Self {
            w1: vec![0; L1_SIZE * 2 * VALUE_HIDDEN1],
            b1: vec![0; VALUE_HIDDEN1],
            w2: vec![0; VALUE_HIDDEN1 * VALUE_HIDDEN2],
            b2: vec![0; VALUE_HIDDEN2],
            w3: vec![0; VALUE_HIDDEN2 * NUM_QUANTILES],
            b3: vec![0; NUM_QUANTILES],
        }
    }
}

impl ValueHead {
    /// Create a value head with random weights for testing.
    pub fn new_random() -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut head = Self::default();

        let mut seed = 0xABCDEF01u64;
        let mut next = |shift_i8: bool| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            if shift_i8 {
                (seed >> 56) as i64
            } else {
                (seed >> 40) as i64
            }
        };

        for w in &mut head.w1 {
            *w = next(true) as i8;
        }
        for b in &mut head.b1 {
            *b = next(false) as i32;
        }
        for w in &mut head.w2 {
            *w = next(true) as i8;
        }
        for b in &mut head.b2 {
            *b = next(false) as i32;
        }
        for w in &mut head.w3 {
            *w = next(true) as i8;
        }
        for b in &mut head.b3 {
            *b = next(false) as i32;
        }

        head
    }
}

/// Policy head: takes concatenated accumulators → move scores.
pub struct PolicyHead {
    /// Single layer: (L1_SIZE * 2) → POLICY_OUTPUT
    pub weights: Vec<i8>,
    pub biases: Vec<i32>,
}

impl Default for PolicyHead {
    fn default() -> Self {
        Self {
            weights: vec![0; L1_SIZE * 2 * POLICY_OUTPUT],
            biases: vec![0; POLICY_OUTPUT],
        }
    }
}

impl PolicyHead {
    /// Create a policy head with random weights for testing.
    pub fn new_random() -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut head = Self::default();

        let mut seed = 0x98765432u64;
        let mut next = |shift_i8: bool| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            if shift_i8 {
                (seed >> 56) as i64
            } else {
                (seed >> 40) as i64
            }
        };

        for w in &mut head.weights {
            *w = next(true) as i8;
        }
        for b in &mut head.biases {
            *b = next(false) as i32;
        }

        head
    }
}

/// Complete NNUE network.
pub struct Network {
    pub feature_transformer: FeatureTransformer,
    pub value_head: ValueHead,
    pub policy_head: PolicyHead,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            feature_transformer: FeatureTransformer::default(),
            value_head: ValueHead::default(),
            policy_head: PolicyHead::default(),
        }
    }
}

impl Network {
    /// Create a network with random weights for testing.
    pub fn new_random() -> Self {
        Self {
            feature_transformer: FeatureTransformer::new_random(),
            value_head: ValueHead::new_random(),
            policy_head: PolicyHead::new_random(),
        }
    }

    /// Forward pass through value head to get quantiles.
    pub fn forward_value(&self, acc: &Accumulator, stm: Color) -> Quantiles {
        // Concatenate accumulators based on side to move
        let (first, second) = if stm == Color::White {
            (&acc.white, &acc.black)
        } else {
            (&acc.black, &acc.white)
        };

        // Apply clipped ReLU and prepare input
        let mut input = [0i8; L1_SIZE * 2];
        for (i, &v) in first.iter().enumerate() {
            input[i] = clipped_relu(v);
        }
        for (i, &v) in second.iter().enumerate() {
            input[L1_SIZE + i] = clipped_relu(v);
        }

        // Layer 1
        let mut hidden1 = [0i32; VALUE_HIDDEN1];
        for (i, h) in hidden1.iter_mut().enumerate() {
            let mut sum = self.value_head.b1[i];
            for (j, &x) in input.iter().enumerate() {
                sum += (x as i32) * (self.value_head.w1[j * VALUE_HIDDEN1 + i] as i32);
            }
            *h = sum;
        }

        // ReLU activation
        let mut act1 = [0i8; VALUE_HIDDEN1];
        for (i, &h) in hidden1.iter().enumerate() {
            act1[i] = clipped_relu_i32(h);
        }

        // Layer 2
        let mut hidden2 = [0i32; VALUE_HIDDEN2];
        for (i, h) in hidden2.iter_mut().enumerate() {
            let mut sum = self.value_head.b2[i];
            for (j, &x) in act1.iter().enumerate() {
                sum += (x as i32) * (self.value_head.w2[j * VALUE_HIDDEN2 + i] as i32);
            }
            *h = sum;
        }

        // ReLU activation
        let mut act2 = [0i8; VALUE_HIDDEN2];
        for (i, &h) in hidden2.iter().enumerate() {
            act2[i] = clipped_relu_i32(h);
        }

        // Output layer
        let mut output = [0i32; NUM_QUANTILES];
        for (i, o) in output.iter_mut().enumerate() {
            let mut sum = self.value_head.b3[i];
            for (j, &x) in act2.iter().enumerate() {
                sum += (x as i32) * (self.value_head.w3[j * NUM_QUANTILES + i] as i32);
            }
            *o = sum;
        }

        // Scale outputs to centipawns (assuming 256 scale factor)
        Quantiles::new(
            (output[0] / 256) as i16,
            (output[1] / 256) as i16,
            (output[2] / 256) as i16,
            (output[3] / 256) as i16,
            (output[4] / 256) as i16,
        )
    }

    /// Forward pass through policy head to get move scores.
    pub fn forward_policy(&self, acc: &Accumulator, stm: Color) -> PolicyOutput {
        // Concatenate accumulators based on side to move
        let (first, second) = if stm == Color::White {
            (&acc.white, &acc.black)
        } else {
            (&acc.black, &acc.white)
        };

        // Apply clipped ReLU and prepare input
        let mut input = [0i8; L1_SIZE * 2];
        for (i, &v) in first.iter().enumerate() {
            input[i] = clipped_relu(v);
        }
        for (i, &v) in second.iter().enumerate() {
            input[L1_SIZE + i] = clipped_relu(v);
        }

        // Single layer to output
        let mut output = PolicyOutput::default();

        // From-square scores
        for i in 0..64 {
            let mut sum = self.policy_head.biases[i];
            for (j, &x) in input.iter().enumerate() {
                sum += (x as i32) * (self.policy_head.weights[j * POLICY_OUTPUT + i] as i32);
            }
            output.from[i] = sum / 256; // Scale down
        }

        // To-square scores
        for i in 0..64 {
            let mut sum = self.policy_head.biases[64 + i];
            for (j, &x) in input.iter().enumerate() {
                sum += (x as i32) * (self.policy_head.weights[j * POLICY_OUTPUT + 64 + i] as i32);
            }
            output.to[i] = sum / 256; // Scale down
        }

        output
    }
}

/// Clipped ReLU: clamp to [0, 127] and convert to i8.
#[inline]
fn clipped_relu(x: i16) -> i8 {
    x.clamp(0, 127) as i8
}

/// Clipped ReLU for i32: clamp to [0, 127] and convert to i8.
#[inline]
fn clipped_relu_i32(x: i32) -> i8 {
    (x >> 6).clamp(0, 127) as i8 // Adjust for quantization scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_network_forward() {
        let network = Network::new_random();
        let pos = Position::startpos();
        let acc = Accumulator::from_position(&pos);

        // Just verify it runs without crashing
        // With random weights, output values may be arbitrary
        let _quantiles = network.forward_value(&acc, Color::White);
    }

    #[test]
    fn test_policy_forward() {
        let network = Network::new_random();
        let pos = Position::startpos();
        let acc = Accumulator::from_position(&pos);

        let policy = network.forward_policy(&acc, Color::White);
        // Just verify it produces some scores
        let sum: i32 = policy.from.iter().sum::<i32>() + policy.to.iter().sum::<i32>();
        assert!(sum != 0 || policy.from[0] != policy.from[63]);
    }

    #[test]
    fn test_clipped_relu() {
        assert_eq!(clipped_relu(-100), 0);
        assert_eq!(clipped_relu(0), 0);
        assert_eq!(clipped_relu(50), 50);
        assert_eq!(clipped_relu(127), 127);
        assert_eq!(clipped_relu(200), 127);
    }
}
