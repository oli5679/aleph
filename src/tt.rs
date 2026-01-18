//! Transposition Table for caching search results.
//!
//! The TT stores previously computed search results to avoid redundant work.
//! It uses Zobrist hashing for position identification and supports:
//! - Move ordering (TT move searched first)
//! - Score cutoffs when depth is sufficient
//! - Aging for replacement strategy

use crate::types::Move;

/// Bound type for TT entries.
/// Determines how the stored score should be interpreted.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Bound {
    /// No valid bound (empty entry)
    None = 0,
    /// Upper bound: score <= stored value (fail-low, no move improved alpha)
    Upper = 1,
    /// Lower bound: score >= stored value (fail-high, beta cutoff)
    Lower = 2,
    /// Exact score (PV node, score is between alpha and beta)
    Exact = 3,
}

impl Bound {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v & 3 {
            1 => Bound::Upper,
            2 => Bound::Lower,
            3 => Bound::Exact,
            _ => Bound::None,
        }
    }
}

/// A single transposition table entry.
/// Packed into 10 bytes for cache efficiency.
///
/// Layout:
/// - key16: u16 - upper 16 bits of hash for verification
/// - mv: u16 - best move found
/// - score: i16 - evaluation score
/// - depth: i8 - search depth
/// - bound_gen: u8 - bound type (2 bits) + generation (6 bits)
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct TTEntry {
    key16: u16,
    mv: u16,
    score: i16,
    depth: i8,
    bound_gen: u8,
}

impl TTEntry {
    const BOUND_MASK: u8 = 0x03;
    const GEN_MASK: u8 = 0xFC;
    const GEN_SHIFT: u8 = 2;

    /// Create a new TT entry.
    #[inline]
    pub fn new(key: u64, mv: Move, score: i16, depth: i8, bound: Bound, generation: u8) -> Self {
        Self {
            key16: (key >> 48) as u16,
            mv: mv.0,
            score,
            depth,
            bound_gen: (bound as u8) | (generation << Self::GEN_SHIFT),
        }
    }

    /// Check if this entry matches the given hash key.
    #[inline]
    pub fn matches(&self, key: u64) -> bool {
        self.key16 == (key >> 48) as u16
    }

    /// Get the stored move.
    #[inline]
    pub fn mv(&self) -> Move {
        Move(self.mv)
    }

    /// Get the stored score.
    #[inline]
    pub fn score(&self) -> i16 {
        self.score
    }

    /// Get the stored depth.
    #[inline]
    pub fn depth(&self) -> i8 {
        self.depth
    }

    /// Get the bound type.
    #[inline]
    pub fn bound(&self) -> Bound {
        Bound::from_u8(self.bound_gen & Self::BOUND_MASK)
    }

    /// Get the generation.
    #[inline]
    pub fn generation(&self) -> u8 {
        self.bound_gen >> Self::GEN_SHIFT
    }

    /// Check if the entry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bound() == Bound::None
    }
}

/// Number of entries per cluster (improves collision handling).
const CLUSTER_SIZE: usize = 3;

/// A cluster of TT entries for better hash collision handling.
#[derive(Copy, Clone, Default)]
#[repr(C)]
struct TTCluster {
    entries: [TTEntry; CLUSTER_SIZE],
    _padding: [u8; 2], // Align to 32 bytes (3 * 10 + 2 = 32)
}

/// Transposition table for caching search results.
pub struct TranspositionTable {
    clusters: Vec<TTCluster>,
    generation: u8,
    mask: usize,
}

impl TranspositionTable {
    /// Create a new transposition table with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;
        let num_clusters = (size_bytes / std::mem::size_of::<TTCluster>()).next_power_of_two() / 2;
        let num_clusters = num_clusters.max(1);

        Self {
            clusters: vec![TTCluster::default(); num_clusters],
            generation: 0,
            mask: num_clusters - 1,
        }
    }

    /// Create a default 16MB table.
    pub fn default_size() -> Self {
        Self::new(16)
    }

    /// Clear the table and reset generation.
    pub fn clear(&mut self) {
        self.clusters.fill(TTCluster::default());
        self.generation = 0;
    }

    /// Start a new search (increment generation for aging).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1) & 0x3F; // 6 bits
    }

    /// Resize the table to a new size in megabytes.
    pub fn resize(&mut self, size_mb: usize) {
        let size_bytes = size_mb * 1024 * 1024;
        let num_clusters = (size_bytes / std::mem::size_of::<TTCluster>()).next_power_of_two() / 2;
        let num_clusters = num_clusters.max(1);

        self.clusters = vec![TTCluster::default(); num_clusters];
        self.mask = num_clusters - 1;
        self.generation = 0;
    }

    /// Get the cluster index for a hash key.
    #[inline]
    fn cluster_index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    /// Probe the table for an entry matching the given key.
    /// Returns (found, entry) where entry contains the data if found.
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let cluster = &self.clusters[self.cluster_index(key)];

        for entry in &cluster.entries {
            if entry.matches(key) && !entry.is_empty() {
                return Some(*entry);
            }
        }

        None
    }

    /// Store an entry in the table.
    /// Uses a replacement strategy that prefers:
    /// 1. Empty slots
    /// 2. Same position (update)
    /// 3. Older entries
    /// 4. Lower depth entries
    pub fn store(&mut self, key: u64, mv: Move, score: i16, depth: i8, bound: Bound) {
        let idx = self.cluster_index(key);
        let cluster = &mut self.clusters[idx];
        let generation = self.generation;

        // Find the best slot to replace
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            // Always replace same position or empty
            if entry.is_empty() || entry.matches(key) {
                replace_idx = i;
                break;
            }

            // Replacement scoring: prefer old entries with low depth
            // Age difference gives big penalty, depth gives smaller penalty
            let age_diff = ((generation as i32) - (entry.generation() as i32) + 64) & 63;
            let entry_score = (entry.depth() as i32) * 8 - age_diff * 2;

            if entry_score < replace_score {
                replace_score = entry_score;
                replace_idx = i;
            }
        }

        // Don't overwrite deeper entries of same position with shallower search
        // unless this is an exact bound (PV node)
        let existing = &cluster.entries[replace_idx];
        if existing.matches(key)
            && !existing.is_empty()
            && existing.depth() > depth
            && bound != Bound::Exact
        {
            // Only update the move if we have one and existing doesn't
            if !mv.is_null() && existing.mv().is_null() {
                cluster.entries[replace_idx].mv = mv.0;
            }
            return;
        }

        cluster.entries[replace_idx] = TTEntry::new(key, mv, score, depth, bound, generation);
    }

    /// Estimate table occupancy (permille, 0-1000).
    pub fn hashfull(&self) -> usize {
        let sample_size = 1000.min(self.clusters.len());
        let mut used = 0;

        for cluster in self.clusters.iter().take(sample_size) {
            for entry in &cluster.entries {
                if !entry.is_empty() && entry.generation() == self.generation {
                    used += 1;
                }
            }
        }

        used * 1000 / (sample_size * CLUSTER_SIZE)
    }

    /// Get the current generation.
    pub fn generation(&self) -> u8 {
        self.generation
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::default_size()
    }
}

/// Adjust a score for TT storage/retrieval.
/// Mate scores need adjustment because they are relative to the root.
#[inline]
pub fn score_to_tt(score: i16, ply: i32) -> i16 {
    if score > 29000 {
        score + ply as i16 // Mate score: add ply to make it absolute
    } else if score < -29000 {
        score - ply as i16 // Being mated: subtract ply
    } else {
        score
    }
}

/// Adjust a TT score for use at the current ply.
#[inline]
pub fn score_from_tt(score: i16, ply: i32) -> i16 {
    if score > 29000 {
        score - ply as i16 // Mate score: subtract ply to make it relative
    } else if score < -29000 {
        score + ply as i16 // Being mated: add ply
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn test_entry_roundtrip() {
        let key = 0x123456789ABCDEF0u64;
        let mv = Move::new(Square::E2, Square::E4);
        let score = 150i16;
        let depth = 8i8;
        let bound = Bound::Exact;
        let gen = 5u8;

        let entry = TTEntry::new(key, mv, score, depth, bound, gen);

        assert!(entry.matches(key));
        assert!(!entry.matches(key ^ 0x0001_0000_0000_0000)); // Change upper bits
        assert_eq!(entry.mv(), mv);
        assert_eq!(entry.score(), score);
        assert_eq!(entry.depth(), depth);
        assert_eq!(entry.bound(), bound);
        assert_eq!(entry.generation(), gen);
    }

    #[test]
    fn test_table_probe_store() {
        let mut tt = TranspositionTable::new(1); // 1 MB

        let key1 = 0x123456789ABCDEF0u64;
        let mv1 = Move::new(Square::E2, Square::E4);

        // Initially empty
        assert!(tt.probe(key1).is_none());

        // Store and retrieve
        tt.store(key1, mv1, 100, 5, Bound::Exact);
        let entry = tt.probe(key1).unwrap();
        assert_eq!(entry.mv(), mv1);
        assert_eq!(entry.score(), 100);
        assert_eq!(entry.depth(), 5);
        assert_eq!(entry.bound(), Bound::Exact);
    }

    #[test]
    fn test_table_replacement() {
        let mut tt = TranspositionTable::new(1);

        let key = 0x123456789ABCDEF0u64;
        let mv1 = Move::new(Square::E2, Square::E4);
        let mv2 = Move::new(Square::D2, Square::D4);

        // Store with depth 5
        tt.store(key, mv1, 100, 5, Bound::Lower);

        // Try to overwrite with depth 3 (should not replace)
        tt.store(key, mv2, 200, 3, Bound::Lower);
        let entry = tt.probe(key).unwrap();
        assert_eq!(entry.depth(), 5);
        assert_eq!(entry.mv(), mv1);

        // Overwrite with depth 7 (should replace)
        tt.store(key, mv2, 200, 7, Bound::Lower);
        let entry = tt.probe(key).unwrap();
        assert_eq!(entry.depth(), 7);
        assert_eq!(entry.mv(), mv2);
    }

    #[test]
    fn test_new_search_generation() {
        let mut tt = TranspositionTable::new(1);
        assert_eq!(tt.generation(), 0);

        tt.new_search();
        assert_eq!(tt.generation(), 1);

        tt.new_search();
        assert_eq!(tt.generation(), 2);
    }

    #[test]
    fn test_score_adjustment() {
        // Normal scores unchanged
        assert_eq!(score_to_tt(100, 5), 100);
        assert_eq!(score_from_tt(100, 5), 100);

        // Mate scores adjusted
        let mate_score = 29500;
        let ply = 3;
        let tt_score = score_to_tt(mate_score, ply);
        assert_eq!(score_from_tt(tt_score, ply), mate_score);

        // Being mated scores adjusted
        let mated_score = -29500;
        let tt_score = score_to_tt(mated_score, ply);
        assert_eq!(score_from_tt(tt_score, ply), mated_score);
    }

    #[test]
    fn test_bound_types() {
        assert_eq!(Bound::from_u8(0), Bound::None);
        assert_eq!(Bound::from_u8(1), Bound::Upper);
        assert_eq!(Bound::from_u8(2), Bound::Lower);
        assert_eq!(Bound::from_u8(3), Bound::Exact);
    }
}
