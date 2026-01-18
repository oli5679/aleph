//! NNUE feature extraction using HalfKAv2_hm (Half-King-A-piece version 2, horizontally mirrored).
//!
//! This feature set is compatible with Stockfish's NNUE architecture.
//!
//! Feature index calculation:
//! - 32 king buckets (4 files × 8 ranks, with e-h files mirrored to a-d)
//! - 11 piece types (6 own pieces + 5 opponent pieces, excluding opponent king)
//! - 64 squares for each piece
//!
//! Total features per perspective: 32 × 11 × 64 = 22,528

use crate::types::{Color, Piece, Square};

/// Number of king buckets (horizontally mirrored: files a-d × ranks 1-8).
pub const NUM_KING_BUCKETS: usize = 32;

/// Number of piece-square combinations (11 piece types × 64 squares).
/// Piece types: own P, N, B, R, Q, K (6) + opponent P, N, B, R, Q (5) = 11
/// (Opponent king is not included as a feature since it's always present)
pub const NUM_PIECE_SQUARES: usize = 11 * 64;

/// Total features per perspective.
pub const HALF_DIMENSIONS: usize = NUM_KING_BUCKETS * NUM_PIECE_SQUARES;

/// HalfKAv2 feature set - Stockfish compatible.
pub struct HalfKAv2;

impl HalfKAv2 {
    /// Get the king bucket index for a king square.
    /// Files e-h are mirrored to a-d for horizontal symmetry.
    #[inline]
    pub fn king_bucket(king_sq: Square) -> usize {
        let file = king_sq.file();
        let rank = king_sq.rank();
        // Mirror files e-h (4-7) to a-d (3-0)
        let mirrored_file = if file >= 4 { 7 - file } else { file };
        (rank as usize) * 4 + (mirrored_file as usize)
    }

    /// Check if a square needs horizontal mirroring based on king position.
    #[inline]
    pub fn needs_mirror(king_sq: Square) -> bool {
        king_sq.file() >= 4
    }

    /// Mirror a square horizontally (a-file <-> h-file).
    #[inline]
    pub fn mirror_square(sq: Square) -> Square {
        Square::new(7 - sq.file(), sq.rank())
    }

    /// Get the piece type index (0-10) for feature calculation.
    /// Own pieces: P=0, N=1, B=2, R=3, Q=4, K=5
    /// Opponent pieces: P=6, N=7, B=8, R=9, Q=10
    #[inline]
    pub fn piece_type_index(piece: Piece, is_own: bool) -> usize {
        let base = piece.index();
        if is_own {
            base
        } else {
            6 + base // Opponent pieces start at index 6
        }
    }

    /// Calculate the feature index for a piece from a specific perspective.
    ///
    /// # Arguments
    /// * `perspective` - The color whose perspective we're calculating from
    /// * `piece_sq` - The square the piece is on
    /// * `piece` - The piece type
    /// * `piece_color` - The color of the piece
    /// * `king_sq` - The king square for this perspective
    ///
    /// # Returns
    /// The feature index in [0, HALF_DIMENSIONS)
    #[inline]
    pub fn make_index(
        perspective: Color,
        piece_sq: Square,
        piece: Piece,
        piece_color: Color,
        king_sq: Square,
    ) -> usize {
        // Skip opponent king (it's implicit)
        if piece == Piece::King && piece_color != perspective {
            return usize::MAX; // Invalid index - caller should skip
        }

        let bucket = Self::king_bucket(king_sq);
        let mirror = Self::needs_mirror(king_sq);

        // Orient square based on perspective and mirroring
        let mut sq = if perspective == Color::Black {
            piece_sq.flip_rank()
        } else {
            piece_sq
        };
        if mirror {
            sq = Self::mirror_square(sq);
        }

        let is_own = piece_color == perspective;
        let piece_idx = Self::piece_type_index(piece, is_own);

        bucket * NUM_PIECE_SQUARES + piece_idx * 64 + sq.index()
    }

    /// Get all active feature indices for a position from one perspective.
    /// Returns a vector of feature indices.
    pub fn get_active_features(
        perspective: Color,
        king_sq: Square,
        pieces: impl Iterator<Item = (Square, Color, Piece)>,
    ) -> Vec<usize> {
        let mut features = Vec::with_capacity(32);

        for (sq, color, piece) in pieces {
            let idx = Self::make_index(perspective, sq, piece, color, king_sq);
            if idx != usize::MAX {
                features.push(idx);
            }
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_bucket() {
        // a1 -> bucket 0
        assert_eq!(HalfKAv2::king_bucket(Square::A1), 0);
        // d1 -> bucket 3
        assert_eq!(HalfKAv2::king_bucket(Square::D1), 3);
        // e1 -> mirrors to d1, bucket 3
        assert_eq!(HalfKAv2::king_bucket(Square::E1), 3);
        // h1 -> mirrors to a1, bucket 0
        assert_eq!(HalfKAv2::king_bucket(Square::H1), 0);
        // a8 -> bucket 28
        assert_eq!(HalfKAv2::king_bucket(Square::A8), 28);
    }

    #[test]
    fn test_mirror() {
        assert!(!HalfKAv2::needs_mirror(Square::A1));
        assert!(!HalfKAv2::needs_mirror(Square::D4));
        assert!(HalfKAv2::needs_mirror(Square::E1));
        assert!(HalfKAv2::needs_mirror(Square::H8));

        assert_eq!(HalfKAv2::mirror_square(Square::A1), Square::H1);
        assert_eq!(HalfKAv2::mirror_square(Square::E4), Square::D4);
    }

    #[test]
    fn test_piece_type_index() {
        // Own pieces
        assert_eq!(HalfKAv2::piece_type_index(Piece::Pawn, true), 0);
        assert_eq!(HalfKAv2::piece_type_index(Piece::King, true), 5);

        // Opponent pieces
        assert_eq!(HalfKAv2::piece_type_index(Piece::Pawn, false), 6);
        assert_eq!(HalfKAv2::piece_type_index(Piece::Queen, false), 10);
    }

    #[test]
    fn test_feature_index_bounds() {
        // Test that indices are within bounds
        for king_file in 0..8u8 {
            for king_rank in 0..8u8 {
                let king_sq = Square::new(king_file, king_rank);

                for sq_file in 0..8u8 {
                    for sq_rank in 0..8u8 {
                        let sq = Square::new(sq_file, sq_rank);

                        for piece in Piece::ALL {
                            for &color in &[Color::White, Color::Black] {
                                let idx = HalfKAv2::make_index(
                                    Color::White,
                                    sq,
                                    piece,
                                    color,
                                    king_sq,
                                );
                                if idx != usize::MAX {
                                    assert!(
                                        idx < HALF_DIMENSIONS,
                                        "Index {} out of bounds for piece {:?}",
                                        idx,
                                        piece
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
