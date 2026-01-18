//! Shared piece values used across evaluation and move ordering.

use crate::types::Piece;

/// Piece values in centipawns.
/// These are used for MVV-LVA move ordering and SEE.
pub const PIECE_VALUES: [i16; 6] = [
    100,   // Pawn
    320,   // Knight
    330,   // Bishop
    500,   // Rook
    900,   // Queen
    20000, // King (nominally infinite, but must fit in i16)
];

/// Get the value of a piece in centipawns.
#[inline]
pub fn piece_value(piece: Piece) -> i16 {
    PIECE_VALUES[piece.index()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_values() {
        assert_eq!(piece_value(Piece::Pawn), 100);
        assert_eq!(piece_value(Piece::Knight), 320);
        assert_eq!(piece_value(Piece::Bishop), 330);
        assert_eq!(piece_value(Piece::Rook), 500);
        assert_eq!(piece_value(Piece::Queen), 900);
        assert_eq!(piece_value(Piece::King), 20000);
    }
}
