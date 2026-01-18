//! Syzygy endgame tablebase support.
//!
//! Provides perfect endgame play for positions with few pieces.
//! Uses the shakmaty-syzygy crate for probing.

use crate::movegen::MoveList;
use crate::position::Position;
use crate::types::Move;
use std::path::Path;

/// Win/Draw/Loss outcome from tablebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wdl {
    Loss,
    BlessedLoss, // Cursed win for opponent (draw under 50-move rule)
    Draw,
    CursedWin, // Win but will be draw under 50-move rule
    Win,
}

impl Wdl {
    /// Convert to a score for search.
    /// Returns a score where positive is good for side to move.
    pub fn to_score(self) -> i16 {
        match self {
            Wdl::Loss => -29000,
            Wdl::BlessedLoss => 0,
            Wdl::Draw => 0,
            Wdl::CursedWin => 0,
            Wdl::Win => 29000,
        }
    }

    /// Is this a decisive result?
    pub fn is_decisive(self) -> bool {
        matches!(self, Wdl::Win | Wdl::Loss)
    }
}

/// Syzygy tablebase wrapper.
pub struct Tablebase {
    inner: shakmaty_syzygy::Tablebase<shakmaty::Chess>,
}

impl Tablebase {
    /// Create a new tablebase probing interface.
    /// Path should be a directory containing .rtbw and .rtbz files.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut inner = shakmaty_syzygy::Tablebase::new();
        inner.add_directory(path)?;
        Ok(Self { inner })
    }

    /// Maximum number of pieces supported by loaded tablebases.
    pub fn max_pieces(&self) -> usize {
        self.inner.max_pieces()
    }

    /// Probe WDL (win/draw/loss) for a position.
    /// Returns None if position has too many pieces or isn't in tablebase.
    pub fn probe_wdl(&self, pos: &Position) -> Option<Wdl> {
        let chess_pos = to_shakmaty_position(pos)?;
        let ambiguous_wdl = self.inner.probe_wdl(&chess_pos).ok()?;

        // Convert AmbiguousWdl to our Wdl type
        // Use unambiguous if available, otherwise assume we're after zeroing
        let wdl = ambiguous_wdl
            .unambiguous()
            .unwrap_or_else(|| ambiguous_wdl.after_zeroing());

        Some(match wdl {
            shakmaty_syzygy::Wdl::Loss => Wdl::Loss,
            shakmaty_syzygy::Wdl::BlessedLoss => Wdl::BlessedLoss,
            shakmaty_syzygy::Wdl::Draw => Wdl::Draw,
            shakmaty_syzygy::Wdl::CursedWin => Wdl::CursedWin,
            shakmaty_syzygy::Wdl::Win => Wdl::Win,
        })
    }

    /// Probe DTZ (distance to zeroing) for a position.
    /// Returns the number of plies until a zeroing move (capture or pawn move).
    pub fn probe_dtz(&self, pos: &Position) -> Option<i32> {
        let chess_pos = to_shakmaty_position(pos)?;
        let dtz = self.inner.probe_dtz(&chess_pos).ok()?;
        Some(dtz.ignore_rounding().0)
    }

    /// Get the best move according to tablebase.
    /// This probes all legal moves and returns the one with best WDL/DTZ.
    pub fn best_move(&self, pos: &Position) -> Option<Move> {
        let piece_count = count_pieces(pos);
        if piece_count > self.max_pieces() {
            return None;
        }

        let mut moves = MoveList::new();
        pos.generate_moves(&mut moves);

        let mut best_move = None;
        let mut best_wdl = Wdl::Loss;
        let mut best_dtz = i32::MAX;

        for mv in moves.iter() {
            let mut new_pos = pos.clone();
            new_pos.make_move(mv);

            // Probe from opponent's perspective, then negate
            if let Some(wdl) = self.probe_wdl(&new_pos) {
                let our_wdl = negate_wdl(wdl);

                // Prefer better WDL, then shorter DTZ for wins
                let dominated = match (our_wdl, best_wdl) {
                    (Wdl::Win, Wdl::Win) => {
                        if let Some(dtz) = self.probe_dtz(&new_pos) {
                            let our_dtz = -dtz;
                            if our_dtz < best_dtz {
                                best_dtz = our_dtz;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    (Wdl::Loss, Wdl::Loss) => {
                        // When losing, prefer longer DTZ (delay the loss)
                        if let Some(dtz) = self.probe_dtz(&new_pos) {
                            let our_dtz = -dtz;
                            if our_dtz > best_dtz {
                                best_dtz = our_dtz;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => wdl_value(our_wdl) > wdl_value(best_wdl),
                };

                if dominated || best_move.is_none() {
                    best_move = Some(mv);
                    best_wdl = our_wdl;
                }
            }
        }

        best_move
    }

    /// Check if the position is in tablebase (has few enough pieces).
    pub fn is_in_tablebase(&self, pos: &Position) -> bool {
        count_pieces(pos) <= self.max_pieces()
    }
}

/// Count total pieces on the board.
fn count_pieces(pos: &Position) -> usize {
    pos.all_occupied().count() as usize
}

/// Negate WDL from opponent's perspective to ours.
fn negate_wdl(wdl: Wdl) -> Wdl {
    match wdl {
        Wdl::Win => Wdl::Loss,
        Wdl::CursedWin => Wdl::BlessedLoss,
        Wdl::Draw => Wdl::Draw,
        Wdl::BlessedLoss => Wdl::CursedWin,
        Wdl::Loss => Wdl::Win,
    }
}

/// Numeric value for WDL comparison.
fn wdl_value(wdl: Wdl) -> i32 {
    match wdl {
        Wdl::Win => 2,
        Wdl::CursedWin => 1,
        Wdl::Draw => 0,
        Wdl::BlessedLoss => -1,
        Wdl::Loss => -2,
    }
}

/// Convert our Position to shakmaty's Chess position via FEN.
fn to_shakmaty_position(pos: &Position) -> Option<shakmaty::Chess> {
    let fen = pos.to_fen();
    let fen_parsed: shakmaty::fen::Fen = fen.parse().ok()?;
    fen_parsed.into_position(shakmaty::CastlingMode::Standard).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wdl_score() {
        assert!(Wdl::Win.to_score() > 0);
        assert!(Wdl::Loss.to_score() < 0);
        assert_eq!(Wdl::Draw.to_score(), 0);
    }

    #[test]
    fn test_wdl_negate() {
        assert_eq!(negate_wdl(Wdl::Win), Wdl::Loss);
        assert_eq!(negate_wdl(Wdl::Loss), Wdl::Win);
        assert_eq!(negate_wdl(Wdl::Draw), Wdl::Draw);
    }

    #[test]
    fn test_to_shakmaty_position() {
        let pos = Position::startpos();
        let sm_pos = to_shakmaty_position(&pos);
        assert!(sm_pos.is_some());
    }

    #[test]
    fn test_count_pieces() {
        let pos = Position::startpos();
        assert_eq!(count_pieces(&pos), 32);
    }
}
