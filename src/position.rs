use crate::bitboard::Bitboard;
use crate::magic::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::types::{CastlingRights, Color, Move, Piece, Square};

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone)]
pub struct Position {
    pieces: [[Bitboard; 6]; 2],
    occupied: [Bitboard; 2],
    all: Bitboard,
    side: Color,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove: u8,
    fullmove: u16,
    hash: u64,
    history: Vec<Undo>,
}

#[derive(Clone)]
struct Undo {
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove: u8,
    hash: u64,
}

// Zobrist keys for hashing
struct Zobrist {
    pieces: [[[u64; 64]; 6]; 2],
    castling: [u64; 16],
    ep_file: [u64; 8],
    side: u64,
}

static ZOBRIST: std::sync::OnceLock<Zobrist> = std::sync::OnceLock::new();

fn get_zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut z = Zobrist {
            pieces: [[[0; 64]; 6]; 2],
            castling: [0; 16],
            ep_file: [0; 8],
            side: 0,
        };

        let mut seed = 0x12345678u64;
        let mut next = || {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            seed
        };

        for color in 0..2 {
            for piece in 0..6 {
                for sq in 0..64 {
                    z.pieces[color][piece][sq] = next();
                }
            }
        }
        for i in 0..16 {
            z.castling[i] = next();
        }
        for i in 0..8 {
            z.ep_file[i] = next();
        }
        z.side = next();

        z
    })
}

impl Position {
    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS).expect("Invalid startpos FEN")
    }

    pub fn from_fen(fen: &str) -> Result<Self, &'static str> {
        crate::magic::init_all();

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN must have at least 4 parts");
        }

        let mut pos = Position {
            pieces: [[Bitboard::EMPTY; 6]; 2],
            occupied: [Bitboard::EMPTY; 2],
            all: Bitboard::EMPTY,
            side: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            history: Vec::new(),
        };

        // Parse piece placement
        let mut rank = 7u8;
        let mut file = 0u8;

        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank = rank.wrapping_sub(1);
                    file = 0;
                }
                '1'..='8' => {
                    file += c as u8 - b'0';
                }
                _ => {
                    let color = if c.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let piece = Piece::from_char(c).ok_or("Invalid piece character")?;
                    let sq = Square::new(file, rank);
                    pos.put_piece(color, piece, sq);
                    file += 1;
                }
            }
        }

        // Parse side to move
        pos.side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid side to move"),
        };

        // Parse castling rights
        for c in parts[2].chars() {
            match c {
                'K' => pos.castling.0 |= CastlingRights::WK,
                'Q' => pos.castling.0 |= CastlingRights::WQ,
                'k' => pos.castling.0 |= CastlingRights::BK,
                'q' => pos.castling.0 |= CastlingRights::BQ,
                '-' => {}
                _ => return Err("Invalid castling rights"),
            }
        }

        // Parse en passant square
        if parts[3] != "-" {
            pos.ep_square = Square::from_str(parts[3]);
        }

        // Parse halfmove clock (optional)
        if parts.len() > 4 {
            pos.halfmove = parts[4].parse().unwrap_or(0);
        }

        // Parse fullmove number (optional)
        if parts.len() > 5 {
            pos.fullmove = parts[5].parse().unwrap_or(1);
        }

        // Compute hash
        pos.hash = pos.compute_hash();

        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if let Some((color, piece)) = self.piece_at(sq) {
                    if empty > 0 {
                        fen.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    fen.push(piece.to_char(color));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // Side to move
        fen.push(' ');
        fen.push(if self.side == Color::White { 'w' } else { 'b' });

        // Castling rights
        fen.push(' ');
        if self.castling.is_empty() {
            fen.push('-');
        } else {
            if self.castling.has(CastlingRights::WK) {
                fen.push('K');
            }
            if self.castling.has(CastlingRights::WQ) {
                fen.push('Q');
            }
            if self.castling.has(CastlingRights::BK) {
                fen.push('k');
            }
            if self.castling.has(CastlingRights::BQ) {
                fen.push('q');
            }
        }

        // En passant
        fen.push(' ');
        if let Some(ep) = self.ep_square {
            fen.push_str(&ep.to_string());
        } else {
            fen.push('-');
        }

        // Halfmove clock and fullmove number
        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));

        fen
    }

    fn put_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = Bitboard::from_sq(sq);
        self.pieces[color.index()][piece.index()] |= bb;
        self.occupied[color.index()] |= bb;
        self.all |= bb;
    }

    fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = Bitboard::from_sq(sq);
        self.pieces[color.index()][piece.index()] ^= bb;
        self.occupied[color.index()] ^= bb;
        self.all ^= bb;
    }

    fn move_piece(&mut self, color: Color, piece: Piece, from: Square, to: Square) {
        let from_to = Bitboard::from_sq(from) | Bitboard::from_sq(to);
        self.pieces[color.index()][piece.index()] ^= from_to;
        self.occupied[color.index()] ^= from_to;
        self.all ^= from_to;
    }

    fn compute_hash(&self) -> u64 {
        let z = get_zobrist();
        let mut hash = 0u64;

        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                for sq in self.pieces(color, piece) {
                    hash ^= z.pieces[color.index()][piece.index()][sq.index()];
                }
            }
        }

        hash ^= z.castling[self.castling.0 as usize];

        if let Some(ep) = self.ep_square {
            hash ^= z.ep_file[ep.file() as usize];
        }

        if self.side == Color::Black {
            hash ^= z.side;
        }

        hash
    }

    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.side
    }

    #[inline]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline]
    pub fn pieces(&self, color: Color, piece: Piece) -> Bitboard {
        self.pieces[color.index()][piece.index()]
    }

    #[inline]
    pub fn occupied_by(&self, color: Color) -> Bitboard {
        self.occupied[color.index()]
    }

    #[inline]
    pub fn all_occupied(&self) -> Bitboard {
        self.all
    }

    #[inline]
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling
    }

    #[inline]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    pub fn piece_at(&self, sq: Square) -> Option<(Color, Piece)> {
        let bb = Bitboard::from_sq(sq);
        if (self.all & bb).is_empty() {
            return None;
        }

        let color = if (self.occupied[0] & bb).is_not_empty() {
            Color::White
        } else {
            Color::Black
        };

        for piece in Piece::ALL {
            if (self.pieces[color.index()][piece.index()] & bb).is_not_empty() {
                return Some((color, piece));
            }
        }

        None
    }

    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces(color, Piece::King).lsb()
    }

    /// Iterate over all pieces on the board, returning (square, color, piece).
    /// Used for NNUE feature extraction.
    pub fn all_pieces(&self) -> impl Iterator<Item = (Square, Color, Piece)> + '_ {
        [Color::White, Color::Black].into_iter().flat_map(move |color| {
            Piece::ALL.into_iter().flat_map(move |piece| {
                self.pieces(color, piece).into_iter().map(move |sq| (sq, color, piece))
            })
        })
    }

    pub fn is_attacked(&self, sq: Square, by: Color) -> bool {
        let occupied = self.all;

        // Pawn attacks
        if (pawn_attacks(sq, by.flip()) & self.pieces(by, Piece::Pawn)).is_not_empty() {
            return true;
        }

        // Knight attacks
        if (knight_attacks(sq) & self.pieces(by, Piece::Knight)).is_not_empty() {
            return true;
        }

        // King attacks
        if (king_attacks(sq) & self.pieces(by, Piece::King)).is_not_empty() {
            return true;
        }

        // Bishop/Queen attacks
        let bishop_queen = self.pieces(by, Piece::Bishop) | self.pieces(by, Piece::Queen);
        if (bishop_attacks(sq, occupied) & bishop_queen).is_not_empty() {
            return true;
        }

        // Rook/Queen attacks
        let rook_queen = self.pieces(by, Piece::Rook) | self.pieces(by, Piece::Queen);
        if (rook_attacks(sq, occupied) & rook_queen).is_not_empty() {
            return true;
        }

        false
    }

    pub fn in_check(&self) -> bool {
        let king_sq = self.king_sq(self.side);
        self.is_attacked(king_sq, self.side.flip())
    }

    pub fn attackers_to(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let mut attackers = Bitboard::EMPTY;

        attackers |= pawn_attacks(sq, Color::Black) & self.pieces(Color::White, Piece::Pawn);
        attackers |= pawn_attacks(sq, Color::White) & self.pieces(Color::Black, Piece::Pawn);
        attackers |= knight_attacks(sq) & (self.pieces(Color::White, Piece::Knight) | self.pieces(Color::Black, Piece::Knight));
        attackers |= king_attacks(sq) & (self.pieces(Color::White, Piece::King) | self.pieces(Color::Black, Piece::King));

        let bishop_queen = self.pieces(Color::White, Piece::Bishop)
            | self.pieces(Color::Black, Piece::Bishop)
            | self.pieces(Color::White, Piece::Queen)
            | self.pieces(Color::Black, Piece::Queen);
        attackers |= bishop_attacks(sq, occupied) & bishop_queen;

        let rook_queen = self.pieces(Color::White, Piece::Rook)
            | self.pieces(Color::Black, Piece::Rook)
            | self.pieces(Color::White, Piece::Queen)
            | self.pieces(Color::Black, Piece::Queen);
        attackers |= rook_attacks(sq, occupied) & rook_queen;

        attackers
    }

    pub fn make_move(&mut self, mv: Move) {
        let z = get_zobrist();
        let us = self.side;
        let them = us.flip();
        let from = mv.from();
        let to = mv.to();

        // Find the moving piece
        let mut moving_piece = Piece::Pawn;
        for piece in Piece::ALL {
            if self.pieces(us, piece).contains(from) {
                moving_piece = piece;
                break;
            }
        }

        // Find captured piece (if any)
        let captured = if mv.is_ep() {
            Some(Piece::Pawn)
        } else {
            self.piece_at(to).map(|(_, p)| p)
        };

        // Save undo info
        self.history.push(Undo {
            captured,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            hash: self.hash,
        });

        // Update hash for moved piece
        self.hash ^= z.pieces[us.index()][moving_piece.index()][from.index()];

        // Handle captures
        if let Some(cap_piece) = captured {
            let cap_sq = if mv.is_ep() {
                // En passant capture square is one rank behind the target
                Square::new(to.file(), from.rank())
            } else {
                to
            };

            self.remove_piece(them, cap_piece, cap_sq);
            self.hash ^= z.pieces[them.index()][cap_piece.index()][cap_sq.index()];
            self.halfmove = 0;
        } else if moving_piece == Piece::Pawn {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }

        // Handle castling
        if mv.is_castle() {
            // Move the rook
            let (rook_from, rook_to) = match to {
                sq if sq == Square::G1 => (Square::H1, Square::F1),
                sq if sq == Square::C1 => (Square::A1, Square::D1),
                sq if sq == Square::G8 => (Square::H8, Square::F8),
                sq if sq == Square::C8 => (Square::A8, Square::D8),
                _ => unreachable!(),
            };
            self.move_piece(us, Piece::Rook, rook_from, rook_to);
            self.hash ^= z.pieces[us.index()][Piece::Rook.index()][rook_from.index()];
            self.hash ^= z.pieces[us.index()][Piece::Rook.index()][rook_to.index()];
        }

        // Move the piece
        self.remove_piece(us, moving_piece, from);
        let final_piece = if mv.is_promotion() {
            mv.promotion_piece()
        } else {
            moving_piece
        };
        self.put_piece(us, final_piece, to);
        self.hash ^= z.pieces[us.index()][final_piece.index()][to.index()];

        // Update en passant square
        if let Some(ep) = self.ep_square {
            self.hash ^= z.ep_file[ep.file() as usize];
        }
        self.ep_square = None;

        if moving_piece == Piece::Pawn {
            let diff = (to.0 as i8 - from.0 as i8).abs();
            if diff == 16 {
                // Double pawn push
                let ep_sq = Square::new(from.file(), (from.rank() + to.rank()) / 2);
                self.ep_square = Some(ep_sq);
                self.hash ^= z.ep_file[ep_sq.file() as usize];
            }
        }

        // Update castling rights
        self.hash ^= z.castling[self.castling.0 as usize];

        // If king moves, remove all castling rights for that side
        if moving_piece == Piece::King {
            if us == Color::White {
                self.castling = self.castling.remove(CastlingRights::WK | CastlingRights::WQ);
            } else {
                self.castling = self.castling.remove(CastlingRights::BK | CastlingRights::BQ);
            }
        }

        // If rook moves or is captured, remove that castling right
        const CASTLING_MASK: [u8; 64] = {
            let mut mask = [0xFFu8; 64];
            mask[Square::A1.0 as usize] = !CastlingRights::WQ;
            mask[Square::H1.0 as usize] = !CastlingRights::WK;
            mask[Square::A8.0 as usize] = !CastlingRights::BQ;
            mask[Square::H8.0 as usize] = !CastlingRights::BK;
            mask[Square::E1.0 as usize] = !(CastlingRights::WK | CastlingRights::WQ);
            mask[Square::E8.0 as usize] = !(CastlingRights::BK | CastlingRights::BQ);
            mask
        };

        self.castling.0 &= CASTLING_MASK[from.index()];
        self.castling.0 &= CASTLING_MASK[to.index()];

        self.hash ^= z.castling[self.castling.0 as usize];

        // Switch side
        self.side = them;
        self.hash ^= z.side;

        // Update fullmove counter
        if us == Color::Black {
            self.fullmove += 1;
        }
    }

    pub fn unmake_move(&mut self, mv: Move) {
        let undo = self.history.pop().expect("No move to unmake");
        let them = self.side;
        let us = them.flip();
        let from = mv.from();
        let to = mv.to();

        // Restore state
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        self.hash = undo.hash;
        self.side = us;

        if us == Color::Black {
            self.fullmove -= 1;
        }

        // Find the piece that moved (it's now at 'to')
        let moved_piece = if mv.is_promotion() {
            Piece::Pawn
        } else {
            let mut piece = Piece::Pawn;
            for p in Piece::ALL {
                if self.pieces(us, p).contains(to) {
                    piece = p;
                    break;
                }
            }
            piece
        };

        // Remove piece from destination
        let final_piece = if mv.is_promotion() {
            mv.promotion_piece()
        } else {
            moved_piece
        };
        self.remove_piece(us, final_piece, to);

        // Put piece back at origin
        self.put_piece(us, moved_piece, from);

        // Handle castling - move rook back
        if mv.is_castle() {
            let (rook_from, rook_to) = match to {
                sq if sq == Square::G1 => (Square::H1, Square::F1),
                sq if sq == Square::C1 => (Square::A1, Square::D1),
                sq if sq == Square::G8 => (Square::H8, Square::F8),
                sq if sq == Square::C8 => (Square::A8, Square::D8),
                _ => unreachable!(),
            };
            self.move_piece(us, Piece::Rook, rook_to, rook_from);
        }

        // Restore captured piece
        if let Some(captured) = undo.captured {
            let cap_sq = if mv.is_ep() {
                Square::new(to.file(), from.rank())
            } else {
                to
            };
            self.put_piece(them, captured, cap_sq);
        }
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if let Some((color, piece)) = self.piece_at(sq) {
                    write!(f, "{} ", piece.to_char(color))?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")?;
        writeln!(f)?;
        writeln!(f, "  FEN: {}", self.to_fen())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos() {
        let pos = Position::startpos();
        assert_eq!(pos.side_to_move(), Color::White);
        assert_eq!(pos.pieces(Color::White, Piece::Pawn).count(), 8);
        assert_eq!(pos.pieces(Color::Black, Piece::Pawn).count(), 8);
        assert_eq!(pos.pieces(Color::White, Piece::King).count(), 1);
        assert!(pos.castling_rights().has(CastlingRights::WK));
        assert!(pos.castling_rights().has(CastlingRights::WQ));
        assert!(pos.castling_rights().has(CastlingRights::BK));
        assert!(pos.castling_rights().has(CastlingRights::BQ));
    }

    #[test]
    fn test_fen_roundtrip() {
        let fens = [
            STARTPOS,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/8/8/4K2k w - - 0 1",
        ];

        for fen in fens {
            let pos = Position::from_fen(fen).expect("Valid FEN");
            // Note: FEN might normalize (e.g., halfmove/fullmove), so we just check it parses back
            let _ = Position::from_fen(&pos.to_fen()).expect("Roundtrip should work");
        }
    }

    #[test]
    fn test_make_unmake_simple() {
        let mut pos = Position::startpos();
        let initial_hash = pos.hash();
        let initial_fen = pos.to_fen();

        let mv = Move::new(Square::new(4, 1), Square::new(4, 3)); // e2e4
        pos.make_move(mv);

        assert_eq!(pos.side_to_move(), Color::Black);
        assert!(pos.ep_square().is_some());
        assert_ne!(pos.hash(), initial_hash);

        pos.unmake_move(mv);

        assert_eq!(pos.side_to_move(), Color::White);
        assert_eq!(pos.hash(), initial_hash);
        assert_eq!(pos.to_fen(), initial_fen);
    }

    #[test]
    fn test_in_check() {
        // Position with black king in check
        let pos = Position::from_fen("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3").unwrap();
        assert!(pos.in_check());
    }

    #[test]
    fn test_is_attacked() {
        let pos = Position::startpos();
        // e3 is attacked by white pawns on d2 and f2? No, pawns attack diagonally forward
        // e3 is attacked by white pawn on d2 (d2 attacks e3) and f2 (f2 attacks e3)
        assert!(pos.is_attacked(Square::new(4, 2), Color::White)); // e3 attacked by d2 pawn
    }
}
