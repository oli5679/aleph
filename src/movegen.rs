use crate::bitboard::Bitboard;
use crate::magic::{bishop_attacks, king_attacks, knight_attacks, queen_attacks, rook_attacks};
use crate::position::Position;
use crate::types::{CastlingRights, Color, Move, Piece, Square};

pub struct MoveList {
    moves: [Move; 256],
    scores: [i32; 256],
    len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> Self {
        Self {
            moves: [Move::NULL; 256],
            scores: [0; 256],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = mv;
        self.scores[self.len] = 0;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().copied()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Move {
        self.moves[idx]
    }

    /// Set the score for a move at the given index.
    #[inline]
    pub fn set_score(&mut self, idx: usize, score: i32) {
        self.scores[idx] = score;
    }

    /// Get the score for a move at the given index.
    #[inline]
    pub fn get_score(&self, idx: usize) -> i32 {
        self.scores[idx]
    }

    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.moves.swap(i, j);
        self.scores.swap(i, j);
    }

    /// Sort moves by their scores in descending order (highest first).
    /// Uses selection sort which is efficient for small lists and allows
    /// lazy evaluation - we only need to find the best move each time.
    pub fn sort_by_score(&mut self) {
        for i in 0..self.len {
            let mut best_idx = i;
            let mut best_score = self.scores[i];
            for j in (i + 1)..self.len {
                if self.scores[j] > best_score {
                    best_score = self.scores[j];
                    best_idx = j;
                }
            }
            if best_idx != i {
                self.swap(i, best_idx);
            }
        }
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.len = len;
    }

    /// Push all four promotion moves for a pawn reaching the back rank.
    #[inline]
    pub fn push_promotions(&mut self, from: Square, to: Square) {
        self.push(Move::new_promotion(from, to, Piece::Queen));
        self.push(Move::new_promotion(from, to, Piece::Rook));
        self.push(Move::new_promotion(from, to, Piece::Bishop));
        self.push(Move::new_promotion(from, to, Piece::Knight));
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl Position {
    pub fn generate_moves(&self, moves: &mut MoveList) {
        let us = self.side_to_move();
        let them = us.flip();
        let our_pieces = self.occupied_by(us);
        let their_pieces = self.occupied_by(them);
        let empty = !self.all_occupied();

        self.gen_pawn_moves(moves, us, empty, their_pieces);
        self.gen_knight_moves(moves, us, our_pieces);
        self.gen_bishop_moves(moves, us, our_pieces);
        self.gen_rook_moves(moves, us, our_pieces);
        self.gen_queen_moves(moves, us, our_pieces);
        self.gen_king_moves(moves, us, our_pieces);
        self.gen_castling_moves(moves, us);

        self.filter_legal(moves);
    }

    pub fn generate_captures(&self, moves: &mut MoveList) {
        let us = self.side_to_move();
        let their_pieces = self.occupied_by(us.flip());

        self.gen_pawn_captures(moves, us, their_pieces);
        self.gen_knight_captures(moves, us, their_pieces);
        self.gen_bishop_captures(moves, us, their_pieces);
        self.gen_rook_captures(moves, us, their_pieces);
        self.gen_queen_captures(moves, us, their_pieces);
        self.gen_king_captures(moves, us, their_pieces);

        self.filter_legal(moves);
    }

    fn gen_pawn_moves(&self, moves: &mut MoveList, us: Color, empty: Bitboard, their_pieces: Bitboard) {
        let pawns = self.pieces(us, Piece::Pawn);

        let (push_dir, start_rank): (fn(Bitboard) -> Bitboard, Bitboard) = if us == Color::White {
            (Bitboard::north, Bitboard::RANK_2)
        } else {
            (Bitboard::south, Bitboard::RANK_7)
        };

        // Single pushes
        let single_push = push_dir(pawns) & empty;

        // Non-promotion pushes
        for to in single_push & !Bitboard::RANK_1 & !Bitboard::RANK_8 {
            let from = if us == Color::White {
                Square(to.0 - 8)
            } else {
                Square(to.0 + 8)
            };
            moves.push(Move::new(from, to));
        }

        // Promotion pushes
        for to in single_push & (Bitboard::RANK_1 | Bitboard::RANK_8) {
            let from = if us == Color::White {
                Square(to.0 - 8)
            } else {
                Square(to.0 + 8)
            };
            moves.push_promotions(from, to);
        }

        // Double pushes
        let double_push_mask = if us == Color::White {
            Bitboard::RANK_4
        } else {
            Bitboard::RANK_5
        };
        let double_push = push_dir(single_push & push_dir(start_rank & pawns)) & empty & double_push_mask;
        for to in double_push {
            let from = if us == Color::White {
                Square(to.0 - 16)
            } else {
                Square(to.0 + 16)
            };
            moves.push(Move::new(from, to));
        }

        // Captures
        self.gen_pawn_captures(moves, us, their_pieces);
    }

    fn gen_pawn_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        let pawns = self.pieces(us, Piece::Pawn);

        let (attack_left, attack_right): (fn(Bitboard) -> Bitboard, fn(Bitboard) -> Bitboard) = if us == Color::White {
            (Bitboard::north_west, Bitboard::north_east)
        } else {
            (Bitboard::south_west, Bitboard::south_east)
        };

        // Left captures
        let left_captures = attack_left(pawns) & their_pieces;
        for to in left_captures & !Bitboard::RANK_1 & !Bitboard::RANK_8 {
            let from = if us == Color::White {
                Square(to.0 - 7)
            } else {
                Square(to.0 + 9)
            };
            moves.push(Move::new(from, to));
        }
        for to in left_captures & (Bitboard::RANK_1 | Bitboard::RANK_8) {
            let from = if us == Color::White {
                Square(to.0 - 7)
            } else {
                Square(to.0 + 9)
            };
            moves.push_promotions(from, to);
        }

        // Right captures
        let right_captures = attack_right(pawns) & their_pieces;
        for to in right_captures & !Bitboard::RANK_1 & !Bitboard::RANK_8 {
            let from = if us == Color::White {
                Square(to.0 - 9)
            } else {
                Square(to.0 + 7)
            };
            moves.push(Move::new(from, to));
        }
        for to in right_captures & (Bitboard::RANK_1 | Bitboard::RANK_8) {
            let from = if us == Color::White {
                Square(to.0 - 9)
            } else {
                Square(to.0 + 7)
            };
            moves.push_promotions(from, to);
        }

        // En passant
        if let Some(ep_sq) = self.ep_square() {
            let ep_bb = Bitboard::from_sq(ep_sq);

            // Pawns that can capture to ep square
            let ep_attackers = if us == Color::White {
                (ep_bb.south_west() | ep_bb.south_east()) & pawns
            } else {
                (ep_bb.north_west() | ep_bb.north_east()) & pawns
            };

            for from in ep_attackers {
                moves.push(Move::new_ep(from, ep_sq));
            }
        }
    }

    fn gen_knight_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        for from in self.pieces(us, Piece::Knight) {
            let attacks = knight_attacks(from) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_knight_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        for from in self.pieces(us, Piece::Knight) {
            let attacks = knight_attacks(from) & their_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_bishop_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Bishop) {
            let attacks = bishop_attacks(from, occupied) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_bishop_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Bishop) {
            let attacks = bishop_attacks(from, occupied) & their_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_rook_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Rook) {
            let attacks = rook_attacks(from, occupied) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_rook_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Rook) {
            let attacks = rook_attacks(from, occupied) & their_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_queen_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Queen) {
            let attacks = queen_attacks(from, occupied) & !our_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_queen_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        let occupied = self.all_occupied();
        for from in self.pieces(us, Piece::Queen) {
            let attacks = queen_attacks(from, occupied) & their_pieces;
            for to in attacks {
                moves.push(Move::new(from, to));
            }
        }
    }

    fn gen_king_moves(&self, moves: &mut MoveList, us: Color, our_pieces: Bitboard) {
        let king_sq = self.king_sq(us);
        let attacks = king_attacks(king_sq) & !our_pieces;
        for to in attacks {
            moves.push(Move::new(king_sq, to));
        }
    }

    fn gen_king_captures(&self, moves: &mut MoveList, us: Color, their_pieces: Bitboard) {
        let king_sq = self.king_sq(us);
        let attacks = king_attacks(king_sq) & their_pieces;
        for to in attacks {
            moves.push(Move::new(king_sq, to));
        }
    }

    fn gen_castling_moves(&self, moves: &mut MoveList, us: Color) {
        let them = us.flip();
        let castling = self.castling_rights();
        let occupied = self.all_occupied();

        if us == Color::White {
            // White kingside
            if castling.has(CastlingRights::WK) {
                let path = Bitboard::from_sq(Square::F1) | Bitboard::from_sq(Square::G1);
                if (occupied & path).is_empty()
                    && !self.is_attacked(Square::E1, them)
                    && !self.is_attacked(Square::F1, them)
                    && !self.is_attacked(Square::G1, them)
                {
                    moves.push(Move::new_castle(Square::E1, Square::G1));
                }
            }
            // White queenside
            if castling.has(CastlingRights::WQ) {
                let path = Bitboard::from_sq(Square::D1) | Bitboard::from_sq(Square::C1) | Bitboard::from_sq(Square::B1);
                if (occupied & path).is_empty()
                    && !self.is_attacked(Square::E1, them)
                    && !self.is_attacked(Square::D1, them)
                    && !self.is_attacked(Square::C1, them)
                {
                    moves.push(Move::new_castle(Square::E1, Square::C1));
                }
            }
        } else {
            // Black kingside
            if castling.has(CastlingRights::BK) {
                let path = Bitboard::from_sq(Square::F8) | Bitboard::from_sq(Square::G8);
                if (occupied & path).is_empty()
                    && !self.is_attacked(Square::E8, them)
                    && !self.is_attacked(Square::F8, them)
                    && !self.is_attacked(Square::G8, them)
                {
                    moves.push(Move::new_castle(Square::E8, Square::G8));
                }
            }
            // Black queenside
            if castling.has(CastlingRights::BQ) {
                let path = Bitboard::from_sq(Square::D8) | Bitboard::from_sq(Square::C8) | Bitboard::from_sq(Square::B8);
                if (occupied & path).is_empty()
                    && !self.is_attacked(Square::E8, them)
                    && !self.is_attacked(Square::D8, them)
                    && !self.is_attacked(Square::C8, them)
                {
                    moves.push(Move::new_castle(Square::E8, Square::C8));
                }
            }
        }
    }

    fn filter_legal(&self, moves: &mut MoveList) {
        let mut i = 0;
        while i < moves.len() {
            let mv = moves.get(i);
            let mut copy = self.clone();
            copy.make_move(mv);

            // Check if our king is in check after the move
            // Note: side has switched, so we check if the opponent's king (our original king) is attacked
            let king_sq = copy.king_sq(self.side_to_move());
            if copy.is_attacked(king_sq, self.side_to_move().flip()) {
                // Illegal move - swap with last and reduce length
                let last = moves.len() - 1;
                moves.swap(i, last);
                moves.truncate(last);
            } else {
                i += 1;
            }
        }
    }
}

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut moves = MoveList::new();
    pos.generate_moves(&mut moves);

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for mv in moves.iter() {
        pos.make_move(mv);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(mv);
    }

    nodes
}

pub fn perft_divide(pos: &mut Position, depth: u32) -> u64 {
    let mut moves = MoveList::new();
    pos.generate_moves(&mut moves);

    let mut total = 0u64;
    for mv in moves.iter() {
        pos.make_move(mv);
        let nodes = if depth > 1 {
            perft(pos, depth - 1)
        } else {
            1
        };
        pos.unmake_move(mv);
        println!("{}: {}", mv, nodes);
        total += nodes;
    }

    println!("\nTotal: {}", total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_moves() {
        let pos = Position::startpos();
        let mut moves = MoveList::new();
        pos.generate_moves(&mut moves);
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn test_perft_startpos() {
        let mut pos = Position::startpos();
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8902);
        assert_eq!(perft(&mut pos, 4), 197281);
    }

    #[test]
    fn test_perft_kiwipete() {
        // Position 2: Kiwipete - tests many edge cases
        let mut pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 48);
        assert_eq!(perft(&mut pos, 2), 2039);
        assert_eq!(perft(&mut pos, 3), 97862);
    }

    #[test]
    fn test_perft_position3() {
        // Position 3: tests en passant
        let mut pos = Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 14);
        assert_eq!(perft(&mut pos, 2), 191);
        assert_eq!(perft(&mut pos, 3), 2812);
        assert_eq!(perft(&mut pos, 4), 43238);
    }

    #[test]
    fn test_perft_position4() {
        // Position 4: tests promotions
        let mut pos = Position::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 6);
        assert_eq!(perft(&mut pos, 2), 264);
        assert_eq!(perft(&mut pos, 3), 9467);
    }

    #[test]
    fn test_perft_position5() {
        // Position 5
        let mut pos = Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap();
        assert_eq!(perft(&mut pos, 1), 44);
        assert_eq!(perft(&mut pos, 2), 1486);
        assert_eq!(perft(&mut pos, 3), 62379);
    }

    #[test]
    fn test_perft_position6() {
        // Position 6
        let mut pos = Position::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
        assert_eq!(perft(&mut pos, 1), 46);
        assert_eq!(perft(&mut pos, 2), 2079);
        assert_eq!(perft(&mut pos, 3), 89890);
    }
}
