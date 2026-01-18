//! NNUE-specific search with incremental accumulator updates.
//!
//! This search is specialized for IncrementalNnue evaluation, calling
//! push/update_after_move/pop to maintain the accumulator efficiently.

use crate::eval::nnue::IncrementalNnue;
use crate::eval::{mated_in, DRAW_SCORE, MATE_SCORE};
use crate::movegen::MoveList;
use crate::position::Position;
use crate::tt::{score_from_tt, score_to_tt, Bound, TranspositionTable};
use crate::types::{Move, Piece};
use crate::values::piece_value;
use std::time::{Duration, Instant};

/// Thresholds for uncertainty-based depth adjustment.
const UNCERTAINTY_LOW: i16 = 30;
const UNCERTAINTY_HIGH: i16 = 150;

/// Search statistics
#[derive(Default, Clone)]
pub struct SearchInfo {
    pub nodes: u64,
    pub depth: i32,
    pub score: i16,
    pub pv: Vec<Move>,
}

/// NNUE-specific searcher with incremental accumulator updates.
pub struct NnueSearcher<'a> {
    pos: Position,
    nnue: IncrementalNnue,
    tt: &'a mut TranspositionTable,
    nodes: u64,
    max_depth: i32,
    stop: bool,
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
    check_interval: u64,
}

impl<'a> NnueSearcher<'a> {
    pub fn new(pos: Position, nnue: IncrementalNnue, tt: &'a mut TranspositionTable) -> Self {
        Self {
            pos,
            nnue,
            tt,
            nodes: 0,
            max_depth: 64,
            stop: false,
            start_time: None,
            time_limit: None,
            check_interval: 2048,
        }
    }

    pub fn set_time_limit(&mut self, limit: Duration) {
        self.time_limit = Some(limit);
    }

    #[inline]
    fn check_time(&mut self) {
        if self.nodes & (self.check_interval - 1) != 0 {
            return;
        }
        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            if start.elapsed() >= limit {
                self.stop = true;
            }
        }
    }

    /// Run iterative deepening search.
    pub fn search(&mut self, max_depth: i32) -> SearchInfo {
        self.nodes = 0;
        self.max_depth = max_depth;
        self.stop = false;
        self.start_time = Some(Instant::now());
        self.tt.new_search();

        let mut best_score = -MATE_SCORE;
        let mut best_depth = 0;
        let mut pv = Vec::new();

        for depth in 1..=max_depth {
            if self.stop {
                break;
            }

            let mut local_pv = Vec::new();
            let score = self.alpha_beta(depth, 0, -MATE_SCORE, MATE_SCORE, &mut local_pv);

            if !self.stop {
                best_score = score;
                best_depth = depth;
                if !local_pv.is_empty() {
                    pv = local_pv;
                }
            }
        }

        SearchInfo {
            nodes: self.nodes,
            depth: best_depth,
            score: best_score,
            pv,
        }
    }

    fn alpha_beta(
        &mut self,
        depth: i32,
        ply: i32,
        mut alpha: i16,
        beta: i16,
        pv: &mut Vec<Move>,
    ) -> i16 {
        pv.clear();

        if depth <= 0 {
            return self.quiescence(alpha, beta);
        }

        self.nodes += 1;
        self.check_time();

        if self.stop {
            return 0;
        }

        let is_pv = (beta as i32) - (alpha as i32) > 1;
        let hash = self.pos.hash();

        // TT probe
        let mut tt_move = Move::NULL;
        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.mv();

            if !is_pv && entry.depth() >= depth as i8 {
                let tt_score = score_from_tt(entry.score(), ply);

                match entry.bound() {
                    Bound::Exact => return tt_score,
                    Bound::Lower if tt_score >= beta => return tt_score,
                    Bound::Upper if tt_score <= alpha => return tt_score,
                    _ => {}
                }
            }
        }

        // UNCERTAINTY-BASED PRUNING using cached accumulator
        let quantiles = self.nnue.evaluate_current(self.pos.side_to_move());

        if !is_pv && quantiles.pessimistic_cutoff(beta) {
            return quantiles.q10;
        }

        if !is_pv && quantiles.optimistic_cutoff(alpha) {
            return quantiles.q90;
        }

        // UNCERTAINTY-BASED DEPTH ADJUSTMENT
        let uncertainty = quantiles.uncertainty();
        let adjusted_depth = if !is_pv && depth > 2 {
            if uncertainty < UNCERTAINTY_LOW {
                depth - 1
            } else if uncertainty > UNCERTAINTY_HIGH {
                (depth + 1).min(self.max_depth)
            } else {
                depth
            }
        } else {
            depth
        };

        // Generate moves
        let mut moves = MoveList::new();
        self.pos.generate_moves(&mut moves);

        if moves.is_empty() {
            return if self.pos.in_check() {
                mated_in(ply as i16)
            } else {
                DRAW_SCORE
            };
        }

        // Move ordering using policy head
        let policy = self.nnue.get_policy(self.pos.side_to_move());
        self.order_moves(&mut moves, tt_move, Some(&policy));

        let mut best_score = -MATE_SCORE;
        let mut best_move = Move::NULL;
        let mut child_pv = Vec::new();
        let original_alpha = alpha;

        for i in 0..moves.len() {
            let mv = moves.get(i);

            // Get move info before making it
            let moved_piece = self.pos.piece_at(mv.from()).map(|(_, p)| p).unwrap_or(Piece::Pawn);
            let captured = self.pos.piece_at(mv.to()).map(|(_, p)| p);

            // Push accumulator state
            self.nnue.push();

            // Make move
            self.pos.make_move(mv);

            // Update accumulator incrementally
            self.nnue.update_after_move(&self.pos, mv, moved_piece, captured);

            // Recurse
            let score = -self.alpha_beta(adjusted_depth - 1, ply + 1, -beta, -alpha, &mut child_pv);

            // Unmake move
            self.pos.unmake_move(mv);

            // Pop accumulator state
            self.nnue.pop();

            if self.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;

                if score > alpha {
                    alpha = score;

                    pv.clear();
                    pv.push(mv);
                    pv.extend(child_pv.iter());

                    if score >= beta {
                        break;
                    }
                }
            }
        }

        // Store in TT
        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };

        self.tt.store(
            hash,
            best_move,
            score_to_tt(best_score, ply),
            depth as i8,
            bound,
        );

        best_score
    }

    fn quiescence(&mut self, mut alpha: i16, beta: i16) -> i16 {
        self.nodes += 1;

        // Stand pat using cached accumulator
        let quantiles = self.nnue.evaluate_current(self.pos.side_to_move());
        let stand_pat = quantiles.q50; // Use median

        if stand_pat >= beta {
            return beta;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Generate captures
        let mut captures = MoveList::new();
        self.pos.generate_captures(&mut captures);

        // Order captures by MVV-LVA
        self.order_captures(&mut captures);

        for i in 0..captures.len() {
            let mv = captures.get(i);

            let moved_piece = self.pos.piece_at(mv.from()).map(|(_, p)| p).unwrap_or(Piece::Pawn);
            let captured = self.pos.piece_at(mv.to()).map(|(_, p)| p);

            self.nnue.push();
            self.pos.make_move(mv);
            self.nnue.update_after_move(&self.pos, mv, moved_piece, captured);

            let score = -self.quiescence(-beta, -alpha);

            self.pos.unmake_move(mv);
            self.nnue.pop();

            if score >= beta {
                return beta;
            }

            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    fn order_moves(&self, moves: &mut MoveList, tt_move: Move, policy: Option<&crate::eval::nnue::PolicyOutput>) {
        for i in 0..moves.len() {
            let mv = moves.get(i);
            let score = self.move_score(mv, tt_move, policy);
            moves.set_score(i, score);
        }
        moves.sort_by_score();
    }

    fn move_score(&self, mv: Move, tt_move: Move, policy: Option<&crate::eval::nnue::PolicyOutput>) -> i32 {
        let mut score = 0i32;

        if mv == tt_move && !tt_move.is_null() {
            return 100_000;
        }

        // Policy network scores
        if let Some(p) = policy {
            score += p.from[mv.from().index()] + p.to[mv.to().index()];
        }

        // MVV-LVA for captures
        if let Some((_, captured)) = self.pos.piece_at(mv.to()) {
            let victim_value = piece_value(captured);
            if let Some((_, attacker)) = self.pos.piece_at(mv.from()) {
                let attacker_value = piece_value(attacker);
                score += 15000 + victim_value as i32 * 10 - attacker_value as i32;
            }
        }

        if mv.is_promotion() {
            score += 14000;
        }

        score
    }

    fn order_captures(&self, moves: &mut MoveList) {
        self.order_moves(moves, Move::NULL, None);
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::Network;

    #[test]
    fn test_nnue_search_startpos() {
        let pos = Position::startpos();
        let network = Network::new_random();
        let nnue = IncrementalNnue::new(network, &pos);
        let mut tt = TranspositionTable::new(16);
        let mut searcher = NnueSearcher::new(pos, nnue, &mut tt);

        let info = searcher.search(3);

        assert!(!info.pv.is_empty());
        assert!(info.nodes > 0);
    }
}
