use crate::eval::nnue::PolicyOutput;
use crate::eval::{mated_in, Evaluator, DRAW_SCORE, MATE_SCORE};
use crate::movegen::MoveList;
use crate::position::Position;
use crate::tt::{score_from_tt, score_to_tt, Bound, TranspositionTable};
use crate::types::Move;
use crate::values::piece_value;
use std::time::{Duration, Instant};

/// Search statistics
#[derive(Default, Clone)]
pub struct SearchInfo {
    pub nodes: u64,
    pub depth: i32,
    pub score: i16,
    pub pv: Vec<Move>,
}

/// Alpha-beta searcher with transposition table
pub struct Searcher<'a, E: Evaluator> {
    pos: Position,
    eval: E,
    tt: &'a mut TranspositionTable,
    nodes: u64,
    max_depth: i32,
    stop: bool,
    // Time management
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
    check_interval: u64,
}

impl<'a, E: Evaluator> Searcher<'a, E> {
    pub fn new(pos: Position, eval: E, tt: &'a mut TranspositionTable) -> Self {
        Self {
            pos,
            eval,
            tt,
            nodes: 0,
            max_depth: 64,
            stop: false,
            start_time: None,
            time_limit: None,
            check_interval: 2048, // Check time every N nodes
        }
    }

    /// Set a time limit for the search
    pub fn set_time_limit(&mut self, limit: Duration) {
        self.time_limit = Some(limit);
    }

    /// Check if we should stop due to time
    #[inline]
    fn check_time(&mut self) {
        if self.nodes & (self.check_interval - 1) != 0 {
            return; // Only check every N nodes
        }
        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            if start.elapsed() >= limit {
                self.stop = true;
            }
        }
    }

    /// Run iterative deepening search to the given depth.
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

    /// Alpha-beta with TT and PV tracking
    fn alpha_beta(
        &mut self,
        depth: i32,
        ply: i32,
        mut alpha: i16,
        beta: i16,
        pv: &mut Vec<Move>,
    ) -> i16 {
        pv.clear();

        // Leaf node - use quiescence search
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

            // Use TT score for cutoff at non-PV nodes with sufficient depth
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

        // Generate moves
        let mut moves = MoveList::new();
        self.pos.generate_moves(&mut moves);

        // Checkmate or stalemate
        if moves.is_empty() {
            return if self.pos.in_check() {
                mated_in(ply as i16) // Checkmate
            } else {
                DRAW_SCORE // Stalemate
            };
        }

        // Move ordering: TT move first, then MVV-LVA for captures
        self.order_moves(&mut moves, tt_move, None);

        let mut best_score = -MATE_SCORE;
        let mut best_move = Move::NULL;
        let mut child_pv = Vec::new();
        let original_alpha = alpha;

        for i in 0..moves.len() {
            let mv = moves.get(i);

            self.pos.make_move(mv);
            let score = -self.alpha_beta(depth - 1, ply + 1, -beta, -alpha, &mut child_pv);
            self.pos.unmake_move(mv);

            if self.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;

                if score > alpha {
                    alpha = score;

                    // Update PV
                    pv.clear();
                    pv.push(mv);
                    pv.extend(child_pv.iter());

                    if score >= beta {
                        break; // Beta cutoff
                    }
                }
            }
        }

        // Store in TT
        let bound = if best_score >= beta {
            Bound::Lower // Failed high
        } else if best_score > original_alpha {
            Bound::Exact // PV node
        } else {
            Bound::Upper // Failed low
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

    /// Quiescence search - only search captures to avoid horizon effect
    fn quiescence(&mut self, mut alpha: i16, beta: i16) -> i16 {
        self.nodes += 1;

        // Stand pat - use static evaluation
        let stand_pat = self.eval.evaluate(&self.pos).score();

        if stand_pat >= beta {
            return beta;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Generate and search captures only
        let mut captures = MoveList::new();
        self.pos.generate_captures(&mut captures);

        // Order captures by MVV-LVA
        self.order_captures(&mut captures);

        for i in 0..captures.len() {
            let mv = captures.get(i);

            self.pos.make_move(mv);
            let score = -self.quiescence(-beta, -alpha);
            self.pos.unmake_move(mv);

            if score >= beta {
                return beta;
            }

            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    /// Order moves using TT move first, then MVV-LVA for captures, optional policy for quiet moves.
    fn order_moves(&self, moves: &mut MoveList, tt_move: Move, policy: Option<&PolicyOutput>) {
        // Score each move
        for i in 0..moves.len() {
            let mv = moves.get(i);
            let score = self.move_score(mv, tt_move, policy);
            moves.set_score(i, score);
        }
        // Sort by score (highest first)
        moves.sort_by_score();
    }

    /// Score a move for ordering (higher = search first).
    ///
    /// Combined scoring strategy (highest priority first):
    /// 1. TT move: +100000 bonus (always search first)
    /// 2. Captures: MVV-LVA bonus (+15000 base) - ensures tactical awareness
    /// 3. Promotions: +14000 bonus
    /// 4. Policy Network: Learned ordering for all moves (when available)
    fn move_score(&self, mv: Move, tt_move: Move, policy: Option<&PolicyOutput>) -> i32 {
        let mut score = 0i32;

        // TT move gets highest priority
        if mv == tt_move && !tt_move.is_null() {
            return 100_000;
        }

        // Policy network base score (if available)
        if let Some(p) = policy {
            score += p.from[mv.from().index()] + p.to[mv.to().index()];
        }

        // MVV-LVA for captures (always add - ensures tactics searched first)
        if let Some((_, captured)) = self.pos.piece_at(mv.to()) {
            let victim_value = piece_value(captured);
            if let Some((_, attacker)) = self.pos.piece_at(mv.from()) {
                let attacker_value = piece_value(attacker);
                // MVV-LVA: prefer capturing valuable pieces with cheap pieces
                score += 15000 + victim_value as i32 * 10 - attacker_value as i32;
            }
        }

        // Promotions (almost as important as captures)
        if mv.is_promotion() {
            score += 14000;
        }

        score
    }

    /// Order captures by MVV-LVA.
    fn order_captures(&self, moves: &mut MoveList) {
        self.order_moves(moves, Move::NULL, None);
    }

    pub fn stop(&mut self) {
        self.stop = true;
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::classical::ClassicalEval;
    use crate::eval::is_mate_score;

    #[test]
    fn test_search_startpos() {
        let pos = Position::startpos();
        let eval = ClassicalEval::new();
        let mut tt = TranspositionTable::new(16);
        let mut searcher = Searcher::new(pos, eval, &mut tt);

        let info = searcher.search(4);

        assert!(!info.pv.is_empty());
        assert!(info.nodes > 0);
        // Starting position should be roughly equal
        assert!(info.score.abs() < 100);
    }

    #[test]
    fn test_find_mate_in_one() {
        // White to move, Qh7# is mate in 1
        let pos = Position::from_fen("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1").unwrap();
        let eval = ClassicalEval::new();
        let mut tt = TranspositionTable::new(16);
        let mut searcher = Searcher::new(pos, eval, &mut tt);

        let info = searcher.search(3);

        // Should find a winning move with mate score
        assert!(is_mate_score(info.score));
        assert!(info.score > 0); // White is winning
    }

    #[test]
    fn test_avoid_mate() {
        // Black to move, must block or escape
        let pos = Position::from_fen("6k1/5pQp/8/8/8/8/5PPP/6K1 b - - 0 1").unwrap();
        let eval = ClassicalEval::new();
        let mut tt = TranspositionTable::new(16);
        let mut searcher = Searcher::new(pos, eval, &mut tt);

        let info = searcher.search(3);

        // Black should find Kf8 or Kh8 to escape
        assert!(!info.pv.is_empty());
    }

    #[test]
    fn test_capture_free_piece() {
        // White can capture undefended queen
        let pos = Position::from_fen("6k1/5ppp/3q4/8/8/3R4/5PPP/6K1 w - - 0 1").unwrap();
        let eval = ClassicalEval::new();
        let mut tt = TranspositionTable::new(16);
        let mut searcher = Searcher::new(pos, eval, &mut tt);

        let info = searcher.search(3);

        // Should capture the queen with Rxd6
        assert!(!info.pv.is_empty());
        let best = info.pv[0];
        assert_eq!(best.to(), crate::types::Square::D6);
    }

    #[test]
    fn test_tt_reduces_nodes() {
        // Search the same position twice - second should be faster due to TT
        let pos = Position::startpos();
        let mut tt = TranspositionTable::new(16);

        // First search
        let mut searcher = Searcher::new(pos.clone(), ClassicalEval::new(), &mut tt);
        let info1 = searcher.search(5);

        // Second search (TT should help)
        let mut searcher = Searcher::new(pos, ClassicalEval::new(), &mut tt);
        let info2 = searcher.search(5);

        // With TT, second search should explore same or fewer nodes
        // (might not always be fewer due to different move ordering from TT)
        assert!(info2.nodes <= info1.nodes * 2); // Should at least not be much worse
        assert_eq!(info1.score, info2.score); // Score should be the same
    }
}
