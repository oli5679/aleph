use crate::book::OpeningBook;
use crate::eval::classical::ClassicalEval;
use crate::eval::is_mate_score;
use crate::eval::nnue::{loader, IncrementalNnue, Network};
use crate::position::Position;
use crate::search::Searcher;
use crate::search_nnue::NnueSearcher;
use crate::tablebase::Tablebase;
use crate::tt::TranspositionTable;
use crate::types::{Color, Move};
use std::io::{self, BufRead, Write};
use std::time::Duration;

const ENGINE_NAME: &str = "Aleph";
const ENGINE_AUTHOR: &str = "Aleph Authors";
const DEFAULT_HASH_MB: usize = 64;

/// Evaluator type - either classical or NNUE
enum EvalType {
    Classical,
    Nnue(Network),
}

/// Engine options that persist across moves.
struct EngineState {
    book: Option<OpeningBook>,
    tablebase: Option<Tablebase>,
    use_book: bool,
    nnue_path: Option<String>,
}

/// Run the UCI protocol loop
pub fn uci_loop() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut pos = Position::startpos();
    let mut tt = TranspositionTable::new(DEFAULT_HASH_MB);
    let mut eval_type = EvalType::Classical;
    let mut state = EngineState {
        book: None,
        tablebase: None,
        use_book: true,
        nnue_path: None,
    };

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name {}", ENGINE_NAME);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default {} min 1 max 4096", DEFAULT_HASH_MB);
                println!("option name EvalFile type string default <empty>");
                println!("option name OwnBook type check default true");
                println!("option name BookFile type string default <empty>");
                println!("option name SyzygyPath type string default <empty>");
                println!("uciok");
                stdout.flush().ok();
            }

            "setoption" => {
                // Parse: setoption name <Name> value <Value>
                if tokens.len() >= 5 && tokens[1] == "name" && tokens[3] == "value" {
                    let name = tokens[2].to_lowercase();
                    let value = tokens[4..].join(" ");

                    match name.as_str() {
                        "hash" => {
                            if let Ok(size_mb) = value.parse::<usize>() {
                                let size_mb = size_mb.max(1).min(4096);
                                tt.resize(size_mb);
                                eprintln!("info string Hash table resized to {} MB", size_mb);
                            }
                        }
                        "evalfile" => {
                            if value.is_empty() || value == "<empty>" {
                                eval_type = EvalType::Classical;
                                state.nnue_path = None;
                                eprintln!("info string Using classical evaluation");
                            } else {
                                match loader::load_network(&value) {
                                    Ok(network) => {
                                        eval_type = EvalType::Nnue(network);
                                        state.nnue_path = Some(value.clone());
                                        eprintln!("info string Loaded NNUE from {}", value);
                                    }
                                    Err(e) => {
                                        eprintln!("info string Failed to load NNUE: {}", e);
                                    }
                                }
                            }
                        }
                        "ownbook" => {
                            state.use_book = value.to_lowercase() == "true";
                            eprintln!("info string OwnBook set to {}", state.use_book);
                        }
                        "bookfile" => {
                            if value.is_empty() || value == "<empty>" {
                                state.book = None;
                                eprintln!("info string Opening book disabled");
                            } else {
                                match OpeningBook::load(&value) {
                                    Ok(book) => {
                                        let entries = book.len();
                                        state.book = Some(book);
                                        eprintln!("info string Loaded opening book with {} entries", entries);
                                    }
                                    Err(e) => {
                                        eprintln!("info string Failed to load opening book: {}", e);
                                    }
                                }
                            }
                        }
                        "syzygypath" => {
                            if value.is_empty() || value == "<empty>" {
                                state.tablebase = None;
                                eprintln!("info string Syzygy tablebases disabled");
                            } else {
                                match Tablebase::new(&value) {
                                    Ok(tb) => {
                                        let max_pieces = tb.max_pieces();
                                        state.tablebase = Some(tb);
                                        eprintln!("info string Loaded Syzygy tablebases ({}-piece)", max_pieces);
                                    }
                                    Err(e) => {
                                        eprintln!("info string Failed to load Syzygy tablebases: {}", e);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            "isready" => {
                println!("readyok");
                stdout.flush().ok();
            }

            "ucinewgame" => {
                pos = Position::startpos();
                tt.clear();
            }

            "position" => {
                pos = parse_position(&tokens[1..]);
            }

            "go" => {
                let go_params = parse_go_params(&tokens[1..]);

                // 1. Check opening book first
                if state.use_book {
                    if let Some(ref book) = state.book {
                        if let Some(book_move) = book.probe(&pos) {
                            eprintln!("info string Book move");
                            println!("bestmove {}", book_move);
                            stdout.flush().ok();
                            continue;
                        }
                    }
                }

                // 2. Check tablebase for few-piece positions
                if let Some(ref tb) = state.tablebase {
                    if tb.is_in_tablebase(&pos) {
                        if let Some(tb_move) = tb.best_move(&pos) {
                            if let Some(wdl) = tb.probe_wdl(&pos) {
                                eprintln!("info string Tablebase {:?}", wdl);
                            }
                            println!("bestmove {}", tb_move);
                            stdout.flush().ok();
                            continue;
                        }
                    }
                }

                // 3. Run normal search
                let time_limit = calculate_time_limit(&go_params, pos.side_to_move());
                let info = run_search(&pos, &mut tt, &eval_type, &state, time_limit, go_params.depth);

                // Print search info
                print_info(&info, &tt);

                // Print best move
                if !info.pv.is_empty() {
                    println!("bestmove {}", info.pv[0]);
                } else {
                    println!("bestmove 0000"); // Null move fallback
                }
                stdout.flush().ok();
            }

            "stop" => {
                // TODO: Implement proper stop mechanism with threads
            }

            "quit" => {
                break;
            }

            "d" | "display" => {
                println!("{}", pos);
                stdout.flush().ok();
            }

            "perft" => {
                let depth: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                let start = std::time::Instant::now();
                let nodes = crate::movegen::perft(&mut pos, depth);
                let elapsed = start.elapsed();
                println!(
                    "Nodes: {} Time: {:.3}s NPS: {:.0}",
                    nodes,
                    elapsed.as_secs_f64(),
                    nodes as f64 / elapsed.as_secs_f64()
                );
                stdout.flush().ok();
            }

            _ => {}
        }
    }
}

/// Run search with the appropriate evaluator.
fn run_search(
    pos: &Position,
    tt: &mut TranspositionTable,
    eval_type: &EvalType,
    _state: &EngineState,
    time_limit: Option<Duration>,
    depth: i32,
) -> crate::search::SearchInfo {
    match eval_type {
        EvalType::Classical => {
            let eval = ClassicalEval::new();
            let mut searcher = Searcher::new(pos.clone(), eval, tt);
            if let Some(limit) = time_limit {
                searcher.set_time_limit(limit);
            }
            searcher.search(depth)
        }
        EvalType::Nnue(network) => {
            // Create incremental NNUE evaluator with proper accumulator management
            let nnue = IncrementalNnue::new(network.clone(), pos);
            let mut searcher = NnueSearcher::new(pos.clone(), nnue, tt);
            if let Some(limit) = time_limit {
                searcher.set_time_limit(limit);
            }
            let info = searcher.search(depth);
            // Convert NnueSearcher::SearchInfo to search::SearchInfo
            crate::search::SearchInfo {
                nodes: info.nodes,
                depth: info.depth,
                score: info.score,
                pv: info.pv,
            }
        }
    }
}

fn parse_position(tokens: &[&str]) -> Position {
    if tokens.is_empty() {
        return Position::startpos();
    }

    let mut pos = if tokens[0] == "startpos" {
        Position::startpos()
    } else if tokens[0] == "fen" {
        // Collect FEN parts (up to "moves" or end)
        let fen_parts: Vec<&str> = tokens[1..]
            .iter()
            .take_while(|&&t| t != "moves")
            .copied()
            .collect();
        let fen = fen_parts.join(" ");
        Position::from_fen(&fen).unwrap_or_else(|_| Position::startpos())
    } else {
        Position::startpos()
    };

    // Find and apply moves
    let moves_idx = tokens.iter().position(|&t| t == "moves");
    if let Some(idx) = moves_idx {
        for move_str in &tokens[idx + 1..] {
            if let Some(mv) = parse_move(&pos, move_str) {
                pos.make_move(mv);
            }
        }
    }

    pos
}

fn parse_move(pos: &Position, s: &str) -> Option<Move> {
    use crate::movegen::MoveList;
    use crate::types::{Piece, Square};

    if s.len() < 4 {
        return None;
    }

    let from = Square::from_str(&s[0..2])?;
    let to = Square::from_str(&s[2..4])?;
    let promo = s.chars().nth(4);

    // Generate legal moves and find matching one
    let mut moves = MoveList::new();
    pos.generate_moves(&mut moves);

    for mv in moves.iter() {
        if mv.from() == from && mv.to() == to {
            // Check promotion match if applicable
            if let Some(p) = promo {
                if mv.is_promotion() {
                    let promo_char = match mv.promotion_piece() {
                        Piece::Queen => 'q',
                        Piece::Rook => 'r',
                        Piece::Bishop => 'b',
                        Piece::Knight => 'n',
                        Piece::Pawn | Piece::King => unreachable!(),
                    };
                    if promo_char == p {
                        return Some(mv);
                    }
                }
            } else if !mv.is_promotion() {
                return Some(mv);
            } else {
                // Default to queen promotion
                if matches!(mv.promotion_piece(), Piece::Queen) {
                    return Some(mv);
                }
            }
        }
    }

    // Try to construct basic move as fallback
    Move::from_uci(s)
}

/// Parameters parsed from the "go" command
#[derive(Default)]
struct GoParams {
    depth: i32,
    wtime: Option<u64>,  // White time in ms
    btime: Option<u64>,  // Black time in ms
    winc: Option<u64>,   // White increment in ms
    binc: Option<u64>,   // Black increment in ms
    movestogo: Option<u32>,
    movetime: Option<u64>, // Fixed time per move in ms
    infinite: bool,
}

fn parse_go_params(tokens: &[&str]) -> GoParams {
    let mut params = GoParams {
        depth: 64, // Default to max depth (time will limit search)
        ..Default::default()
    };

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if i + 1 < tokens.len() {
                    params.depth = tokens[i + 1].parse().unwrap_or(64);
                }
                i += 2;
            }
            "wtime" => {
                if i + 1 < tokens.len() {
                    params.wtime = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "btime" => {
                if i + 1 < tokens.len() {
                    params.btime = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "winc" => {
                if i + 1 < tokens.len() {
                    params.winc = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "binc" => {
                if i + 1 < tokens.len() {
                    params.binc = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "movestogo" => {
                if i + 1 < tokens.len() {
                    params.movestogo = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "movetime" => {
                if i + 1 < tokens.len() {
                    params.movetime = tokens[i + 1].parse().ok();
                }
                i += 2;
            }
            "infinite" => {
                params.infinite = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    // If no time controls specified and not infinite, use a reasonable default depth
    if params.wtime.is_none() && params.movetime.is_none() && !params.infinite && params.depth == 64 {
        params.depth = 6;
    }

    params.depth = params.depth.min(64);
    params
}

/// Calculate how much time to spend on this move
fn calculate_time_limit(params: &GoParams, side: Color) -> Option<Duration> {
    // Fixed time per move
    if let Some(movetime) = params.movetime {
        // Use slightly less than movetime to account for overhead
        return Some(Duration::from_millis(movetime.saturating_sub(50)));
    }

    // Infinite search - no time limit
    if params.infinite {
        return None;
    }

    // Get our time and increment
    let (our_time, our_inc) = match side {
        Color::White => (params.wtime?, params.winc.unwrap_or(0)),
        Color::Black => (params.btime?, params.binc.unwrap_or(0)),
    };

    // Simple time management:
    // - If movestogo is set, divide remaining time by moves to go
    // - Otherwise, assume ~30 moves left in the game
    // - Add a portion of increment
    // - Use a safety margin to avoid flagging

    let moves_to_go = params.movestogo.unwrap_or(30) as u64;

    // Base time: divide remaining time by expected moves
    let base_time = our_time / moves_to_go.max(1);

    // Add most of our increment (save a little for safety)
    let with_inc = base_time + (our_inc * 3 / 4);

    // Don't use more than 1/5 of remaining time on any single move
    let max_time = our_time / 5;

    // Apply limits and safety margin
    let time_ms = with_inc.min(max_time).saturating_sub(50);

    // Minimum 100ms to have any meaningful search
    if time_ms < 100 {
        return Some(Duration::from_millis(100));
    }

    Some(Duration::from_millis(time_ms))
}

fn print_info(info: &crate::search::SearchInfo, tt: &TranspositionTable) {
    let score_str = if is_mate_score(info.score) {
        let mate_dist = if info.score > 0 {
            (30000 - info.score + 1) / 2
        } else {
            -(30000 + info.score + 1) / 2
        };
        format!("mate {}", mate_dist)
    } else {
        format!("cp {}", info.score)
    };

    let pv_str: String = info
        .pv
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    println!(
        "info depth {} score {} nodes {} hashfull {} pv {}",
        info.depth, score_str, info.nodes, tt.hashfull(), pv_str
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_position_startpos() {
        let pos = parse_position(&["startpos"]);
        assert_eq!(pos.to_fen().split_whitespace().next(), Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"));
    }

    #[test]
    fn test_parse_position_with_moves() {
        let pos = parse_position(&["startpos", "moves", "e2e4", "e7e5"]);
        // After 1. e4 e5
        assert!(pos.ep_square().is_some());
    }

    #[test]
    fn test_parse_position_fen() {
        let pos = parse_position(&["fen", "8/8/8/8/8/8/8/4K2k", "w", "-", "-", "0", "1"]);
        assert_eq!(pos.pieces(crate::types::Color::White, crate::types::Piece::King).count(), 1);
    }
}
