use crate::eval::classical::ClassicalEval;
use crate::eval::is_mate_score;
use crate::position::Position;
use crate::search::Searcher;
use crate::types::Move;
use std::io::{self, BufRead, Write};

const ENGINE_NAME: &str = "Aleph";
const ENGINE_AUTHOR: &str = "Aleph Authors";

/// Run the UCI protocol loop
pub fn uci_loop() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut pos = Position::startpos();

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
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("uciok");
                stdout.flush().ok();
            }

            "isready" => {
                println!("readyok");
                stdout.flush().ok();
            }

            "ucinewgame" => {
                pos = Position::startpos();
            }

            "position" => {
                pos = parse_position(&tokens[1..]);
            }

            "go" => {
                let depth = parse_go_depth(&tokens[1..]);
                let eval = ClassicalEval::new();
                let mut searcher = Searcher::new(pos.clone(), eval);
                let info = searcher.search(depth);

                // Print search info
                print_info(&info, depth);

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
    use crate::types::Square;

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
                        crate::types::Piece::Queen => 'q',
                        crate::types::Piece::Rook => 'r',
                        crate::types::Piece::Bishop => 'b',
                        crate::types::Piece::Knight => 'n',
                        _ => continue,
                    };
                    if promo_char == p {
                        return Some(mv);
                    }
                }
            } else if !mv.is_promotion() {
                return Some(mv);
            } else {
                // Default to queen promotion
                if matches!(mv.promotion_piece(), crate::types::Piece::Queen) {
                    return Some(mv);
                }
            }
        }
    }

    // Try to construct basic move as fallback
    Move::from_uci(s)
}

fn parse_go_depth(tokens: &[&str]) -> i32 {
    let mut depth = 6; // Default depth

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if i + 1 < tokens.len() {
                    depth = tokens[i + 1].parse().unwrap_or(6);
                }
                i += 2;
            }
            "movetime" => {
                // TODO: Implement time management
                i += 2;
            }
            "wtime" | "btime" | "winc" | "binc" | "movestogo" => {
                // TODO: Implement time management
                i += 2;
            }
            "infinite" => {
                depth = 100;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    depth.min(64) // Cap at reasonable depth
}

fn print_info(info: &crate::search::SearchInfo, depth: i32) {
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
        "info depth {} score {} nodes {} pv {}",
        depth, score_str, info.nodes, pv_str
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
