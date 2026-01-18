use aleph::movegen::{perft, perft_divide};
use aleph::position::Position;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        // No arguments - start UCI mode
        aleph::uci::uci_loop();
        return;
    }

    match args[1].as_str() {
        "uci" => {
            aleph::uci::uci_loop();
        }
        "perft" => {
            let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = args.get(3).map(|s| s.as_str()).unwrap_or(aleph::position::STARTPOS);
            let mut pos = Position::from_fen(fen).expect("Invalid FEN");

            println!("Position: {}", fen);
            println!("Depth: {}", depth);
            println!();

            let start = Instant::now();
            let nodes = perft(&mut pos, depth);
            let elapsed = start.elapsed();

            let nps = nodes as f64 / elapsed.as_secs_f64();
            println!("Nodes: {}", nodes);
            println!("Time: {:.3}s", elapsed.as_secs_f64());
            println!("NPS: {:.0}", nps);
        }
        "divide" => {
            let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = args.get(3).map(|s| s.as_str()).unwrap_or(aleph::position::STARTPOS);
            let mut pos = Position::from_fen(fen).expect("Invalid FEN");

            println!("Position: {}", fen);
            println!("Depth: {}", depth);
            println!();

            perft_divide(&mut pos, depth);
        }
        "bench" => {
            run_bench();
        }
        "search" => {
            let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
            let fen = args.get(3).map(|s| s.as_str()).unwrap_or(aleph::position::STARTPOS);
            let pos = Position::from_fen(fen).expect("Invalid FEN");

            println!("Position: {}", fen);
            println!("Depth: {}", depth);
            println!("{}", pos);

            let eval = aleph::eval::classical::ClassicalEval::new();
            let mut searcher = aleph::search::Searcher::new(pos, eval);

            let start = Instant::now();
            let info = searcher.search(depth);
            let elapsed = start.elapsed();

            println!();
            println!("Best move: {}", if info.pv.is_empty() { "none".to_string() } else { info.pv[0].to_string() });
            println!("Score: {} cp", info.score);
            println!("Nodes: {}", info.nodes);
            println!("Time: {:.3}s", elapsed.as_secs_f64());
            println!("NPS: {:.0}", info.nodes as f64 / elapsed.as_secs_f64());
            print!("PV: ");
            for mv in &info.pv {
                print!("{} ", mv);
            }
            println!();
        }
        "--help" | "-h" | "help" => {
            print_help();
        }
        _ => {
            println!("Unknown command: {}", args[1]);
            print_help();
        }
    }
}

fn print_help() {
    println!("Aleph Chess Engine v0.1.0");
    println!();
    println!("Usage:");
    println!("  aleph                     - Start UCI mode");
    println!("  aleph uci                 - Start UCI mode");
    println!("  aleph search <depth> [fen] - Search position");
    println!("  aleph perft <depth> [fen]  - Run perft test");
    println!("  aleph divide <depth> [fen] - Perft with move breakdown");
    println!("  aleph bench               - Run perft benchmarks");
    println!("  aleph help                - Show this help");
}

fn run_bench() {
    let positions = [
        ("Startpos", aleph::position::STARTPOS, 6, 119_060_324u64),
        (
            "Kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            5,
            193_690_690,
        ),
        (
            "Position 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            7,
            178_633_661,
        ),
        (
            "Position 4",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            5,
            15_833_292,
        ),
        (
            "Position 5",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            5,
            89_941_194,
        ),
    ];

    let mut total_nodes = 0u64;
    let total_start = Instant::now();

    for (name, fen, depth, expected) in positions {
        let mut pos = Position::from_fen(fen).expect("Invalid FEN");

        let start = Instant::now();
        let nodes = perft(&mut pos, depth);
        let elapsed = start.elapsed();

        let status = if nodes == expected { "OK" } else { "FAIL" };
        println!(
            "{}: depth {} = {} ({}) [{:.3}s]",
            name,
            depth,
            nodes,
            status,
            elapsed.as_secs_f64()
        );

        total_nodes += nodes;

        if nodes != expected {
            println!("  Expected: {}", expected);
        }
    }

    let total_elapsed = total_start.elapsed();
    let nps = total_nodes as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("Total nodes: {}", total_nodes);
    println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
    println!("NPS: {:.0}", nps);
}
