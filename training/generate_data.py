#!/usr/bin/env python3
"""
Generate training data for uncertainty NNUE.

For each position:
1. Search at multiple depths (4, 8, 12)
2. Record eval at each depth
3. Compute target quantiles from eval distribution
4. Record best move for policy training

Usage:
    python generate_data.py --engine ../target/release/aleph --output training_data.jsonl --positions 50000
"""

import argparse
import json
import subprocess
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class SearchResult:
    depth: int
    score: int
    best_move: str
    nodes: int


@dataclass
class TrainingExample:
    fen: str
    evals: list[int]  # Evals at each depth
    best_move: str    # Best move from deepest search
    quantiles: list[float]  # Target quantiles [q10, q25, q50, q75, q90]


class UCIEngine:
    """Simple UCI engine wrapper."""

    def __init__(self, path: str):
        self.path = path
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        self.process = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def stop(self):
        if self.process:
            self._send("quit")
            self.process.wait(timeout=5)
            self.process = None

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def set_position(self, fen: str):
        if fen == "startpos":
            self._send("position startpos")
        else:
            self._send(f"position fen {fen}")

    def search(self, depth: int) -> SearchResult:
        self._send(f"go depth {depth}")

        score = 0
        best_move = ""
        nodes = 0

        while True:
            line = self._read_line()
            if line.startswith("info"):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "score" and i + 2 < len(parts):
                        if parts[i + 1] == "cp":
                            score = int(parts[i + 2])
                        elif parts[i + 1] == "mate":
                            mate_in = int(parts[i + 2])
                            score = 30000 - abs(mate_in) * 100
                            if mate_in < 0:
                                score = -score
                    elif part == "nodes" and i + 1 < len(parts):
                        nodes = int(parts[i + 1])
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        return SearchResult(depth=depth, score=score, best_move=best_move, nodes=nodes)

    def get_legal_moves(self, fen: str) -> list[str]:
        """Get legal moves by trying each possible move."""
        # This is a hacky way - ideally we'd have a proper move gen
        # For now, just return empty and handle in self-play
        return []

    def _send(self, cmd: str):
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def _read_line(self) -> str:
        if self.process and self.process.stdout:
            return self.process.stdout.readline().strip()
        return ""

    def _wait_for(self, expected: str):
        while True:
            line = self._read_line()
            if expected in line:
                break


def compute_quantiles(values: list[int], quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> list[float]:
    """Compute quantiles from a list of values."""
    if len(values) < 2:
        # If only one value, all quantiles are the same
        return [float(values[0])] * len(quantiles)

    # Add some noise to avoid degenerate cases
    values_arr = np.array(values, dtype=float)

    # Compute quantiles
    result = np.percentile(values_arr, [q * 100 for q in quantiles])
    return result.tolist()


def generate_random_position(engine: UCIEngine, max_moves: int = 40) -> Optional[str]:
    """Generate a random position by playing random moves from startpos."""
    engine.set_position("startpos")

    fen = "startpos"
    moves = []

    # Play random number of moves
    num_moves = random.randint(5, max_moves)

    for _ in range(num_moves):
        # Use depth 1 search to get a legal move (hacky but works)
        result = engine.search(1)
        if not result.best_move or result.best_move == "(none)":
            break

        moves.append(result.best_move)

        # 70% chance to play best move, 30% chance to try random
        # (This creates more diverse positions)
        if random.random() < 0.3 and len(moves) > 1:
            # Try a different move by searching again with different seed
            # For now just use the best move
            pass

        # Update position
        engine._send(f"position startpos moves {' '.join(moves)}")

    # Get final FEN (we need to extract it somehow)
    # For now, return the move sequence as a pseudo-FEN
    # In practice, you'd want to get the actual FEN from the engine

    # Actually, let's just track with moves for now
    return f"startpos moves {' '.join(moves)}" if moves else None


def generate_position_from_games(game_fens: list[str]) -> str:
    """Pick a random position from pre-loaded game FENs."""
    return random.choice(game_fens)


# Some common opening positions to start from
STARTING_POSITIONS = [
    "startpos",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",  # 1.c4
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",   # 1.Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",  # 1.d4 d5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # 1.e4 e5 2.Nf3
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # 1.e4 e5 2.Nf3 Nc6
]


def self_play_game(engine: UCIEngine, search_depth: int = 4) -> list[str]:
    """Play a self-play game and return FENs of positions."""
    engine.new_game()
    engine.set_position("startpos")

    positions = []
    moves = []

    for move_num in range(100):  # Max 100 ply
        # Search for best move
        result = engine.search(search_depth)

        if not result.best_move or result.best_move == "(none)":
            break

        # Check for game end (mate scores)
        if abs(result.score) > 25000:
            break

        moves.append(result.best_move)

        # Store position as "startpos moves ..."
        if moves:
            positions.append(f"startpos moves {' '.join(moves)}")

        # Update position for next move
        engine._send(f"position startpos moves {' '.join(moves)}")

    return positions


def main():
    parser = argparse.ArgumentParser(description="Generate NNUE training data")
    parser.add_argument("--engine", type=str, default="../target/release/aleph",
                       help="Path to UCI engine")
    parser.add_argument("--output", type=str, default="training_data.jsonl",
                       help="Output file (JSONL format)")
    parser.add_argument("--positions", type=int, default=50000,
                       help="Number of positions to generate")
    parser.add_argument("--depths", type=str, default="4,8,12",
                       help="Comma-separated search depths")
    parser.add_argument("--games", type=int, default=500,
                       help="Number of self-play games")
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    print(f"Generating {args.positions} positions with depths {depths}")

    # Check engine exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Build with: cargo build --release")
        sys.exit(1)

    engine = UCIEngine(str(engine_path.absolute()))
    engine.start()

    examples = []
    output_path = Path(args.output)

    try:
        # Generate positions through self-play
        print(f"Playing {args.games} self-play games...")
        all_positions = []

        for game_num in range(args.games):
            if game_num % 50 == 0:
                print(f"  Game {game_num}/{args.games}...")
            positions = self_play_game(engine, search_depth=3)
            all_positions.extend(positions)

            if len(all_positions) >= args.positions:
                break

        print(f"Generated {len(all_positions)} raw positions from self-play")

        # Sample positions if we have too many
        if len(all_positions) > args.positions:
            all_positions = random.sample(all_positions, args.positions)

        # Now evaluate each position at multiple depths
        print(f"Evaluating {len(all_positions)} positions at depths {depths}...")

        with open(output_path, "w") as f:
            for i, pos_spec in enumerate(all_positions):
                if i % 1000 == 0:
                    print(f"  Position {i}/{len(all_positions)}...")

                engine.new_game()
                engine._send(f"position {pos_spec}")

                # Search at each depth
                evals = []
                best_move = ""

                for depth in depths:
                    result = engine.search(depth)
                    evals.append(result.score)
                    best_move = result.best_move  # Last (deepest) search's best move

                # Compute target quantiles
                quantiles = compute_quantiles(evals)

                # Create example
                example = {
                    "position": pos_spec,
                    "evals": evals,
                    "best_move": best_move,
                    "quantiles": quantiles,
                }

                f.write(json.dumps(example) + "\n")
                examples.append(example)

        print(f"Saved {len(examples)} examples to {output_path}")

        # Print some statistics
        all_q10 = [e["quantiles"][0] for e in examples]
        all_q90 = [e["quantiles"][4] for e in examples]
        uncertainties = [q90 - q10 for q10, q90 in zip(all_q10, all_q90)]

        print(f"\nStatistics:")
        print(f"  Mean uncertainty (q90-q10): {np.mean(uncertainties):.1f}")
        print(f"  Std uncertainty: {np.std(uncertainties):.1f}")
        print(f"  Min uncertainty: {np.min(uncertainties):.1f}")
        print(f"  Max uncertainty: {np.max(uncertainties):.1f}")

    finally:
        engine.stop()


if __name__ == "__main__":
    main()
