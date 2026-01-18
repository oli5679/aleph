#!/usr/bin/env python3
"""
Load Stockfish evaluation dataset from Hugging Face and convert to training format.

Dataset: https://huggingface.co/datasets/bingbangboom/stockfish-evaluation-SAN

Since this dataset has single-depth evals, we estimate uncertainty from:
1. Depth: Lower depth = higher uncertainty
2. Eval magnitude: Extreme evals tend to be more certain
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


# Piece mappings for SAN parsing
PIECE_CHARS = {'K': 'k', 'Q': 'q', 'R': 'r', 'B': 'b', 'N': 'n'}


def parse_fen_for_san(fen: str) -> dict:
    """Parse FEN to get board state for SAN conversion."""
    parts = fen.split()
    board_str = parts[0]
    stm = parts[1] if len(parts) > 1 else 'w'

    # Build board array
    board = [[None for _ in range(8)] for _ in range(8)]
    rank = 7
    file = 0
    for char in board_str:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            board[rank][file] = char
            file += 1

    return {'board': board, 'stm': stm}


def find_piece_square(board: list, piece: str, to_sq: tuple, from_hint: str = None) -> tuple:
    """Find the source square for a piece move."""
    to_rank, to_file = to_sq
    candidates = []

    for rank in range(8):
        for file in range(8):
            p = board[rank][file]
            if p != piece:
                continue

            # Check if this piece can reach to_sq
            if can_reach(piece, (rank, file), to_sq, board):
                candidates.append((rank, file))

    if len(candidates) == 1:
        return candidates[0]

    # Use disambiguation hint
    if from_hint:
        for r, f in candidates:
            if from_hint.isdigit() and int(from_hint) - 1 == r:
                return (r, f)
            if from_hint.isalpha() and ord(from_hint) - ord('a') == f:
                return (r, f)

    # Return first candidate if ambiguous
    return candidates[0] if candidates else None


def can_reach(piece: str, from_sq: tuple, to_sq: tuple, board: list) -> bool:
    """Check if piece can legally move from from_sq to to_sq (simplified)."""
    fr, ff = from_sq
    tr, tf = to_sq
    p = piece.upper()

    dr, df = tr - fr, tf - ff

    if p == 'N':
        return (abs(dr), abs(df)) in [(1, 2), (2, 1)]
    elif p == 'B':
        return abs(dr) == abs(df) and dr != 0
    elif p == 'R':
        return (dr == 0 or df == 0) and (dr != 0 or df != 0)
    elif p == 'Q':
        return (dr == 0 or df == 0 or abs(dr) == abs(df)) and (dr != 0 or df != 0)
    elif p == 'K':
        return abs(dr) <= 1 and abs(df) <= 1 and (dr != 0 or df != 0)
    elif p == 'P':
        # Simplified pawn logic
        return True

    return False


def san_to_uci(san: str, fen: str) -> str:
    """Convert SAN move to UCI format. Returns empty string on failure."""
    try:
        san = san.strip().rstrip('+#')

        if not san:
            return ""

        # Castling
        if san in ['O-O', '0-0']:
            stm = fen.split()[1] if len(fen.split()) > 1 else 'w'
            return 'e1g1' if stm == 'w' else 'e8g8'
        if san in ['O-O-O', '0-0-0']:
            stm = fen.split()[1] if len(fen.split()) > 1 else 'w'
            return 'e1c1' if stm == 'w' else 'e8c8'

        state = parse_fen_for_san(fen)
        board = state['board']
        stm = state['stm']

        # Parse the SAN
        promo = None
        if '=' in san:
            san, promo = san.split('=')
            promo = promo.lower()

        san = san.replace('x', '')  # Remove capture indicator

        # Pawn move
        if san[0].islower():
            to_file = ord(san[0]) - ord('a')

            # Check if it's a capture (has file disambiguation)
            if len(san) >= 3 and san[1].isalpha():
                from_file = to_file
                to_file = ord(san[1]) - ord('a')
                to_rank = int(san[2]) - 1
            else:
                to_rank = int(san[1]) - 1
                from_file = to_file

            # Find pawn
            pawn = 'P' if stm == 'w' else 'p'
            direction = 1 if stm == 'w' else -1

            from_rank = None
            for dr in [1, 2]:
                check_rank = to_rank - direction * dr
                if 0 <= check_rank < 8:
                    if board[check_rank][from_file] == pawn:
                        from_rank = check_rank
                        break

            # For captures, check diagonal
            if from_rank is None and len(san) >= 3:
                orig_from_file = ord(san[0]) - ord('a')
                check_rank = to_rank - direction
                if 0 <= check_rank < 8 and board[check_rank][orig_from_file] == pawn:
                    from_rank = check_rank
                    from_file = orig_from_file

            if from_rank is None:
                return ""

            from_sq = chr(ord('a') + from_file) + str(from_rank + 1)
            to_sq = chr(ord('a') + to_file) + str(to_rank + 1)

            uci = from_sq + to_sq
            if promo:
                uci += promo
            return uci

        # Piece move
        piece = san[0]
        san = san[1:]

        # Handle disambiguation
        disambig = ""
        if len(san) >= 3 and (san[0].isdigit() or (san[0].isalpha() and san[0] != san[-2])):
            disambig = san[0]
            san = san[1:]
        elif len(san) >= 4:
            disambig = san[0]
            san = san[1:]

        to_file = ord(san[0]) - ord('a')
        to_rank = int(san[1]) - 1

        # Find piece
        piece_char = piece if stm == 'w' else piece.lower()
        from_pos = find_piece_square(board, piece_char, (to_rank, to_file), disambig)

        if from_pos is None:
            return ""

        from_sq = chr(ord('a') + from_pos[1]) + str(from_pos[0] + 1)
        to_sq_str = chr(ord('a') + to_file) + str(to_rank + 1)

        return from_sq + to_sq_str

    except Exception as e:
        return ""


def parse_eval(eval_str: str) -> tuple[float, bool]:
    """Parse evaluation string. Returns (centipawns, is_mate)."""
    eval_str = eval_str.strip()

    if eval_str.startswith('M') or eval_str.startswith('#'):
        # Mate score
        mate_in = int(eval_str[1:]) if len(eval_str) > 1 else 1
        return 30000 - abs(mate_in) * 100, True
    elif eval_str.startswith('-M') or eval_str.startswith('-#'):
        mate_in = int(eval_str[2:]) if len(eval_str) > 2 else 1
        return -(30000 - abs(mate_in) * 100), True
    else:
        try:
            # Centipawn score (given as pawns, multiply by 100)
            return float(eval_str) * 100, False
        except:
            return 0.0, False


def estimate_quantiles(eval_cp: float, depth: int, is_mate: bool) -> list[float]:
    """
    Use the actual eval as the target for ALL quantiles.

    The pinball loss will naturally learn the spread:
    - tau=0.1 penalizes overestimates → network learns lower values for q10
    - tau=0.9 penalizes underestimates → network learns higher values for q90
    - tau=0.5 is symmetric → learns median

    By using the same target, the loss function does the work of finding
    the right quantile spread based on the variance in the training data.
    """
    # Clamp extreme values to avoid training instability
    eval_cp = max(-3000, min(3000, eval_cp))

    # All quantiles get the same target - pinball loss handles the rest
    return [eval_cp, eval_cp, eval_cp, eval_cp, eval_cp]


def process_dataset(num_samples: int = 50000, output_path: str = "training_data.jsonl"):
    """Download and process the HuggingFace dataset."""
    if not HAS_DATASETS:
        print("Please install datasets: pip install datasets")
        return

    print(f"Loading dataset from HuggingFace (streaming {num_samples} samples)...")

    # Load dataset in streaming mode to avoid downloading everything
    dataset = load_dataset(
        "bingbangboom/stockfish-evaluation-SAN",
        split="train",
        streaming=True
    )

    examples = []
    skipped = 0

    print("Processing examples...")
    for i, item in enumerate(dataset):
        if i >= num_samples * 2:  # Sample more to account for skips
            break

        if len(examples) >= num_samples:
            break

        if i % 10000 == 0:
            print(f"  Processed {i}, kept {len(examples)}, skipped {skipped}")

        fen = item['fen']
        depth = item['depth']
        eval_str = item['evaluation']
        best_move_san = item['best_move']

        # Parse evaluation
        eval_cp, is_mate = parse_eval(eval_str)

        # Convert SAN to UCI
        best_move_uci = san_to_uci(best_move_san, fen)
        if not best_move_uci or len(best_move_uci) < 4:
            skipped += 1
            continue

        # Estimate quantiles
        quantiles = estimate_quantiles(eval_cp, depth, is_mate)

        example = {
            "position": fen,
            "evals": [eval_cp],  # Single eval
            "best_move": best_move_uci,
            "quantiles": quantiles,
            "depth": depth,
        }
        examples.append(example)

    print(f"Processed {len(examples)} examples, skipped {skipped}")

    # Save to file
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Saved to {output_path}")

    # Print statistics
    all_q10 = [e["quantiles"][0] for e in examples]
    all_q90 = [e["quantiles"][4] for e in examples]
    uncertainties = [q90 - q10 for q10, q90 in zip(all_q10, all_q90)]

    print(f"\nStatistics:")
    print(f"  Mean uncertainty (q90-q10): {np.mean(uncertainties):.1f}")
    print(f"  Std uncertainty: {np.std(uncertainties):.1f}")
    print(f"  Min uncertainty: {np.min(uncertainties):.1f}")
    print(f"  Max uncertainty: {np.max(uncertainties):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Load HuggingFace chess dataset")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--output", type=str, default="training_data.jsonl", help="Output path")
    args = parser.parse_args()

    process_dataset(args.samples, args.output)


if __name__ == "__main__":
    main()
