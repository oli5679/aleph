#!/usr/bin/env python3
"""
HalfKAv2 feature extraction in Python.
Mirrors the Rust implementation for training data generation.
"""

import numpy as np
from typing import Optional

# Constants matching Rust code
NUM_KING_BUCKETS = 32
NUM_PIECE_SQUARES = 11 * 64  # 11 piece types * 64 squares
HALF_DIMENSIONS = NUM_KING_BUCKETS * NUM_PIECE_SQUARES  # 22528

# Piece indices
PIECE_TO_IDX = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

def parse_fen(fen: str) -> tuple[list[tuple[int, int, str]], bool]:
    """
    Parse FEN string to (pieces, white_to_move).
    pieces: list of (square_idx, color_is_white, piece_char)
    """
    parts = fen.split()
    board_str = parts[0]
    stm = parts[1] if len(parts) > 1 else 'w'

    pieces = []
    rank = 7
    file = 0

    for char in board_str:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            square = rank * 8 + file
            is_white = char.isupper()
            pieces.append((square, is_white, char.upper()))
            file += 1

    return pieces, stm == 'w'


def king_bucket(king_sq: int) -> int:
    """Get king bucket index (0-31) with horizontal mirroring."""
    file = king_sq % 8
    rank = king_sq // 8
    # Mirror files e-h (4-7) to a-d (3-0)
    mirrored_file = 7 - file if file >= 4 else file
    return rank * 4 + mirrored_file


def needs_mirror(king_sq: int) -> bool:
    """Check if squares need horizontal mirroring."""
    return (king_sq % 8) >= 4


def mirror_square(sq: int) -> int:
    """Mirror a square horizontally."""
    file = sq % 8
    rank = sq // 8
    return rank * 8 + (7 - file)


def flip_rank(sq: int) -> int:
    """Flip a square vertically (for black's perspective)."""
    file = sq % 8
    rank = sq // 8
    return (7 - rank) * 8 + file


def piece_type_index(piece: str, is_own: bool) -> int:
    """Get piece type index (0-10) for feature calculation."""
    base = PIECE_TO_IDX[piece]
    return base if is_own else 6 + base


def make_feature_index(perspective_is_white: bool, piece_sq: int, piece: str,
                       piece_is_white: bool, king_sq: int) -> Optional[int]:
    """
    Calculate feature index for a piece from a specific perspective.
    Returns None for opponent king (not a feature).
    """
    # Skip opponent king
    if piece == 'K' and piece_is_white != perspective_is_white:
        return None

    bucket = king_bucket(king_sq)
    mirror = needs_mirror(king_sq)

    # Orient square based on perspective and mirroring
    sq = piece_sq
    if not perspective_is_white:
        sq = flip_rank(sq)
    if mirror:
        sq = mirror_square(sq)

    is_own = piece_is_white == perspective_is_white
    piece_idx = piece_type_index(piece, is_own)

    return bucket * NUM_PIECE_SQUARES + piece_idx * 64 + sq


def extract_features(fen: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Extract HalfKAv2 features from FEN.

    Returns:
        white_features: [HALF_DIMENSIONS] sparse binary features for white perspective
        black_features: [HALF_DIMENSIONS] sparse binary features for black perspective
        white_to_move: bool
    """
    pieces, white_to_move = parse_fen(fen)

    # Find king squares
    white_king_sq = None
    black_king_sq = None
    for sq, is_white, piece in pieces:
        if piece == 'K':
            if is_white:
                white_king_sq = sq
            else:
                black_king_sq = sq

    if white_king_sq is None or black_king_sq is None:
        # Invalid position, return zeros
        return np.zeros(HALF_DIMENSIONS), np.zeros(HALF_DIMENSIONS), white_to_move

    # Extract features for each perspective
    white_features = np.zeros(HALF_DIMENSIONS, dtype=np.float32)
    black_features = np.zeros(HALF_DIMENSIONS, dtype=np.float32)

    for sq, is_white, piece in pieces:
        # White perspective
        idx = make_feature_index(True, sq, piece, is_white, white_king_sq)
        if idx is not None and 0 <= idx < HALF_DIMENSIONS:
            white_features[idx] = 1.0

        # Black perspective (use black's king for bucket calculation)
        idx = make_feature_index(False, sq, piece, is_white, black_king_sq)
        if idx is not None and 0 <= idx < HALF_DIMENSIONS:
            black_features[idx] = 1.0

    return white_features, black_features, white_to_move


def position_to_fen(position_spec: str) -> str:
    """
    Convert position specification to FEN.
    Handles both FEN strings and "startpos moves ..." format.
    """
    if not position_spec:
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    if position_spec.startswith("startpos"):
        # Parse move sequence and apply
        parts = position_spec.split()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        if len(parts) > 2 and parts[1] == "moves":
            moves = parts[2:]
            fen = apply_moves(fen, moves)

        return fen
    else:
        # Already a FEN - but might be missing halfmove/fullmove clocks
        parts = position_spec.split()
        if len(parts) == 4:
            # Add default halfmove and fullmove clocks
            return position_spec + " 0 1"
        return position_spec


def apply_moves(fen: str, moves: list[str]) -> str:
    """Apply a sequence of UCI moves to a FEN, returning final FEN."""
    # Simple board representation for move application
    board = [[None for _ in range(8)] for _ in range(8)]
    parts = fen.split()
    board_str = parts[0]

    # Parse initial position
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

    stm_is_white = parts[1] == 'w' if len(parts) > 1 else True
    castling = parts[2] if len(parts) > 2 else "KQkq"
    ep = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    def sq_to_coords(sq_str):
        file = ord(sq_str[0]) - ord('a')
        rank = int(sq_str[1]) - 1
        return (rank, file)

    for move in moves:
        if len(move) < 4:
            continue

        from_rank, from_file = sq_to_coords(move[0:2])
        to_rank, to_file = sq_to_coords(move[2:4])
        piece = board[from_rank][from_file]

        if piece is None:
            continue

        # Handle promotion
        promo = None
        if len(move) == 5:
            promo = move[4]
            if stm_is_white:
                promo = promo.upper()
            else:
                promo = promo.lower()

        # Handle castling
        if piece.upper() == 'K' and abs(from_file - to_file) == 2:
            # Castling
            if to_file > from_file:
                # Kingside
                rook = board[from_rank][7]
                board[from_rank][7] = None
                board[from_rank][5] = rook
            else:
                # Queenside
                rook = board[from_rank][0]
                board[from_rank][0] = None
                board[from_rank][3] = rook

        # Handle en passant
        if piece.upper() == 'P' and from_file != to_file and board[to_rank][to_file] is None:
            # En passant capture
            board[from_rank][to_file] = None

        # Make the move
        board[from_rank][from_file] = None
        board[to_rank][to_file] = promo if promo else piece

        stm_is_white = not stm_is_white
        if stm_is_white:
            fullmove += 1

    # Convert back to FEN
    fen_parts = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            if board[rank][file] is None:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += board[rank][file]
        if empty > 0:
            row += str(empty)
        fen_parts.append(row)

    stm = 'w' if stm_is_white else 'b'
    return f"{'/'.join(fen_parts)} {stm} - - 0 1"


# Test
if __name__ == "__main__":
    # Test with starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    white_feat, black_feat, stm = extract_features(fen)

    print(f"FEN: {fen}")
    print(f"White features active: {np.sum(white_feat > 0)}")
    print(f"Black features active: {np.sum(black_feat > 0)}")
    print(f"White to move: {stm}")

    # Test with a simple position after e4
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    white_feat2, black_feat2, stm2 = extract_features(fen2)
    print(f"\nFEN: {fen2}")
    print(f"White features active: {np.sum(white_feat2 > 0)}")
    print(f"Black features active: {np.sum(black_feat2 > 0)}")
    print(f"White to move: {stm2}")

    # Test move application
    pos = "startpos moves e2e4 e7e5"
    fen3 = position_to_fen(pos)
    print(f"\nPosition: {pos}")
    print(f"FEN: {fen3}")
