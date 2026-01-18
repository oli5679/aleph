#!/bin/bash
# Test Aleph vs Stockfish at various skill levels
#
# Usage:
#   ./scripts/test-vs-stockfish.sh [elo] [games] [depth]
#
# Examples:
#   ./scripts/test-vs-stockfish.sh 1500 20 4    # Test vs SF ELO 1500, 20 games, depth 4
#   ./scripts/test-vs-stockfish.sh 2100 10 6    # Test vs SF ELO 2100, 10 games, depth 6
#   ./scripts/test-vs-stockfish.sh              # Run full skill estimation

set -e

ALEPH="./target/release/aleph"
CUTECHESS="$HOME/bin/cutechess-cli"
STOCKFISH="stockfish"

# Build release if needed
if [ ! -f "$ALEPH" ]; then
    echo "Building Aleph..."
    RUSTFLAGS="-C target-cpu=native" cargo build --release
fi

# Single match function
run_match() {
    local elo=$1
    local games=$2
    local depth=$3

    echo "Testing Aleph vs Stockfish ELO $elo (${games} games, depth ${depth})..."

    $CUTECHESS \
        -engine name=Aleph cmd="$ALEPH" proto=uci \
        -engine name="SF-$elo" cmd="$STOCKFISH" proto=uci \
            option.UCI_LimitStrength=true \
            option.UCI_Elo=$elo \
        -each tc=inf depth=$depth \
        -games $games \
        -repeat \
        -recover \
        2>/dev/null
}

# Full skill estimation
estimate_skill() {
    echo "=== Aleph Skill Estimation ==="
    echo ""

    local depth=4
    local games=10

    # Test at various ELO levels
    for elo in 1320 1500 1750 2000 2100 2250 2400; do
        echo ""
        echo "--- Testing vs Stockfish ELO $elo ---"
        run_match $elo $games $depth
        echo ""
    done

    echo ""
    echo "=== Estimation Complete ==="
    echo ""
    echo "Interpretation:"
    echo "  Win rate > 70%: Engine is stronger than this ELO"
    echo "  Win rate 40-60%: Engine is approximately this ELO"
    echo "  Win rate < 30%: Engine is weaker than this ELO"
}

# Main
if [ $# -eq 0 ]; then
    estimate_skill
elif [ $# -eq 3 ]; then
    run_match $1 $2 $3
else
    echo "Usage: $0 [elo] [games] [depth]"
    echo "       $0                        # Full skill estimation"
    exit 1
fi
