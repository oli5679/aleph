#!/bin/bash
# Download the Stockfish NNUE network for Aleph
#
# Usage:
#   ./scripts/download-network.sh
#
# This downloads the network file used by the engine.

set -e

NETWORK_DIR="networks"
NETWORK_FILE="nn-c288c895ea92.nnue"
NETWORK_URL="https://tests.stockfishchess.org/api/nn/$NETWORK_FILE"

mkdir -p "$NETWORK_DIR"

if [ -f "$NETWORK_DIR/$NETWORK_FILE" ]; then
    echo "Network already exists: $NETWORK_DIR/$NETWORK_FILE"
    exit 0
fi

echo "Downloading $NETWORK_FILE..."
curl -L -o "$NETWORK_DIR/$NETWORK_FILE" "$NETWORK_URL"

echo "Downloaded to $NETWORK_DIR/$NETWORK_FILE"
ls -lh "$NETWORK_DIR/$NETWORK_FILE"
