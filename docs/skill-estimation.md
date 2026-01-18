# Aleph Engine Skill Estimation

## Test Configuration

- **Test Method**: cutechess-cli matches
- **Search Depth**: Fixed depth 4
- **Games per Level**: 10 games (alternating colors)
- **Opponent**: Stockfish 17.1 with UCI_LimitStrength

## Results Summary

| SF UCI_Elo | W-L-D   | Score  | Interpretation |
|------------|---------|--------|----------------|
| 1500       | 10-0-0  | 100%   | Dominates |
| 2000       | 7-1-2   | 80%    | Much stronger |
| 2200       | 7-3-0   | 70%    | Stronger |
| 2400       | 6-3-1   | 65%    | Competitive edge |
| 2500       | 5-3-2   | 60%    | Slight edge |
| 2600       | 1-6-3   | 25%    | Outclassed |

## Estimated ELO: **~2450-2500**

Based on these results:
- Aleph maintains winning chances up to ~2500 ELO
- Performance drops sharply against 2600+ opponents
- The engine plays solidly at the Expert/Candidate Master level

## Interpretation Guide

| Win Rate   | Meaning |
|------------|---------|
| > 70%      | Engine is significantly stronger |
| 55-70%     | Engine has slight advantage |
| 45-55%     | Approximately equal strength |
| 30-45%     | Engine is slightly weaker |
| < 30%      | Engine is significantly weaker |

## Running Your Own Tests

```bash
# Quick test at specific ELO
./scripts/test-vs-stockfish.sh 2200 10 4

# Full skill estimation
./scripts/test-vs-stockfish.sh

# Manual test with cutechess-cli
~/bin/cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph proto=uci \
  -engine name="SF-2200" cmd=stockfish proto=uci \
      option.UCI_LimitStrength=true option.UCI_Elo=2200 \
  -each tc=inf depth=4 \
  -games 20 -repeat
```

## Notes

- Tests use fixed depth (no time control) for consistency
- Stockfish's UCI_LimitStrength simulates weaker play
- ELO estimates are approximate and may vary with different test conditions
- Higher search depths would likely yield different results
