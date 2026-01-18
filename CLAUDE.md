# Aleph Chess Engine - Development Instructions

## Before Starting Any Task

1. **Review documentation**
   - Read `plan.md` for current implementation status and next steps
   - Read `docs/` files for component-specific design details

2. **Ask clarifying questions** before writing code
   - Confirm understanding of requirements
   - Clarify any ambiguities in the spec
   - Discuss trade-offs if multiple approaches exist

## Branch Workflow

When creating a new feature branch:

1. **Create branch from main**
   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **After implementing changes, review code quality**
   - Use type system to constrain valid states (not defensive runtime checks)
   - Avoid overly defensive code patterns
   - Prefer compile-time guarantees over runtime validation
   - Remove dead code paths and unused variables
   - Run `cargo clippy` for additional linting

3. **Run full test suite before PR**
   ```bash
   cargo test
   cargo build --release
   ./target/release/aleph bench
   ```

4. **Benchmark against Stockfish** (see below) to verify no regression

## Implementation Workflow

1. **Pick the next item from plan.md** (Phase 1 first, then Phase 2, etc.)

2. **Implement clean, clear code**
   - Prefer simplicity over cleverness
   - Follow patterns established in existing code
   - No over-engineering - implement what's needed, nothing more
   - No hardcoded magic constants for pruning (that's the whole point)
   - Use the type system to make invalid states unrepresentable

3. **After implementation, verify:**
   - **Logical correctness**: Does the code do what it's supposed to?
   - **Not hacky**: Is it clean and maintainable?
   - **Not over-engineered**: Is it solving only the problem at hand?

4. **Create test cases and run them**
   - Unit tests for new functionality
   - Perft tests for move generation
   - Integration tests where appropriate
   - `cargo test` must pass

5. **Update documentation**
   - Mark completed items in `plan.md`
   - Add/update doc comments in code

## Code Quality Standards

- **Performance-first**: This is a chess engine, performance matters
- **Zero-cost abstractions**: Use `#[repr(transparent)]`, `#[inline]` where appropriate
- **Clean interfaces**: Components should have clear boundaries
- **Type-driven design**: Use newtypes and enums to prevent invalid states
- **No magic constants**: Pruning thresholds come from quantile evaluation, not hardcoded values

## Testing Commands

```bash
# Run all tests
cargo test

# Build release (optimized for current CPU)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run perft benchmark (verifies move generation)
./target/release/aleph bench

# Search a position
./target/release/aleph search 8

# Start UCI mode
./target/release/aleph
```

## Benchmarking vs Stockfish

### Prerequisites
- `stockfish` installed via `brew install stockfish`
- `cutechess-cli` installed in `~/bin/` (built from source)

### Quick Benchmark (Depth-Based)

Since we don't have time management yet, use fixed depth with UCI_LimitStrength:

```bash
# Test vs Stockfish at ELO 1320 (lowest)
~/bin/cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph proto=uci \
  -engine name="SF-1320" cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=1320 \
  -each tc=inf depth=4 \
  -games 10 -repeat

# Test vs Stockfish at ELO 2100
~/bin/cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph proto=uci \
  -engine name="SF-2100" cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2100 \
  -each tc=inf depth=4 \
  -games 10 -repeat

# Test vs Stockfish at ELO 2250
~/bin/cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph proto=uci \
  -engine name="SF-2250" cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2250 \
  -each tc=inf depth=4 \
  -games 10 -repeat
```

### Interpreting Results

- **Win rate > 70%**: Engine is stronger than this ELO level, test higher
- **Win rate 40-60%**: Engine is approximately this ELO level
- **Win rate < 30%**: Engine is weaker than this ELO level

### Current Baseline (NNUE Infrastructure, depth=4)

| SF UCI_Elo | W-L-D   | Win Rate | Notes |
|------------|---------|----------|-------|
| 1500       | 10-0-0  | 100%     | Dominates |
| 2000       | 7-1-2   | 80%      | Much stronger |
| 2200       | 7-3-0   | 70%      | Stronger |
| 2400       | 6-3-1   | 65%      | Competitive edge |
| 2500       | 5-3-2   | 60%      | Slight edge |
| 2600       | 1-6-3   | 25%      | Outclassed |

**Estimated Aleph ELO: ~2450-2500** (at depth 4)

### Regression Testing

Before merging any changes, run this regression test to verify no performance degradation:

```bash
# Regression test: must achieve >= 50% against SF ELO 2400
~/bin/cutechess-cli \
  -engine name=Aleph cmd=./target/release/aleph proto=uci \
  -engine name="SF-2400" cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2400 \
  -each tc=inf depth=4 \
  -games 20 -repeat \
  -pgnout regression_test.pgn

# Expected: >= 10/20 wins+draws (50%+)
# If significantly below 50%, there may be a regression
```

**Regression Criteria:**
- Must achieve >= 50% score against SF ELO 2400 at depth 4
- Must not drop more than 10 percentage points from baseline at any level
- Any new loss patterns (e.g., repeated mating threats) should be investigated

### Phase 2 Targets

After implementing TT, move ordering, and time management:
- Target: Beat SF ELO 2400 at equal time control
- Target Elo: ~2300-2500

## Current Status

Check `plan.md` for:
- [x] Completed items
- [ ] Pending items
- Current phase and milestone

**Phase 1 Complete** - Estimated ELO: ~2100-2150

Phase 2 priorities:
- Transposition table (will improve search efficiency)
- Move ordering (MVV-LVA, killers, history heuristic)
- Time management (enables real time control matches)
- PVS search (principal variation search)
