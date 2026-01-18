# Aleph Chess Engine - Development Instructions

## Before Starting Any Task

1. **Review documentation**
   - Read `plan.md` for current implementation status and next steps
   - Read `docs/` files for component-specific design details
   - Review `../Stockfish/src/` and `../pleco/` for reference implementations

2. **Ask clarifying questions** before writing code
   - Confirm understanding of requirements
   - Clarify any ambiguities in the spec
   - Discuss trade-offs if multiple approaches exist

## Implementation Workflow

1. **Pick the next item from plan.md** (Phase 1 first, then Phase 2, etc.)

2. **Implement clean, clear code**
   - Prefer simplicity over cleverness
   - Follow patterns established in existing code
   - No over-engineering - implement what's needed, nothing more
   - No hardcoded magic constants for pruning (that's the whole point)

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
   - Update README if user-facing changes
   - Add/update doc comments in code

## Code Quality Standards

- **Performance-first**: This is a chess engine, performance matters
- **Zero-cost abstractions**: Use `#[repr(transparent)]`, `#[inline(always)]` where appropriate
- **Clean interfaces**: Components should have clear boundaries
- **No magic constants**: Pruning thresholds come from quantile evaluation, not hardcoded values

## Reference Implementations

Study but don't copy:
- `../Stockfish/src/` - State of the art, but complex
- `../pleco/` - Rust idioms, good patterns

Our goal is cleaner, simpler code with the novel distributional evaluation approach.

## Testing Commands

```bash
# Run all tests
cargo test

# Build release
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run perft (when implemented)
./target/release/aleph bench perft

# Play vs Stockfish (when UCI implemented)
cutechess-cli -engine name=Aleph cmd=./target/release/aleph -engine name=Stockfish cmd=stockfish -each proto=uci tc=inf -games 10
```

## Current Status

Check `plan.md` for:
- [x] Completed items
- [ ] Pending items
- Current phase and milestone
