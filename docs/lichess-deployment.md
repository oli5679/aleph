# Lichess Bot Deployment Specification

This document outlines how to deploy the Aleph chess engine as a bot on Lichess.

## Overview

Lichess provides a [Bot API](https://lichess.org/api#tag/Bot) that allows UCI-compatible chess engines to play on the platform. Since Aleph already implements the UCI protocol, deployment requires:

1. A Lichess bot account
2. A bridge application (lichess-bot)
3. A hosting environment

## Prerequisites

### Aleph UCI Compatibility

Aleph already supports the required UCI commands (`src/uci.rs`):
- `uci` - Engine identification
- `isready` / `readyok` - Synchronization
- `ucinewgame` - Reset state
- `position startpos/fen moves ...` - Set position
- `go depth/wtime/btime/winc/binc` - Start search
- `bestmove` - Return result

### Required Before Deployment

| Feature | Status | Priority |
|---------|--------|----------|
| UCI protocol | ✅ Complete | - |
| Time management | ⚠️ Parsed but not used | High |
| `stop` command | ⚠️ TODO | High |
| Pondering | ❌ Not implemented | Low |

**Note:** Time management (`wtime`, `btime`, etc.) is parsed in `parse_go_depth()` but not implemented. This should be completed before deployment for competitive play (see `plan.md` Phase 2).

---

## Step 1: Create Lichess Bot Account

1. **Create a new Lichess account** at https://lichess.org/signup
   - Use a name like `AlephBot` or `Aleph-Engine`
   - **CRITICAL:** Do not play any games on this account before upgrading

2. **Generate an API token** at https://lichess.org/account/oauth/token
   - Select scopes: `bot:play`, `challenge:read`, `challenge:write`
   - Save the token securely (it won't be shown again)

3. **Upgrade to bot account** (irreversible):
   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

---

## Step 2: Choose a Bridge

### Option A: lichess-bot (Recommended)

The [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) project is the most actively maintained bridge.

**Features:**
- Python 3.10+
- Supports all variants and time controls
- Opening book support (Polyglot)
- Endgame tablebase support (Syzygy)
- Matchmaking (auto-challenge other bots)
- Tournament participation
- Docker support

**Setup:**
```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt

# Copy Aleph binary
mkdir -p engines
cp /path/to/target/release/aleph engines/

# Configure
cp config.yml.default config.yml
```

**config.yml:**
```yaml
token: "YOUR_LICHESS_TOKEN"

engine:
  dir: "./engines"
  name: "aleph"
  protocol: "uci"

  # UCI options (optional)
  uci_options:
    Hash: 128
    # Threads: 1  # When multi-threading is implemented

challenge:
  concurrency: 1  # Games to play simultaneously
  min_rating: 0
  max_rating: 4000

  # Accept these time controls
  time_controls:
    bullet: true
    blitz: true
    rapid: true
    classical: true

# Auto-accept challenges
accept_bot: true
accept_human: true
```

### Option B: BotLi

[BotLi](https://github.com/Torom/BotLi) is an alternative with more configuration options.

**Advantages:**
- Different engines per time control
- More granular challenge filtering
- Built-in Dockerfile

---

## Step 3: Local Testing

Before deploying, test locally:

```bash
# Terminal 1: Start the bot
cd lichess-bot
python lichess-bot.py

# The bot should connect and show:
# "Connected to lichess.org"
# "Waiting for challenges..."
```

Then challenge your bot from another Lichess account to verify it works.

---

## Step 4: Hosting Options

### Option A: Local Machine (Development)

Run directly on your machine. Not suitable for 24/7 operation.

```bash
python lichess-bot.py
```

### Option B: Railway (Recommended for simplicity)

[Railway](https://railway.app) provides simple deployment with a $5/month starter plan.

1. Create `Procfile`:
   ```
   worker: python lichess-bot.py
   ```

2. Create `runtime.txt`:
   ```
   python-3.11
   ```

3. Deploy via GitHub integration or CLI

4. Set environment variable `LICHESS_TOKEN` in Railway dashboard

### Option C: Fly.io (Recommended for performance)

[Fly.io](https://fly.io) offers edge deployment for low latency.

1. Install flyctl: `brew install flyctl`

2. Create `fly.toml`:
   ```toml
   app = "aleph-chess-bot"

   [build]
     dockerfile = "Dockerfile"

   [env]
     # Non-sensitive config here

   [[services]]
     internal_port = 8080
     protocol = "tcp"
   ```

3. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   RUN pip install --no-cache-dir requests chess pyyaml

   # Clone lichess-bot
   RUN apt-get update && apt-get install -y git && \
       git clone https://github.com/lichess-bot-devs/lichess-bot.git . && \
       pip install -r requirements.txt

   # Copy Aleph binary (pre-compiled for linux-amd64)
   COPY aleph engines/aleph
   RUN chmod +x engines/aleph

   # Copy config
   COPY config.yml .

   CMD ["python", "lichess-bot.py"]
   ```

4. Deploy:
   ```bash
   fly launch
   fly secrets set LICHESS_TOKEN=your_token
   fly deploy
   ```

### Option D: VPS (Full control)

Any Linux VPS (DigitalOcean, Linode, Vultr ~$5/month) works well.

```bash
# On server
sudo apt update
sudo apt install python3 python3-pip

# Clone and setup
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip3 install -r requirements.txt

# Copy binary (cross-compile for Linux if needed)
scp target/release/aleph user@server:lichess-bot/engines/

# Run with systemd or screen/tmux
screen -S lichess-bot
python3 lichess-bot.py
```

---

## Step 5: Cross-Compilation for Linux

If deploying to Linux from macOS:

```bash
# Install cross-compilation toolchain
rustup target add x86_64-unknown-linux-gnu

# Option A: Using cross (recommended)
cargo install cross
cross build --release --target x86_64-unknown-linux-gnu

# Option B: Using musl for static binary
rustup target add x86_64-unknown-linux-musl
brew install filosottile/musl-cross/musl-cross
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release --target x86_64-unknown-linux-musl
```

The resulting binary will be at:
- `target/x86_64-unknown-linux-gnu/release/aleph` or
- `target/x86_64-unknown-linux-musl/release/aleph`

---

## Step 6: Monitoring

### Lichess Dashboard

View your bot's games and stats at:
- `https://lichess.org/@/YOUR_BOT_NAME`
- `https://lichess.org/@/YOUR_BOT_NAME/tv` (live games)

### Logging

lichess-bot logs to stdout. For production, redirect to a file:

```bash
python lichess-bot.py 2>&1 | tee -a bot.log
```

---

## Implementation Checklist

### Before First Deployment
- [ ] Complete time management in Aleph (Phase 2)
- [ ] Implement `stop` command for interrupting search
- [ ] Create new Lichess account (fresh, no games played)
- [ ] Generate API token with correct scopes
- [ ] Upgrade account to bot (irreversible)

### Deployment
- [ ] Clone and configure lichess-bot
- [ ] Copy Aleph binary to `engines/`
- [ ] Configure `config.yml`
- [ ] Test locally with a challenge from another account
- [ ] Choose and configure hosting platform
- [ ] Deploy and verify bot connects

### Post-Deployment
- [ ] Monitor first 10+ games for issues
- [ ] Check for timeout losses (indicates time management issues)
- [ ] Verify challenge acceptance/rejection works correctly

---

## Future Enhancements

Once basic deployment works:

1. **Opening book** - Add Polyglot book support via lichess-bot config
2. **Endgame tablebases** - Syzygy for perfect endgame play
3. **Multi-threading** - Utilize server CPU cores
4. **Matchmaking** - Auto-challenge other bots for continuous play
5. **Rating tracking** - Monitor ELO progression over time
6. **Variant support** - Chess960, etc.

---

## Troubleshooting

### Bot doesn't respond to challenges
- Check token has correct scopes
- Verify bot account upgrade completed
- Check `config.yml` challenge filters aren't too restrictive

### Games timeout
- Time management not implemented - bot plays without clock awareness
- Implement `wtime`/`btime` handling in search

### "Engine process died" errors
- Binary not found or wrong architecture
- Check binary permissions (`chmod +x`)
- Verify cross-compilation target matches server

### High latency moves
- Consider Fly.io edge deployment
- Reduce search depth for bullet/blitz
- Implement pondering

---

## Resources

- [Lichess Bot API Documentation](https://lichess.org/api#tag/Bot)
- [lichess-bot GitHub](https://github.com/lichess-bot-devs/lichess-bot)
- [lichess-bot Wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki)
- [BotLi GitHub](https://github.com/Torom/BotLi)
- [UCI Protocol Specification](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html)
