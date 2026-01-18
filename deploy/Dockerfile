# Multi-stage build for Aleph + lichess-bot

# Stage 1: Build Aleph
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary
RUN cargo build --release

# Stage 2: Runtime with lichess-bot
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install git and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone lichess-bot
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create engines directory and copy Aleph binary
RUN mkdir -p engines
COPY --from=builder /build/target/release/aleph engines/aleph
RUN chmod +x engines/aleph

# Copy config template (will be overwritten by mounted config or env vars)
COPY deploy/config.yml config.yml

# The entrypoint script handles token injection
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cloud Run expects port 8080 but lichess-bot doesn't serve HTTP
# We'll use a dummy health check endpoint
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
