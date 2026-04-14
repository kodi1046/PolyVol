FROM python:3.13-slim-bookworm

# System deps + Node.js 20 LTS (for Claude Code CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        sqlite3 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

# Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Dashboard port
EXPOSE 8080

# paper_trades.db lives outside the image so data survives restarts.
# Mount a host directory or named volume at /data and point the app there:
#   docker run -v ./data:/data -e DB_PATH=/data/paper_trades.db ...
# Without DB_PATH the db writes to /app/paper_trades.db inside the container.
ENV DB_PATH=""

# .env is NOT baked in — pass secrets via docker run --env-file .env
# Required env vars:
#   CHAINLINK_API_KEY        — Chainlink Data Streams key
#   CHAINLINK_API_SECRET     — Chainlink Data Streams secret
#   ANTHROPIC_API_KEY        — for Claude Code CLI (optional at runtime)
# Optional:
#   MIN_DEVIATION            — signal threshold, default 0.0003
#   CHAINLINK_POLL_SEC       — REST poll interval, default 1

CMD ["python", "main.py"]
FROM python:3.13-slim-bookworm

# 1. System deps + Node.js 20 + Git (Needed for Cloning and Claude Code)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        sqlite3 \
        tmux \
        curl \
        vim \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

# 3. Clone the repository directly into the container
# We clone into a temp dir or the current WORKDIR
RUN git clone https://github.com/kodi1046/PBTC5.git .

# 4. Python deps (pip install from the freshly cloned requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Dashboard port
EXPOSE 8080

# Environment setup
ENV DB_PATH="/data/paper_trades.db"
# Create the data directory so SQLite has a place to live
RUN mkdir -p /data

# Default to a Bash shell so you can use Claude Code immediately
CMD ["/bin/bash"]