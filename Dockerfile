# Runs the draft-replies bridge on a Docker host (see nas/).
#
# The image contains CODE ONLY:
# - model weights: mounted read-only into the llama.cpp container (never in
#   git or an image)
# - SQLite DB: on the /data volume so it survives rebuilds
# - Gmail OAuth credentials + facts/allowlist: mounted from /config,
#   host-local, never baked in
#
# Generation happens on the llama.cpp server container (see nas/); this
# image runs the bridge one-shot with --backend remote, invoked by cron or
# a systemd timer.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV AUTOREPLY_DB=/data/autoreply.db
VOLUME ["/data"]

# One-shot: scheduling lives outside the image (nas/crontab.example or
# nas/systemd/).
CMD ["uv", "run", "draft-replies", "--backend", "remote", \
     "--facts", "/config/facts.yaml", "--allowlist", "/config/allowlist.yaml"]
