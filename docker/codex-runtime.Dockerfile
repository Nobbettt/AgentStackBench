
FROM node:22.16.0-bookworm-slim

ARG CODEX_CLI_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        passwd \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        sudo \
        tar \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "$CODEX_CLI_VERSION" \
    && npm install -g @openai/codex@${CODEX_CLI_VERSION}

WORKDIR /workspace
