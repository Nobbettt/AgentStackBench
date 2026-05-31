FROM node:22-trixie-slim@sha256:55931e785da6feb47a9ed9ee54093b7710f3cbab9962708e5c4c9b5318c66451

ARG CLAUDE_CODE_VERSION
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

RUN test -n "$CLAUDE_CODE_VERSION" \
    && npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

WORKDIR /workspace
