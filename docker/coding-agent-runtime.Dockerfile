
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        nodejs \
        npm \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        sudo \
        tar \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# This image intentionally does not install Codex CLI or Claude Code CLI.
# Install and authenticate the desired agent CLI in a derived image, or mount a
# preconfigured runtime home through ContextBench runtime settings.

WORKDIR /workspace
