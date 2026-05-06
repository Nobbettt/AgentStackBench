FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/contextbench

COPY pyproject.toml README.md LICENSE requirements.txt /opt/contextbench/
COPY contextbench /opt/contextbench/contextbench

RUN python -m pip install --upgrade pip==26.0.1 \
    && python -m pip install -r /opt/contextbench/requirements.txt \
    && python -m pip install /opt/contextbench

ENTRYPOINT ["python"]
