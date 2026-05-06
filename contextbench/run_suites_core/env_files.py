
"""Small .env parser for explicit run-suite secret injection."""

from __future__ import annotations

from pathlib import Path

_SENSITIVE_ENV_NAME_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)


def is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SENSITIVE_ENV_NAME_PARTS)


def redact_secrets(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if is_sensitive_env_name(str(key)) else redact_secrets(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def read_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Configured env file not found: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env file line {path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env file line {path}:{line_number}: empty key")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values
