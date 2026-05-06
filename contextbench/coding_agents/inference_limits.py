
"""Centralized static limits for conservative trace inference."""

from __future__ import annotations

# Ignore only very large command outputs for trace inference. Keep this
# relatively generous so broad-but-legitimate searches are still represented.
MAX_COMMAND_OUTPUT_CHARS = 120_000

# Bound the number of grep/file-list matches we turn into inferred retrieval
# context so a single broad search does not dominate trajectory extraction,
# while still preserving a substantial amount of exploratory context.
MAX_GREP_SPAN_MATCHES = 1_500
MAX_FILE_LIST_MATCHES = 1_500

# Skip absurdly long lines when interpreting grep/path-like outputs.
MAX_GREP_LINE_CHARS = 4_096
MAX_PLAIN_PATH_LINE_CHARS = 2_048

# Avoid expensive shell tokenization on very large or heavily quoted commands.
MAX_COMMAND_TOKENIZATION_CHARS = 4_096
