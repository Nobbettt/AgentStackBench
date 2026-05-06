"""Extract trajectory from MiniSWE-agent `.traj.json` format.

Strict extraction policy (by design):
- Steps: extract only from `<explore_context> ... </explore_context>` blocks.
- Final: extract only from `<PATCH_CONTEXT> ... </PATCH_CONTEXT>` blocks.

Fallback policy (only for steps):
- If a trajectory contains no `<explore_context>` blocks, we fall back to parsing bash commands
  to reconstruct step-level *file* retrievals (and explicit ranged views when possible).
- This fallback never affects final context: final is still `<PATCH_CONTEXT>` only.
"""

import json
import re
from typing import Dict, List, Any


def extract_trajectory(traj_file: str) -> Dict[str, Any]:
    """Extract trajectory steps and final context from a MiniSWE `.traj.json` file."""
    with open(traj_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    steps: List[Dict[str, Any]] = []

    # Heuristic: in Multi-SWE-bench docker images, repos are usually under /home/<repo_dir>/...
    # We normalize such absolute paths to repo-relative paths when the repo dir is known.
    repo_dir_name = _infer_repo_dir_name(data)

    # --- steps: <explore_context> blocks (preferred) ---
    for msg in data.get("messages", []):
        content = msg.get("content", "") or ""
        for block in _extract_tag_blocks(content, tag="explore_context"):
            spans_by_file = _parse_file_lines_pairs(block, repo_dir_name=repo_dir_name)
            if not spans_by_file:
                continue
            files = sorted(spans_by_file.keys())
            steps.append({"files": files, "spans": spans_by_file})

    # --- steps fallback: bash command parsing (only if no explore_context exists) ---
    if not steps:
        for msg in data.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "") or ""
            # Be tolerant to different newline conventions and extra whitespace.
            match = re.search(r"```bash\s*\r?\n([\s\S]*?)\r?\n```", content)
            if not match:
                continue
            cmd = match.group(1).strip()
            if "COMPLETE_TASK" in cmd:
                continue
            views = _extract_views_from_command(cmd)
            if not views:
                continue

            step_data = {"files": [], "spans": {}}
            for view in views:
                file_path = _normalize_path(view.get("file", ""), repo_dir_name=repo_dir_name)
                if not file_path:
                    continue
                step_data["files"].append(file_path)

                # Safety: only record spans when an explicit line range is present.
                # This avoids accidental "whole file" spans causing symbol explosions.
                if "start_line" in view and "end_line" in view:
                    step_data["spans"].setdefault(file_path, []).append(
                        {"type": "line", "start": int(view["start_line"]), "end": int(view["end_line"])}
                    )

            # De-dup files for this step, keep deterministic order
            if step_data["files"]:
                step_data["files"] = sorted(set(step_data["files"]))
                steps.append(step_data)

    # --- final: last <PATCH_CONTEXT> block (strict) ---
    patch_blocks: List[str] = []
    for msg in data.get("messages", []):
        content = msg.get("content", "") or ""
        patch_blocks.extend(_extract_tag_blocks(content, tag="PATCH_CONTEXT"))

    final_files: List[str] = []
    final_spans: Dict[str, List[Dict[str, int]]] = {}
    if patch_blocks:
        final_spans = _parse_file_lines_pairs(patch_blocks[-1], repo_dir_name=repo_dir_name)
        final_files = sorted(final_spans.keys())

    return {"pred_steps": steps, "pred_files": final_files, "pred_spans": final_spans}


def _infer_repo_dir_name(data: Dict[str, Any]) -> str:
    """Infer the repo directory name under /home from the MiniSWE trajectory metadata."""
    try:
        image = data.get("info", {}).get("config", {}).get("environment", {}).get("image", "") or ""
        # Example: mswebench/alibaba_m_fastjson2:pr-2559 -> fastjson2
        m = re.search(r"_m_([A-Za-z0-9_.-]+)(?::|$)", image)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


def _normalize_path(path: str, repo_dir_name: str = "") -> str:
    """Normalize container paths to *candidate* repo-relative paths.

    Note: final validation is performed later in evaluation by resolving path suffixes
    against the checked-out repo worktree. Here we avoid hard-coded prefix rules as much
    as possible and do not drop unknown absolute prefixes.
    """
    p = (path or "").strip().strip("'\"")
    if not p:
        return ""

    # Common Multi-SWE-bench prefix.
    if p.startswith("/testbed/"):
        p = p[len("/testbed/"):]

    # Common docker working directory prefix.
    if p.startswith("/home/"):
        rest = p[len("/home/"):]
        parts = rest.split("/", 1)
        if len(parts) == 2:
            first, tail = parts[0], parts[1]
            # If /home/<repo>/..., optionally drop the repo dir when known.
            # Otherwise, keep the full suffix and let evaluation-time suffix resolution decide.
            p = tail if (repo_dir_name and first == repo_dir_name) else rest
        else:
            # /home/<something> with no subpath: not a source file
            return ""

    # For any other absolute prefix, keep it as a candidate by stripping the leading '/'.
    # Evaluation-time suffix resolution will drop it if it does not map into the repo.
    if p.startswith("/"):
        p = p.lstrip("/")

    # Normalize leading "./"
    if p.startswith("./"):
        p = p[2:]

    return p


def _extract_tag_blocks(text: str, tag: str) -> List[str]:
    """Return list of inner text payloads for <tag>...</tag> blocks (case-sensitive)."""
    if not text:
        return []
    # Support both lowercase and uppercase tag names as they appear in different traj templates.
    # Example: <explore_context>...</explore_context> and <PATCH_CONTEXT>...</PATCH_CONTEXT>
    pattern = rf"<{re.escape(tag)}>\s*([\s\S]*?)\s*</{re.escape(tag)}>"
    return [m.group(1) for m in re.finditer(pattern, text)]


def _parse_file_lines_pairs(text: str, repo_dir_name: str = "") -> Dict[str, List[Dict[str, int]]]:
    """Parse blocks containing repeated pairs:

    File: /path/to/file
    Lines: start-end
    """
    result: Dict[str, List[Dict[str, int]]] = {}
    current_file: str = ""

    for raw in (text or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        if line.startswith("File:"):
            f = line[len("File:") :].strip()
            f = _normalize_path(f, repo_dir_name=repo_dir_name)
            current_file = f or ""
            continue
        if line.startswith("Lines:") and current_file:
            m = re.match(r"(\d+)\s*-\s*(\d+)", line[len("Lines:") :].strip())
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            result.setdefault(current_file, []).append({"type": "line", "start": a, "end": b})

    return result


def _extract_views_from_command(cmd: str) -> List[Dict[str, int | str]]:
    """Extract file viewing operations from a bash command.

    Used only as a fallback when `<explore_context>` is missing.
    We intentionally return line ranges only when explicit.
    """
    if not cmd:
        return []
    if any(p in cmd for p in ["sed -i", "echo ", "mkdir", "rm ", "git add", "git commit"]):
        return []

    views: List[Dict[str, int | str]] = []

    # Split composite commands so we can extract multiple views from one bash block.
    chunks = re.split(r"\s*(?:&&|\|\||;)\s*", cmd)
    for chunk in chunks:
        c = (chunk or "").strip()
        if not c:
            continue

        # nl -ba file | sed -n 'start,endp'
        m = re.search(r"nl\s+[^|]+\s+([^\s|]+)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p", c)
        if m:
            f = m.group(1).strip("'\"")
            if f.startswith("/testbed/"):
                f = f[len("/testbed/") :]
            if _is_source_file(f):
                views.append({"file": f, "start_line": int(m.group(2)), "end_line": int(m.group(3))})
            continue

        # sed -n 'start,endp' file
        m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s&|>;<]+)", c)
        if m:
            f = m.group(3).strip("'\"")
            if f.startswith("/testbed/"):
                f = f[len("/testbed/") :]
            if _is_source_file(f):
                views.append({"file": f, "start_line": int(m.group(1)), "end_line": int(m.group(2))})
            continue

        # head -n N file
        m = re.search(r"\bhead\s+-n\s+(\d+)\s+([^\s&|>]+)", c)
        if m:
            f = m.group(2).strip("'\"")
            if f.startswith("/testbed/"):
                f = f[len("/testbed/") :]
            if _is_source_file(f):
                views.append({"file": f, "start_line": 1, "end_line": int(m.group(1))})
            continue

        # cat file (file-only)
        m = re.search(r"\bcat\s+([^\s&|>]+)", c)
        if m:
            f = m.group(1).strip("'\"")
            if f.startswith("/testbed/"):
                f = f[len("/testbed/") :]
            if _is_source_file(f):
                views.append({"file": f})
            continue

        # grep ... file (file-only)
        m = re.search(
            r"\bgrep\s+.*?\s+([^\s&|>]+\.(?:py|js|java|go|rs|c|cpp|h|hpp|ts|tsx|jsx|rb|php|cs|kt|scala|swift))\b",
            c,
        )
        if m:
            f = m.group(1).strip("'\"")
            if f.startswith("/testbed/"):
                f = f[len("/testbed/") :]
            if _is_source_file(f):
                views.append({"file": f})
            continue

    # De-dup while preserving order
    seen = set()
    out: List[Dict[str, int | str]] = []
    for v in views:
        key = (v.get("file"), v.get("start_line"), v.get("end_line"))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _is_source_file(path: str) -> bool:
    """Check if path looks like source file."""
    exts = [
        ".py",
        ".js",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".ts",
        ".tsx",
        ".jsx",
        ".rb",
        ".php",
        ".cs",
        ".kt",
        ".scala",
        ".swift",
    ]
    return any((path or "").endswith(e) for e in exts)

