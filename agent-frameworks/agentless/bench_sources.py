import csv
import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

csv.field_size_limit(10 * 1024 * 1024)


LANG_DIR_MAP = {
    "javascript": "js",
    "typescript": "ts",
}

EXT_LANG_MAP = {
    ".py": "python",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
}

IGNORE_DIRS = {
    "node_modules",
    "dist",
    "build",
    "target",
    "vendor",
    ".git",
    "__pycache__",
    "venv",
    ".venv",
}


def map_language_dir(language: str) -> str:
    return LANG_DIR_MAP.get(language.lower(), language.lower())


def load_task_ids(bench_name: str, limit: int | None) -> list[str]:
    csv_path = DATA_DIR / f"{bench_name}.csv"
    tasks = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_inst_id = (row.get("original_inst_id") or "").strip()
            if not original_inst_id:
                continue
            tasks.append(original_inst_id)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def _load_records_from_csv(csv_path: Path, id_field: str, target_ids: set[str]) -> dict[str, dict]:
    records = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = (row.get(id_field) or "").strip()
            if not instance_id or instance_id not in target_ids:
                continue
            records[instance_id] = row
            if len(records) >= len(target_ids):
                break
    return records


def _load_verified_records(parquet_path: Path, target_ids: set[str]) -> dict[str, dict]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pq is None and pd is None:
        raise RuntimeError(
            "pyarrow or pandas is required to read Verified parquet. "
            "Install one of: `uv pip install pyarrow` or `uv pip install pandas`."
        )

    if pq is not None:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
    else:
        df = pd.read_parquet(parquet_path)

    df = df[df["instance_id"].isin(target_ids)]
    records = {row["instance_id"]: row.to_dict() for _, row in df.iterrows()}
    return records


def fetch_records(bench_name: str, target_ids: list[str]) -> dict[str, dict]:
    target_set = set(target_ids)
    if bench_name == "Poly":
        return _load_records_from_csv(DATA_DIR / "Poly/test.csv", "instance_id", target_set)
    if bench_name == "Pro":
        return _load_records_from_csv(DATA_DIR / "Pro/test.csv", "instance_id", target_set)
    if bench_name == "Verified":
        return _load_verified_records(
            DATA_DIR / "Verified/data/test-00000-of-00001.parquet", target_set
        )
    if bench_name == "Multi":
        # Multi uses JSONL library in data/Multi/{lang_dir}
        return _load_records_from_multi(target_set)
    raise ValueError(f"Unknown bench_name: {bench_name}")


def _load_records_from_multi(target_set: set[str]) -> dict[str, dict]:
    records = {}
    data_root = DATA_DIR / "Multi"
    for lang_dir in data_root.iterdir():
        if not lang_dir.is_dir():
            continue
        for jsonl in lang_dir.glob("*.jsonl"):
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                org = raw.get("org")
                repo = raw.get("repo")
                number = raw.get("number")
                if org and repo and number is not None:
                    instance_id = f"{org}__{repo}-{number}"
                else:
                    continue
                if instance_id in target_set:
                    raw["instance_id"] = instance_id
                    raw["_language_dir"] = lang_dir.name
                    records[instance_id] = raw
                if len(records) >= len(target_set):
                    return records
    return records


def infer_language_from_repo(repo: str, base_commit: str) -> str | None:
    repo_slug = repo.split("/")[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / f"{repo_slug}_{uuid.uuid4().hex}"
        os.system(f"git clone https://github.com/{repo}.git {repo_dir} > /dev/null 2>&1")
        os.system(f"git -C {repo_dir} checkout {base_commit} > /dev/null 2>&1")

        counts = Counter()
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for name in files:
                if name.endswith(".min.js"):
                    continue
                ext = Path(name).suffix.lower()
                if ext not in EXT_LANG_MAP:
                    continue
                lang = EXT_LANG_MAP[ext]
                counts[lang] += 1

        if not counts:
            return None
        total = sum(counts.values())
        lang, count = counts.most_common(1)[0]
        if total == 0 or count / total < 0.5:
            return None
        return lang


def resolve_language(record: dict, bench_name: str) -> str | None:
    repo = record.get("repo")
    base_commit = record.get("base_commit")
    if repo and base_commit:
        inferred = infer_language_from_repo(repo, base_commit)
        if inferred:
            return inferred

    if bench_name == "Poly":
        lang = (record.get("language") or "").lower()
        return None if lang == "mixed" or not lang else lang
    if bench_name == "Pro":
        lang = (record.get("repo_language") or "").lower()
        return lang or None
    if bench_name == "Verified":
        lang = (record.get("language") or "").lower()
        return lang or "python"
    if bench_name == "Multi":
        lang_dir = record.get("_language_dir")
        if lang_dir:
            for key, value in LANG_DIR_MAP.items():
                if value == lang_dir:
                    return key
            return lang_dir
    return None


def _split_problem_statement(problem_statement: str) -> tuple[str, str]:
    if not problem_statement:
        return "", ""
    parts = problem_statement.splitlines()
    title = parts[0].strip() if parts else ""
    body = "\n".join(parts[1:]).strip() if len(parts) > 1 else ""
    return title, body


def to_agentless_raw_json(record: dict, instance_id: str) -> dict:
    repo = record.get("repo")
    base_commit = record.get("base_commit")
    if not base_commit:
        base_commit = (record.get("base") or {}).get("sha")

    resolved_issues = record.get("resolved_issues")
    if not resolved_issues:
        problem_statement = record.get("problem_statement") or ""
        title, body = _split_problem_statement(problem_statement)
        resolved_issues = [{"title": title, "body": body}]
    org = record.get("org")
    repo_name = None
    if repo and "/" in repo:
        org, repo_name = repo.split("/", 1)
    elif repo:
        repo_name = repo
    if not org or not repo_name:
        raise ValueError(f"Invalid repo fields for instance {instance_id}: org={org}, repo={repo}")
    return {
        "org": org,
        "repo": repo_name,
        "instance_id": instance_id,
        "base": {"sha": base_commit},
        "resolved_issues": resolved_issues,
    }
