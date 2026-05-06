import json
import os
from pathlib import Path

from agentless.multilang.const import LANGUAGE, LANG_EXT


def process(raw_data):
    raw = raw_data if isinstance(raw_data, dict) else json.loads(raw_data)
    if "instance_id" not in raw:
        org = raw.get("org")
        repo = raw.get("repo")
        number = raw.get("number")
        if org and repo and number is not None:
            raw["instance_id"] = f"{org}__{repo}-{number}"
    data = {
        'repo': f'{raw["org"]}/{raw["repo"]}',
        'instance_id': raw['instance_id'],
        'base_commit': raw['base']['sha'],
        'problem_statement': raw['resolved_issues'][0]['title'] + '\n' + raw['resolved_issues'][0]['body'],
    }
    return data


def load_local_json():
    dataset = []
    data_root = os.environ.get("DATA_ROOT")
    bench_name = os.environ.get("BENCH_NAME")
    target_id = os.environ.get("TARGET_ID")
    if LANGUAGE == 'javascript':
        lang = 'js'
    elif LANGUAGE == 'typescript':
        lang = 'ts'
    else:
        lang = LANGUAGE
    if data_root:
        root_path = Path(data_root)
    elif bench_name:
        root_path = Path("data") / bench_name
    else:
        root_path = Path("data")
    path = root_path / lang
    for file in path.iterdir():
        if not file.is_file():
            continue
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if target_id and raw.get("instance_id") != target_id:
                continue
            dataset.append(process(raw))
    return dataset


def end_with_ext(file_name):
    for ext in LANG_EXT:
        if file_name.endswith(f'.{ext}'):
            return True
    return False
