"""Local experiment history and separate, persistent human annotations."""
import json
import os
import re
import tempfile
from pathlib import Path
from .core import ROOT

RESULTS = ROOT / "results"


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="pending-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def review_path(experiment_id):
    if not re.fullmatch(r"[0-9a-f-]{36}", experiment_id):
        raise ValueError("Invalid experiment identifier.")
    return RESULTS / f"{experiment_id}.reviews.json"


def load_reviews(experiment_id):
    path = review_path(experiment_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_review(experiment_id, task_id, review):
    reviews = load_reviews(experiment_id)
    reviews[task_id] = review
    atomic_json(review_path(experiment_id), reviews)


def history():
    experiments = []
    for path in RESULTS.glob("*.json"):
        if path.name.endswith(".reviews.json"):
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if content.get("schema_version") == 1 and isinstance(content.get("rows"), list):
                experiments.append(content)
        except (OSError, ValueError):
            continue
    return sorted(experiments, key=lambda x: x["created_at"], reverse=True)


def human_agreement(experiment, reviews):
    pairs = [(row["judge_score"], reviews[row["task_id"]]["score"]) for row in experiment["rows"]
             if row.get("judge_score") is not None and reviews.get(row["task_id"], {}).get("score") in (0, 1)]
    if not pairs:
        return {"count": 0, "agreement": None, "kappa": None}
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    judge_p = sum(a for a, _ in pairs) / n
    human_p = sum(b for _, b in pairs) / n
    expected = judge_p * human_p + (1 - judge_p) * (1 - human_p)
    return {"count": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}
