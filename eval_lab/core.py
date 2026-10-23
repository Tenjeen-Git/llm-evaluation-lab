"""Dataset validation, transparent scoring and experiment aggregation."""
import hashlib
import json
import re
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("Knowledge", "Reasoning", "Truthfulness", "Code generation", "Repository reasoning")


def load_dataset(path=None):
    items = json.loads(Path(path or ROOT / "data/tasks.json").read_text(encoding="utf-8"))
    return validate_dataset(items)


def validate_dataset(items):
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a non-empty list.")
    if len(items) > 100:
        raise ValueError("This student app supports at most 100 tasks per dataset.")
    ids = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every task must be a JSON object.")
        for key in ("id", "category", "question", "reference", "rubric"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Missing or invalid field: {key}")
        if item["id"] in ids:
            raise ValueError("Duplicate task ID.")
        ids.add(item["id"])
        if item["category"] not in CATEGORIES:
            raise ValueError("Unknown category.")
        if "options" in item:
            if not isinstance(item["options"], dict) or set(item["options"]) != set("ABCD") or item["reference"] not in "ABCD" or len(item["reference"]) != 1:
                raise ValueError("MCQ requires A-D options and one reference letter.")
            if any(not isinstance(v, str) or not v.strip() for v in item["options"].values()):
                raise ValueError("Every option needs non-empty text.")
        if sum(len(str(v)) for v in item.values()) > 30000:
            raise ValueError("A task is too large (30,000 character limit).")
        if "tests" in item:
            if item["category"] != "Code generation" or not isinstance(item["tests"], list) or not item["tests"]:
                raise ValueError("Tests are supported only for code generation tasks.")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", item.get("entry_point", "")):
                raise ValueError("Code tasks require a Python entry_point identifier.")
            for test in item["tests"]:
                if not isinstance(test, dict) or not isinstance(test.get("args"), list) or "expected" not in test:
                    raise ValueError("Each test needs args (a list) and expected.")
    return items


def dataset_hash(items):
    encoded = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def model_inputs(task):
    """Never expose references or grading rubrics to the answering model."""
    options = task.get("options", {})
    question = task["question"]
    if options:
        question += "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        question += "\nReturn only one option letter: A, B, C, or D."
    return {"question": question}


def extract_option(answer):
    answer = answer.strip()
    if re.fullmatch(r"[A-Da-d][.)]?", answer):
        return answer[0].upper()
    matches = re.findall(r"(?:final\s+answer|answer|option)\s*[:=]\s*\(?([A-D])\)?\b", answer, re.I)
    return matches[-1].upper() if matches else None


def score_answer(task, answer):
    if "options" not in task:
        return {"correctness": None, "format_compliance": None,
                "grading": "Human review required", "grading_reason": "Open-ended task; no automatic correctness claim."}
    return {"correctness": int(extract_option(answer) == task["reference"]),
            "format_compliance": int(bool(re.fullmatch(r"[A-D]", answer.strip()))),
            "grading": "Reference option match", "grading_reason": "Correctness and strict letter-only formatting are scored separately."}


def summary(rows):
    """Failed calls remain in the MCQ denominator; ungraded text is never zero-filled."""
    mcq = [r for r in rows if r.get("is_mcq")]
    judged = [r for r in rows if r.get("judge_score") is not None]
    code = [r for r in rows if r.get("code_score") is not None]
    return {"examples": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
            "mcq_accuracy": mean(r["correctness"] or 0 for r in mcq) if mcq else None,
            "mcq_count": len(mcq),
            "format_compliance": mean(r["format_compliance"] or 0 for r in mcq) if mcq else None,
            "judge_score": mean(r["judge_score"] for r in judged) if judged else None,
            "judge_count": len(judged),
            "code_pass_rate": mean(r["code_score"] for r in code) if code else None,
            "code_count": len(code),
            "mean_latency_ms": mean(r["latency_ms"] for r in rows) if rows else 0,
            "tokens": sum(r.get("total_tokens", 0) or 0 for r in rows)}


def wilson_interval(successes, count):
    if count == 0:
        return None
    z = 1.96
    p = successes / count
    denominator = 1 + z * z / count
    midpoint = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0, midpoint - radius), min(1, midpoint + radius)


def compare_experiments(left, right):
    if left["dataset_hash"] != right["dataset_hash"] or left["mode"] != right["mode"]:
        raise ValueError("Compare the same dataset and execution mode only.")
    a = {r["task_id"]: r for r in left["rows"]}
    b = {r["task_id"]: r for r in right["rows"]}
    if set(a) != set(b):
        raise ValueError("Both experiments must contain the same tasks.")
    paired = []
    improved = regressed = 0
    for task_id, old in a.items():
        new = b[task_id]
        status = "Needs review"
        if old["is_mcq"]:
            delta = (new["correctness"] or 0) - (old["correctness"] or 0)
            status = "Improved" if delta > 0 else "Regressed" if delta < 0 else "Unchanged"
            improved += delta > 0
            regressed += delta < 0
        paired.append({"task_id": task_id, "category": old["category"], "change": status,
                       "answer_A": old["answer"], "answer_B": new["answer"],
                       "correct_A": old["correctness"], "correct_B": new["correctness"],
                       "latency_A_ms": old["latency_ms"], "latency_B_ms": new["latency_ms"]})
    discordant = improved + regressed
    p = min(1, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2 ** discordant) if discordant else 1.0
    return {"rows": paired, "improved": improved, "regressed": regressed, "mcnemar_exact_p": p}
