"""Portable HTML reports, printable from a browser, and CSV result exports."""
import csv
import io
from html import escape
from .core import summary


def csv_export(rows):
    buffer = io.StringIO(newline="")
    if not rows:
        return ""
    columns = sorted(set().union(*(row.keys() for row in rows)))
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        # Spreadsheet apps must display generated text rather than interpret formulas.
        safe = {k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v)
                for k, v in row.items()}
        writer.writerow(safe)
    return buffer.getvalue()


def html_report(experiment, reviews=None):
    reviews = reviews or {}
    s = summary(experiment["rows"])
    def percent(value):
        return "Not scored" if value is None else f"{value:.1%}"
    cards = "".join(f"<div><strong>{escape(label)}</strong><p>{escape(value)}</p></div>" for label, value in [
        ("MCQ accuracy", percent(s["mcq_accuracy"])), ("MCQ examples", str(s["mcq_count"])),
        ("LLM judge", percent(s["judge_score"])), ("Code pass rate", percent(s["code_pass_rate"]))])
    body = []
    for row in experiment["rows"]:
        review = reviews.get(row["task_id"], {})
        body.append(f"<article><h2>{escape(row['task_id'])} - {escape(row['category'])}</h2>"
                    f"<h3>Question</h3><pre>{escape(row['question'])}</pre>"
                    f"<h3>Model answer</h3><pre>{escape(row['answer'])}</pre>"
                    f"<h3>Reference</h3><pre>{escape(row['reference'])}</pre>"
                    f"<p>MCQ correctness: {row['correctness']} | Judge: {row['judge_score']} | "
                    f"Code tests: {row.get('code_score')} | Error: {escape(str(row['error']))}</p>"
                    f"<p>Human score: {escape(str(review.get('score', 'Not reviewed')))}. "
                    f"{escape(review.get('note', ''))}</p></article>")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>LLM Evaluation Report</title>
<style>body{{font:15px/1.6 system-ui;max-width:1050px;margin:45px auto;padding:0 25px;color:#17233a}}
h1{{color:#143e77}}.cards{{display:flex;gap:25px;flex-wrap:wrap}}.cards>div{{background:#eef5ff;padding:16px;min-width:150px}}
article{{border-top:1px solid #ccd7e5;margin-top:26px;page-break-inside:avoid}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}}
@media print{{body{{margin:0}}}}</style><h1>LLM Evaluation Lab</h1>
<p>Mode: <strong>{escape(experiment['mode'])}</strong> | Model: {escape(experiment['model'])} |
Prompt: {escape(experiment['prompt_version'])}</p><p>Dataset: {escape(experiment['dataset_hash'])} |
Created: {escape(experiment['created_at'])}</p><div class="cards">{cards}</div>
<p>These are self-authored teaching tasks, not official MMLU, GPQA, HumanEval, SWE-bench or TruthfulQA results.
Scripted demo outputs are not evidence of model quality. AI judge scores need human validation.
Coding results cover only the supplied tests. Repository tasks do not execute real repositories.</p>
{''.join(body)}</html>'''
