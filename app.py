"""Run with: python -m streamlit run app.py"""
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from eval_lab.core import (ROOT, CATEGORIES, compare_experiments, dataset_hash,
                           load_dataset, summary, validate_dataset, wilson_interval)
from eval_lab.langsmith_sync import publish_experiment
from eval_lab.reports import csv_export, html_report
from eval_lab.runner import PROMPTS, run_experiment, save_experiment
from eval_lab.storage import RESULTS, history, human_agreement, load_reviews, save_review

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="LLM Evaluation Lab", page_icon="🔬", layout="wide")
st.markdown('''<style>
.stApp{background:#f5f7fc}h1,h2,h3{color:#132d51}.block-container{padding-top:2rem;max-width:1450px}
[data-testid="stMetric"]{background:white;border:1px solid #e0e7f1;border-radius:14px;padding:16px}
[data-testid="stSidebar"]{background:#eaf0f9}.lab-eyebrow{color:#4672ab;letter-spacing:2px;font-size:12px;font-weight:700}
.lab-hero{background:linear-gradient(115deg,#102746,#224f7b);padding:26px 30px;border-radius:18px;margin-bottom:22px;color:white}
.lab-hero h1{color:white;margin:0}.lab-hero p{color:#d9e6f8;margin-bottom:0}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="lab-hero"><span class="lab-eyebrow">LANGCHAIN + LANGSMITH</span>'
            '<h1>LLM Evaluation Lab</h1><p>Measure answers. Compare experiments. Understand failures.</p></div>',
            unsafe_allow_html=True)

if "tasks" not in st.session_state:
    st.session_state.tasks = load_dataset()
tasks = st.session_state.tasks
experiments = history()
labels = {e["id"]: f"{e['created_at'][:19]} | {e['model']} | {e['prompt_version']} | {e['mode']} | {e['id'][:6]}" for e in experiments}
by_id = {e["id"]: e for e in experiments}

with st.sidebar:
    st.subheader("Workspace")
    st.caption("Final-year project · Evaluation & observability")
    page = st.radio("Navigate", ["Dashboard", "Dataset Studio", "Run Experiments", "Compare Results", "Human Review", "LangSmith & Guide"])
    st.divider()
    st.metric("Dataset tasks", len(tasks))
    st.caption(f"Version: {dataset_hash(tasks)}")
    st.caption("Demo needs no keys. Live inference uses your configured provider.")


def pct(value):
    return "Not scored" if value is None else f"{value:.1%}"


def metrics(experiment):
    s = summary(experiment["rows"])
    columns = st.columns(4)
    columns[0].metric("MCQ accuracy", pct(s["mcq_accuracy"]), help=f"Denominator: {s['mcq_count']} MCQs; failed model calls count as incorrect.")
    columns[1].metric("Format compliance", pct(s["format_compliance"]))
    columns[2].metric("AI rubric judge", pct(s["judge_score"]), help=f"{s['judge_count']} open-ended answers scored. Not ground truth.")
    columns[3].metric("Code tests passed", pct(s["code_pass_rate"]), help=f"{s['code_count']} functions graded with container tests.")
    st.caption(f"{s['examples']} tasks · {s['errors']} generation errors · answer tokens reported: {s['tokens']}. "
               "Token counts exclude judge calls; missing usage is counted as zero. No dollar cost estimate is made.")
    if experiment["mode"] == "scripted_demo":
        st.warning("SCRIPTED DEMO: hand-authored responses, not live inference. Scores demonstrate the evaluators only; demo timing is not model latency.")
    else:
        st.caption(f"Mean generation latency: {s['mean_latency_ms']:.0f} ms. Judge latency is recorded separately.")
        if s["mcq_count"]:
            successes = sum(r["correctness"] or 0 for r in experiment["rows"] if r["is_mcq"])
            low, high = wilson_interval(successes, s["mcq_count"])
            st.caption(f"MCQ Wilson 95% interval: {low:.1%} to {high:.1%}. Small teaching samples cannot establish broad model ability.")


def exports(experiment):
    reviews = load_reviews(experiment["id"])
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download full JSON", json.dumps({"experiment": experiment, "human_reviews": reviews}, indent=2),
                       f"{experiment['id']}.json", "application/json")
    c2.download_button("Download result CSV", csv_export(experiment["rows"]), f"{experiment['id']}.csv", "text/csv")
    c3.download_button("Download HTML report", html_report(experiment, reviews), f"{experiment['id']}.html", "text/html")
    st.caption("Open the HTML report in a browser. Use Print → Save as PDF if needed.")


if page == "Dashboard":
    st.subheader("Experiment overview")
    st.write("Evaluate the same dataset across prompts or models, then inspect where behavior changes.")
    if not experiments:
        st.info("Start with the offline demonstration. It runs the scoring pipeline without calling an LLM.")
        if st.button("Create two demo experiments", type="primary"):
            for version in PROMPTS:
                save_experiment(run_experiment(load_dataset(), prompt_version=version), RESULTS)
            st.rerun()
    else:
        selected = st.selectbox("Inspect experiment", list(by_id), format_func=labels.get)
        experiment = by_id[selected]
        metrics(experiment)
        grouped = []
        for category in CATEGORIES:
            rows = [r for r in experiment["rows"] if r["category"] == category]
            if rows:
                s = summary(rows)
                grouped.append({"Category": category, "Tasks": len(rows), "MCQ accuracy": pct(s["mcq_accuracy"]),
                                "AI judge": pct(s["judge_score"]), "Errors": s["errors"]})
        st.dataframe(pd.DataFrame(grouped), hide_index=True, width="stretch")
        exports(experiment)
    st.caption("Five benchmark lenses from the submitted study: MMLU, GPQA, HumanEval, SWE-bench Verified, TruthfulQA. "
               "Included tasks are self-authored examples, not official benchmark datasets or leaderboard scores.")

elif page == "Dataset Studio":
    st.subheader("Dataset Studio")
    st.write("Inspect questions, reference answers and grading rubrics before running an experiment.")
    uploaded = st.file_uploader("Import a dataset (JSON or CSV; maximum 100 tasks)", type=["json", "csv"])
    if uploaded is not None and st.button("Validate and use uploaded dataset"):
        try:
            if uploaded.size > 2_000_000:
                raise ValueError("Maximum upload size is 2 MB.")
            if uploaded.name.lower().endswith(".json"):
                candidate = json.loads(uploaded.getvalue())
            else:
                frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), keep_default_na=False)
                candidate = frame.to_dict("records")
                for task in candidate:
                    for key in ("options", "tests"):
                        if task.get(key):
                            task[key] = json.loads(task[key])
                        else:
                            task.pop(key, None)
            st.session_state.tasks = validate_dataset(candidate)
            st.success("Dataset validated. Switch to Run Experiments.")
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(f"Dataset rejected: {exc}")
    if st.button("Restore bundled teaching dataset"):
        st.session_state.tasks = load_dataset()
        st.rerun()
    category = st.selectbox("Category", ["All"] + list(CATEGORIES))
    visible = [t for t in tasks if category == "All" or t["category"] == category]
    st.dataframe(pd.DataFrame([{k: t[k] for k in ("id", "category", "question")} for t in visible]), hide_index=True, width="stretch")
    if visible:
        detail = st.selectbox("Inspect task", [t["id"] for t in visible])
        st.json(next(t for t in visible if t["id"] == detail))
    st.download_button("Download dataset template", json.dumps(load_dataset(), indent=2), "tasks.json", "application/json")
    st.caption("References, rubrics and tests are withheld from the answering model. The judge receives reference answers and rubrics.")

elif page == "Run Experiments":
    st.subheader("Configure a comparison")
    provider = st.selectbox("Execution mode", ["Demo", "OpenAI", "Ollama"])
    default = "gpt-4o-mini" if provider == "OpenAI" else "llama3.2:3b" if provider == "Ollama" else "scripted-fixtures"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Experiment A**")
        model_a = st.text_input("Model A", default, disabled=provider == "Demo")
        prompt_a = st.selectbox("Prompt A", list(PROMPTS), index=0)
        with st.expander("View prompt A"):
            st.write(PROMPTS[prompt_a])
    with c2:
        st.markdown("**Experiment B**")
        model_b = st.text_input("Model B", default, disabled=provider == "Demo")
        prompt_b = st.selectbox("Prompt B", list(PROMPTS), index=1)
        with st.expander("View prompt B"):
            st.write(PROMPTS[prompt_b])
    categories = st.multiselect("Task categories", list(CATEGORIES), default=list(CATEGORIES))
    chosen = [t for t in tasks if t["category"] in categories]
    use_judge = st.checkbox("Grade open-ended answers with an LLM rubric judge", disabled=provider == "Demo")
    run_code = st.checkbox("Run code-generation tests in Docker (requires Docker Desktop and prepared image)")
    st.caption(f"Two experiments × {len(chosen)} tasks = {2*len(chosen)} answer calls in live mode. "
               "Enabling the judge adds one call per successful open-ended answer. Each experiment uses its own model as judge; "
               "judge scores across different judge models are not directly comparable. Use human review to validate them.")
    if provider == "Demo":
        st.info("Demo uses the bundled dataset and scripted answers. Uploads are for live runs; no API key is needed for demo.")
    elif provider == "OpenAI":
        st.info("Set OPENAI_API_KEY in .env. API access and billing are separate from ChatGPT subscriptions. "
                "You may enter another model ID available to your API account.")
    else:
        st.info("Install Ollama, start its local server, and pull the exact model tag entered above. Models run on your computer.")
    if st.button("Run both experiments", type="primary", disabled=not chosen):
        try:
            if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is missing. Add it to .env and restart the app.")
            if provider == "Demo":
                bundled = {t["id"]: t for t in load_dataset()}
                if any(bundled.get(t["id"]) != t for t in chosen):
                    raise ValueError("Restore the bundled dataset to use demo mode.")
            if model_a == model_b and prompt_a == prompt_b:
                st.warning("A and B have the same configuration. Differences may reflect nondeterminism, not a changed system.")
            for label, name, version in [("A", model_a, prompt_a), ("B", model_b, prompt_b)]:
                st.write(f"Running experiment {label}…")
                bar = st.progress(0.0)
                experiment = run_experiment(chosen, provider, name.strip(), version, use_judge, bar.progress, run_code=run_code)
                save_experiment(experiment, RESULTS)
                errors = summary(experiment["rows"])["errors"]
                if errors:
                    st.warning(f"Experiment {label} saved with {errors} failed calls. Inspect the error column in Compare Results. "
                               "For OpenAI check key, billing and model access; for Ollama check the server and model tag.")
            st.success("Saved. Open Compare Results to inspect the two experiments.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))

elif page == "Compare Results":
    st.subheader("Compare Results")
    if not experiments:
        st.info("Run an experiment first, or create the demo from Dashboard.")
    else:
        col_a, col_b = st.columns(2)
        left_id = col_a.selectbox("Experiment A", list(by_id), index=min(1, len(by_id)-1), format_func=labels.get)
        right_id = col_b.selectbox("Experiment B", list(by_id), format_func=labels.get)
        left, right = by_id[left_id], by_id[right_id]
        if left_id == right_id:
            st.info("Choose two different experiments for a meaningful comparison.")
        try:
            comparison = compare_experiments(left, right)
            st.caption("Changes below are B relative to A. Only reference-scored MCQs contribute to improvement/regression counts.")
            cols = st.columns(3)
            cols[0].metric("MCQs improved", comparison["improved"])
            cols[1].metric("MCQs regressed", comparison["regressed"])
            cols[2].metric("Paired exact test p", f"{comparison['mcnemar_exact_p']:.3f}" if left["mode"] == "live_model" else "Demo only")
            if left["mode"] == "live_model":
                st.caption("Exact McNemar test on discordant MCQ pairs; exploratory on this small, non-random teaching sample.")
            filter_change = st.selectbox("Filter changes", ["All", "Improved", "Regressed", "Unchanged", "Needs review"])
            rows = [r for r in comparison["rows"] if filter_change == "All" or r["change"] == filter_change]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.download_button("Download paired comparison", csv_export(comparison["rows"]), "comparison.csv", "text/csv")
        except ValueError as exc:
            st.warning(str(exc))
        st.divider()
        selected_id = st.selectbox("Inspect one experiment", list(by_id), format_func=labels.get)
        experiment = by_id[selected_id]
        metrics(experiment)
        task_id = st.selectbox("Inspect answer and grading", [r["task_id"] for r in experiment["rows"]])
        row = next(r for r in experiment["rows"] if r["task_id"] == task_id)
        st.write(row["question"])
        ca, cb = st.columns(2)
        ca.markdown("**Model answer**")
        ca.code(row["answer"], language="python" if row["category"] == "Code generation" else None)
        cb.markdown("**Reference answer**")
        cb.code(row["reference"], language="python" if row["category"] == "Code generation" else None)
        with st.expander("Grading, errors and local metadata"):
            st.json(row)
            st.write("Exact system prompt:")
            st.code(experiment["prompt_text"], language=None)
        exports(experiment)

elif page == "Human Review":
    st.subheader("Human review & judge calibration")
    if not experiments:
        st.info("Create an experiment first.")
    else:
        experiment_id = st.selectbox("Review experiment", list(by_id), format_func=labels.get)
        experiment = by_id[experiment_id]
        reviews = load_reviews(experiment_id)
        agreement = human_agreement(experiment, reviews)
        cols = st.columns(3)
        cols[0].metric("Human-reviewed answers", len(reviews))
        cols[1].metric("Human / AI agreement", pct(agreement["agreement"]))
        cols[2].metric("Cohen's kappa", "Not available" if agreement["kappa"] is None else f"{agreement['kappa']:.2f}")
        st.caption(f"Agreement uses {agreement['count']} answers with both a binary human score and an AI-judge score. "
                   "MCQ reference scores are not treated as AI-judge scores. Kappa is undefined when expected agreement is 1.")
        task_id = st.selectbox("Task to review", [r["task_id"] for r in experiment["rows"]])
        row = next(r for r in experiment["rows"] if r["task_id"] == task_id)
        st.write(row["question"])
        st.markdown("**Candidate answer**")
        st.code(row["answer"], language=None)
        with st.expander("Reference and automatic grading"):
            st.code(row["reference"], language=None)
            st.write({k: row.get(k) for k in ("correctness", "judge_score", "judge_reason", "code_score", "code_reason")})
        existing = reviews.get(task_id, {})
        with st.form("review"):
            reviewer = st.text_input("Reviewer label", value=existing.get("reviewer", "Student reviewer"))
            verdict = st.radio("Your assessment", ["Correct", "Incorrect", "Unsure"],
                               index={1: 0, 0: 1, None: 2}[existing.get("score")])
            note = st.text_area("Reason or correction", value=existing.get("note", ""))
            if st.form_submit_button("Save review"):
                save_review(experiment_id, task_id, {"score": {"Correct": 1, "Incorrect": 0, "Unsure": None}[verdict],
                            "note": note, "reviewer": reviewer, "created_at": datetime.now(timezone.utc).isoformat()})
                st.rerun()
        st.download_button("Download human reviews", json.dumps(reviews, indent=2), "human_reviews.json", "application/json")
        st.caption("Reviews persist on this computer. This is a local review workflow, not a LangSmith annotation queue or a multi-user service.")

else:
    st.subheader("LangSmith experiments & tracing")
    st.write("Set LANGSMITH_API_KEY in .env. Set LANGSMITH_TRACING=true before a live run to record the real LangChain model calls.")
    st.info("Publishing below sends questions, references, answers and evaluation scores to your LangSmith workspace. "
            "It replays existing outputs without making additional LLM calls. Replay duration is not inference latency; "
            "the original latency is stored as a metric. Publish only material you intend to upload.")
    live = [e for e in experiments if e["mode"] == "live_model"]
    if live:
        live_ids = [e["id"] for e in live]
        selected = st.selectbox("Live experiment to publish", live_ids, format_func=labels.get)
        experiment = by_id[selected]
        if st.button("Publish experiment to LangSmith"):
            try:
                # Each saved experiment includes enough task information in rows, but full rubrics must match.
                candidates = [experiment.get("dataset_snapshot", []), tasks, load_dataset()]
                matching = next((source for source in candidates if dataset_hash([t for t in source if t["id"] in experiment["task_ids"]]) == experiment["dataset_hash"]), None)
                if matching is None:
                    raise ValueError("Reload the dataset used for this experiment in Dataset Studio before publishing.")
                chosen = [t for t in matching if t["id"] in experiment["task_ids"]]
                with st.spinner("Publishing recorded outputs and scores…"):
                    result = publish_experiment(experiment, chosen)
                st.success(f"Published experiment: {result.experiment_name}")
                st.link_button("Open LangSmith", "https://smith.langchain.com")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"LangSmith request failed ({type(exc).__name__}). Check your key, workspace region and connection.")
    else:
        st.caption("A successful live-model experiment is required. Scripted demo runs cannot be published as live results.")
    st.markdown("""**How this follows the submitted report**

| Report concept | Working implementation |
|---|---|
| Dataset construction | Version-hashed JSON/CSV tasks with references and rubrics |
| MMLU / GPQA lenses | Multiple-choice knowledge and reasoning tasks with separate correctness and format scoring |
| HumanEval lens | Python function generation with optional Docker-isolated teaching tests |
| SWE-bench Verified lens | Toy issue contexts and proposed fixes, graded by a rubric or human; no full repository execution |
| TruthfulQA lens | Misconception questions, reference-based AI judging and human review |
| Experiment comparison | Paired outputs, improvements/regressions and live-mode uncertainty statistics |
| Tracing | Real LangChain call tracing when LangSmith tracing is enabled |
| Human calibration | Persistent local reviews, agreement and Cohen's kappa |

**Scope:** This application evaluates existing LLMs. It does not train model weights or reproduce official benchmark scores.
Online monitoring, Prompt Hub publishing and LangSmith annotation queues are extension work, not implemented features.
""")
