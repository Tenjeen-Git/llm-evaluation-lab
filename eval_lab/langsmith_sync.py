"""Upload already-computed outputs as LangSmith experiments without new LLM calls."""
import os

from langsmith import Client
from langsmith.utils import LangSmithConflictError

from .core import dataset_hash, model_inputs


def publish_experiment(experiment, tasks, client=None):
    if experiment["mode"] != "live_model":
        raise ValueError("Scripted demo outputs must not be uploaded as live-model experiments.")
    if any(row["error"] for row in experiment["rows"]):
        raise ValueError("Resolve model errors before publishing this experiment.")
    if experiment["dataset_hash"] != dataset_hash(tasks):
        raise ValueError("Dataset does not match the experiment.")
    if client is None:
        if not os.getenv("LANGSMITH_API_KEY"):
            raise ValueError("Set LANGSMITH_API_KEY in .env first.")
        client = Client()
    name = f"evaluation-lab-{dataset_hash(tasks)}"
    try:
        dataset = client.create_dataset(dataset_name=name, description="Self-authored teaching tasks, not official benchmarks.")
    except LangSmithConflictError:
        dataset = client.read_dataset(dataset_name=name)
    # Stable per-task keys make a retry repair partial uploads without duplicating examples.
    existing = {e.inputs.get("task_id") for e in client.list_examples(dataset_id=dataset.id)}
    missing = [t for t in tasks if t["id"] not in existing]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=[{
            "inputs": {"task_id": t["id"], "question": model_inputs(t)["question"], "category": t["category"]},
            "outputs": {"answer": t["reference"], "is_mcq": "options" in t},
            "metadata": {"rubric": t["rubric"], "source": "Self-authored teaching task"}
        } for t in missing])
    by_id = {row["task_id"]: row for row in experiment["rows"]}

    def target(inputs):
        row = by_id[inputs["task_id"]]
        return {"answer": row["answer"], "correctness": row["correctness"],
                "format_compliance": row["format_compliance"], "judge_score": row["judge_score"],
                "code_score": row.get("code_score"),
                "generation_latency_ms": row["latency_ms"], "total_tokens": row["total_tokens"]}

    def evaluator(outputs, reference_outputs):
        scores = [{"key": "generation_latency_ms", "score": outputs["generation_latency_ms"]}]
        if reference_outputs["is_mcq"]:
            scores.extend([{"key": "mcq_correctness", "score": outputs["correctness"]},
                           {"key": "format_compliance", "score": outputs["format_compliance"]}])
        if outputs["judge_score"] is not None:
            scores.append({"key": "llm_rubric_judge", "score": outputs["judge_score"]})
        if outputs["code_score"] is not None:
            scores.append({"key": "teaching_code_tests_passed", "score": outputs["code_score"]})
        return {"results": scores}

    return client.evaluate(target, data=client.list_examples(dataset_id=dataset.id),
                           evaluators=[evaluator], experiment_prefix=experiment["prompt_version"],
                           metadata={"local_experiment_id": experiment["id"], "model": experiment["model"],
                                     "prompt_text": experiment["prompt_text"], "dataset_hash": experiment["dataset_hash"],
                                     "output_source": "Previously generated local outputs; replay is not inference latency"},
                           max_concurrency=1)
