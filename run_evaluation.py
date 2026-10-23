"""Command-line alternative to the Streamlit interface."""
import argparse
import json
from dotenv import load_dotenv
from eval_lab.core import ROOT, load_dataset, summary
from eval_lab.runner import PROMPTS, run_experiment, save_experiment
from eval_lab.storage import RESULTS
from eval_lab.langsmith_sync import publish_experiment


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run benchmark-inspired LLM evaluations.")
    parser.add_argument("--provider", choices=["Demo", "OpenAI", "Ollama"], default="Demo")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="baseline-v1")
    parser.add_argument("--dataset", help="JSON dataset path; demo supports bundled tasks only")
    parser.add_argument("--judge", action="store_true", help="Add live LLM rubric grading")
    parser.add_argument("--code-tests", action="store_true", help="Run teaching tests in Docker")
    parser.add_argument("--langsmith", action="store_true", help="Upload recorded live outputs to LangSmith")
    args = parser.parse_args()
    if args.langsmith and args.provider == "Demo":
        parser.error("Scripted demo outputs cannot be published as live results.")
    tasks = load_dataset(args.dataset)
    experiment = run_experiment(tasks, args.provider, args.model, args.prompt, args.judge, run_code=args.code_tests)
    target = save_experiment(experiment, RESULTS)
    print(json.dumps({"mode": experiment["mode"], "file": str(target), "summary": summary(experiment["rows"])}, indent=2))
    if args.langsmith:
        result = publish_experiment(experiment, tasks)
        print("LangSmith experiment:", result.experiment_name)


if __name__ == "__main__":
    main()
