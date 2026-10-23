# LLM Evaluation Lab

A runnable Python application for comparing LLM prompts/models with LangChain and LangSmith.
Created as a practical companion to a study titled **LangSmith Evaluation as an LLM Tool**.

**Windows users: start with [START_HERE_WINDOWS.md](START_HERE_WINDOWS.md).**

## What it does

- Inspect and import version-hashed question/reference datasets (JSON and CSV).
- Compare two prompt versions or model IDs on identical tasks.
- Score MCQ correctness separately from output formatting.
- Optionally grade open-ended answers with a structured LLM rubric judge.
- Optionally run generated Python functions against small test suites in Docker containers.
- Find per-example improvements and regressions instead of relying on an average alone.
- Track answer latency and reported token usage; no unsupported dollar-cost estimate.
- Persist human review and measure human/AI agreement, including Cohen's kappa.
- Export JSON, CSV and a printable HTML report.
- Record real model-call traces and publish recorded results as LangSmith experiments.
- Run a labelled scripted demonstration without keys or model downloads.

## Quick start

Python 3.12 recommended (3.11 supported).

```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.
The UI opens at http://localhost:8501. The default offline demonstration requires no configuration.
For live inference, copy `.env.example` to `.env`, add your own provider credentials if needed, and restart.

## Study-to-implementation mapping

| Study lens | Implemented task/evaluator | Boundary |
|---|---|---|
| MMLU | Knowledge MCQs, reference option scoring | Five teaching questions; not the official 57-subject dataset |
| GPQA | Reasoning MCQs; format separated from correctness | Five introductory reasoning tasks; not graduate expert GPQA questions |
| HumanEval | Python function generation; optional isolated unit tests | Five custom tasks and small teaching tests; not official HumanEval/pass@k |
| SWE-bench Verified | Toy issue contexts with proposed fixes; rubric/human grading | No real-repository checkout, patch application or SWE-bench resolution score |
| TruthfulQA | Misconception and evaluation-literacy questions; rubric/human grading | Five custom questions; not official TruthfulQA |

The project evaluates existing models. It does **not** train a language model or claim to reproduce official benchmark scores.
The advanced work is in evaluation design, provenance, comparison, tracing and human calibration.

## Screens

1. **Dashboard** - experiment metrics, category summary and exports.
2. **Dataset Studio** - dataset upload, schema validation and reference inspection.
3. **Run Experiments** - provider/model/prompt selection and optional graders.
4. **Compare Results** - paired answer differences, regression filters and per-answer grading details.
5. **Human Review** - persistent assessments, reasons and judge agreement.
6. **LangSmith & Guide** - publish successful recorded live outputs and review the report mapping.

## Technical structure

```text
app.py                     Streamlit interface
run_evaluation.py          CLI entry point
eval_lab/core.py           Validation, scoring, dataset hashes, statistics
eval_lab/runner.py         LangChain model/prompt pipeline and judge
eval_lab/code_sandbox.py   Optional Docker functional-test harness
eval_lab/langsmith_sync.py LangSmith dataset and experiment upload
eval_lab/storage.py        Local history and human annotations
eval_lab/reports.py        Escaped HTML and spreadsheet-safe CSV exports
data/tasks.json           25 self-authored teaching tasks and references
data/demo_answers.json    Explicitly scripted demonstration outputs
tests/                    Unit and Streamlit interaction tests
docs/                     Architecture, limitations, and demonstration script
```

## CLI examples

```bash
python run_evaluation.py --provider Demo --prompt baseline-v1
python run_evaluation.py --provider Demo --prompt careful-v2
python run_evaluation.py --provider OpenAI --model gpt-4o-mini --judge --langsmith
python run_evaluation.py --provider Ollama --model llama3.2:3b --prompt careful-v2
python run_evaluation.py --provider OpenAI --model gpt-4o-mini --code-tests
python -m pytest -q
```

Use the virtual environment's Python in these commands if it is not activated.

## Reproducibility and interpretation

- A dataset snapshot, hash, exact system prompt, model ID and creation time are saved per run.
- References, rubrics and test cases are withheld from the answering model; judges receive references and rubrics.
- Failed MCQ calls remain in the accuracy denominator. Unscored free-text answers are never silently scored as correct or incorrect.
- Live MCQ accuracy has a Wilson 95% interval; paired changes have an exploratory exact McNemar test.
- Demo runs contain hand-authored fixtures: their scores and timings describe the demonstration only.
- Provider model aliases, hosted service changes and stochastic inference can still prevent exact reproduction.
- Open-ended judge scores are a separate metric from deterministic MCQ correctness and executed code tests.
- Each live experiment uses its own model as judge. Different judge models introduce a comparison confound; use human review.
- Tokens include only generation calls with reported usage. Judge tokens are excluded; actual billing must be checked with the provider.
- LangSmith upload replays recorded outputs. Actual model-call traces require tracing enabled before inference.

## Scope and limitations

Local single-user application; not a public multi-tenant execution service. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
No production deployment, authentication, online evaluation rules, Prompt Hub integration, LangSmith annotation queue,
official benchmark ingestion, or full SWE-bench agent harness is included.
Do not describe these extension ideas as completed features.

## Credits

Implementation prepared with AI coding assistance for this project. It is not a fork of the previously discussed RAG repository.
LangChain, LangSmith, Streamlit, Ollama and the model providers are independent third-party projects/services.
Check your institution's rules for declaring AI assistance and understand the code before demonstrating it.

Primary implementation references:

- https://docs.langchain.com/langsmith/evaluate-llm-application
- https://docs.langchain.com/langsmith/manage-datasets-programmatically
- https://reference.langchain.com/python/langchain-openai/langchain_openai
- https://docs.langchain.com/oss/python/integrations/chat/ollama
