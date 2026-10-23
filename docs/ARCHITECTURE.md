# Architecture and evaluation design

## Data flow

Dataset Studio validates a JSON/CSV dataset and computes a content hash. The runner creates a LangChain prompt/model pipeline
for each experiment. Only questions/options reach the answering model. Responses are scored locally and saved with a dataset
snapshot and exact prompt. Optional graders add AI rubric judgments or Docker functional-test outcomes. The dashboard compares
paired MCQ answers. Local human annotations provide an independent spot-check of AI judging. Successful live results can be
published to LangSmith datasets/experiments; tracing is configured before live inference.

## Metric definitions

| Metric | Definition | Denominator / scope |
|---|---|---|
| MCQ accuracy | Correct reference option selected | All selected MCQs, including generation failures |
| Format compliance | Entire trimmed answer is a single uppercase A-D letter | All selected MCQs |
| AI rubric score | Structured binary judge verdict against reference/rubric | Successfully judge-scored open-ended answers only |
| Code pass rate | Fraction of functions passing every provided test | Code tasks attempted with Docker enabled; generation failures count as zero |
| Latency | Wall-clock duration of generation attempt | Stored per answer; judge latency separate; demo not inference |
| Tokens | Sum of generation usage reported by providers | Excludes judge calls; missing usage recorded as zero |
| Human / AI agreement | Fraction of matching binary verdicts | Examples with both human and AI-judge scores |
| Cohen's kappa | Observed agreement adjusted for marginal expected agreement | Same reviewed subset; undefined if expected agreement equals one |
| Wilson 95% interval | Binomial proportion interval | Live MCQ reference scores only |
| Exact McNemar test | Two-sided binomial test on discordant paired MCQ outcomes | Same dataset/mode pairs; exploratory, tiny non-random task sample |

No combined grand score is computed across incomparable metrics. In particular, code-style rubric grading is never labelled
as functional execution. MCQ correct-letter extraction permits a final `Answer: B` while the stricter formatting score fails it;
unrecognizable text is not guessed into an option.

## Storage and configuration

- `results/<uuid>.json`: experiment provenance, dataset snapshot and per-example outputs.
- `results/<uuid>.reviews.json`: independently stored human annotations, atomically replaced.
- `.env`: local secrets and trace configuration, excluded by `.gitignore`.
- Datasets are addressed in LangSmith by hash. Missing task IDs are added on retry without intentionally duplicating examples.
- LangSmith API failures are surfaced, not substituted with mock success.
- Provider exception classes are stored; raw exception text is not persisted because it can contain request details.

## Functional execution boundary

Generated code is never run by the host Python interpreter. The optional runner starts a disposable Docker container using
`python:3.12-slim`, no network, no host mounts, non-root user, read-only root, restricted capabilities, memory/CPU/PID limits and
a bounded subprocess duration. Container cleanup is attempted on success and on timeout. Code and teaching tests travel over stdin.

This is an educational isolation layer, not a hardened sandbox for adversarial public workloads. A candidate can potentially
inspect or interfere with an in-process test harness; small visible tests are not secure hidden benchmark tests. For rigorous
code benchmarking, use an established, separately audited sandbox and benchmark harness.

## Validation boundary

Automated tests exercise scoring, references withheld from answering inputs, malformed dataset rejection, statistics,
failure handling, report escaping, fake-model pipeline execution, mocked LangSmith API behavior, mocked Docker flags/cleanup,
and Streamlit page rendering. Passing these tests does not verify live API credentials, external service behavior, model quality,
or Docker availability on another computer. Those require the local end-to-end setup described in the Windows guide.

## Future work

Official licensed benchmark ingestion, full repository patch execution, independent calibrated judge configuration,
multiple stochastic repetitions, cost accounting for all calls, signed dataset manifests, provider-specific model versioning,
LangSmith annotation queues, online monitoring and authenticated multi-user deployment are not implemented in this version.
