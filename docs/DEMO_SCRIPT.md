# Demonstration script

## Opening explanation

"My project implements an LLM evaluation workflow using LangChain and LangSmith. It compares prompts or models on
the same versioned dataset, uses appropriate evaluators, inspects failures and validates some automatic judgments with human review.
It evaluates existing language models; it does not train their weights."

## 1. Dataset Studio

Show the five categories. Open one question and point to the reference and rubric.
Explain that these are 25 self-authored tasks representing capability areas from the report, not official benchmark samples.
The reasoning questions do not represent GPQA's graduate-level difficulty. The issue snippets are toy examples, not full SWE-bench repositories.

## 2. Run Experiments

Use the same model ID in A and B to isolate a prompt comparison as far as practical.
Show the two exact prompts. Start with Knowledge and Reasoning to keep the first live run short.
State the number of calls shown by the application before running.

If only demonstrating the interface offline, explicitly say: "These are scripted fixtures, not live model measurements."

## 3. Compare Results

Choose baseline as A and careful as B. Explain changes are B relative to A.
Open one improved example and one regression, if the actual live results contain them.
Do not guarantee that the careful prompt will improve a real model; finding no difference is also a valid result.
Show correctness and formatting separately. A model may choose the right answer but violate the required format.

## 4. Functional code evaluation (optional)

If Docker is installed, run Code generation with code tests enabled. Open a task's `code_score` and `code_reason`.
Explain that these are small custom test suites and the result is not official HumanEval pass@1/pass@k.
An AI judge saying code looks correct is not equivalent to executing tests.

## 5. Truthfulness and human review

Enable the AI judge in a live run, inspect its reason, and independently review several answers.
Show persisted human feedback and agreement. Explain possible model-judge bias and the limitations of a tiny review sample.
Human review is local in this implementation, not a hosted LangSmith annotation queue.

## 6. LangSmith

Enable tracing before a live run. Show an actual LangChain model-call trace in your workspace.
Publish both successful experiments and compare them over the same dataset in LangSmith.
Explain why uploading recorded results doesn't rerun the model, and why replay duration is not inference latency.

## 7. Export

Download the HTML report and open it. It includes mode/provenance labels, task answers, scores and human feedback.
Optionally use the browser's print dialog to save it as a PDF.

## Questions to prepare for

- Why evaluate an application instead of quoting a published model benchmark score?
- Why are deterministic scores and AI-judge scores reported separately?
- How do you stop reference answers leaking into the answering prompt?
- How is the dataset version recorded, and what changes its hash?
- Why might a better average score hide regressions?
- Why can different judge models produce misleading comparisons?
- What are the limits of tiny datasets and visible teaching tests?
- What does LangSmith trace, and which components did you implement yourself?
- Which features are implemented and which remain future work?

## Closing explanation

"The contribution is a transparent evaluation pipeline: versioned inputs, repeatable configurations, per-example comparison,
different grading methods and human validation. Results belong to the selected model, task set and evaluation setup,
not to LangSmith independently."
