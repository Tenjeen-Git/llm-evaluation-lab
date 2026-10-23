# Validation record

Validated on 23 October 2026 using Python 3.12 in the build environment.

- `python -m pytest -q`: **17 passed**.
- All Python files compiled successfully.
- Both 25-task scripted experiments completed without generation errors.
- Expected fixture MCQ accuracy: baseline 60%, careful 80% (10 MCQs each).
- Paired fixture comparison: three improvements, one regression.
- Streamlit testing exercised all six screens, running both experiments and saving a human review.
- LangChain's prompt/model pipeline was exercised with a local fake model; no paid provider calls were made.
- LangSmith upload logic was tested with a fake client; no real workspace was used.
- Docker invocation, isolation flags and cleanup were tested with a mocked subprocess; no real Docker runtime was used.

Live-model quality, credentials, hosted LangSmith behavior, real container execution and Windows batch scripts
must be checked on the user's configured computer. Tests do not certify official benchmark reproduction or model performance.
