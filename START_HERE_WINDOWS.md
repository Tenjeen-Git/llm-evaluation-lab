# Start here - Windows setup

This folder contains the complete application source. Finish local setup first; GitHub is a later step.
Your existing VS Code project can remain open in its own window.

## 1. Extract the project

1. Download `LLM_Evaluation_Lab_Complete.zip`.
2. Right-click it and choose **Extract All**.
3. Put the extracted `llm-evaluation-lab` folder somewhere convenient, such as Documents.
4. Open the folder. You should see `app.py`, `setup_windows.bat`, `requirements.txt`, and the `eval_lab` folder together.
5. Do not run the files from inside the ZIP preview. Extract first.

## 2. Check Python

Open a new VS Code window using **File > New Window**.
Choose **File > Open Folder**, then select the extracted `llm-evaluation-lab` folder.
Choose **Terminal > New Terminal** and run:

```powershell
python --version
```

Use Python **3.12** (3.11 is also supported). If it is missing, install Python from
https://www.python.org/downloads/ and select **Add python.exe to PATH** during installation.
Close and reopen VS Code after installation. If you have multiple Python installations,
`py -3.12 -m venv .venv` can explicitly select Python 3.12 instead of the generic `python` command.

## 3. Install once

The easiest option: double-click `setup_windows.bat` in File Explorer.
It creates an isolated `.venv`, installs the packages and creates an empty `.env` configuration file.
The first installation needs internet access and can take several minutes.
Wait until it says **Setup finished**. Do not close the installation window early.

Alternatively run these commands in the project terminal, one line at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Only copy `.env.example` on the first setup. Do not overwrite an `.env` containing your keys.
The commands use the environment's Python directly, so you do not need to activate it or change PowerShell execution policy.

## 4. Open the application

Double-click `run_windows.bat`, or run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Your browser should open at **http://localhost:8501**.
If it does not, paste that address into the browser yourself.
Keep the terminal window open while using the app. Press **Ctrl+C** in that terminal to stop it.

## 5. Try the offline demonstration first

1. Open **Dashboard**.
2. Click **Create two demo experiments**.
3. Open **Compare Results**.
4. Select `baseline-v1` as A and `careful-v2` as B. Both should say `scripted_demo`.
5. You will see three improved MCQs and one regressed MCQ. Baseline MCQ accuracy is 60%; careful is 80%.
6. Inspect `K03`: the baseline answer is correct but violates the letter-only format. This demonstrates separate correctness and formatting scores.
7. Inspect `K04`: the careful fixture is wrong. This demonstrates a regression despite improved average accuracy.
8. Open **Human Review**, choose an answer, write a reason, and save your assessment.
9. Return to Dashboard or Compare Results and download JSON, CSV or the HTML report.

**These demo answers are deliberately hand-written fixtures. They are not measured model performance.**
No LLM runs in demo mode; AI-judge scores remain unscored. Demo timings are not model inference timings.

## 6. Connect a real LLM

Choose **one** of the following routes. You do not need both.

### Option A - OpenAI API

1. Create your own API key at https://platform.openai.com/api-keys.
2. Ensure your API account has model access and sufficient billing/credits. A ChatGPT subscription does not itself supply API credits.
3. Open `.env` in VS Code. Add your key after `OPENAI_API_KEY=`. Do not put it in `app.py`.

```dotenv
OPENAI_API_KEY=your_actual_key_here
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=llm-evaluation-lab
```

4. Save `.env`, stop the app with Ctrl+C, and start it again.
5. Open **Run Experiments** and choose **OpenAI**.
6. The model fields default to `gpt-4o-mini`. If that model is unavailable to your account, enter a supported chat-model ID in both fields.
7. Start with **Knowledge** and **Reasoning** only. Keep AI judging and Docker tests unchecked for the first live run.
8. Click **Run both experiments**. This makes 20 answer calls on the bundled dataset. Provider usage may incur charges.
9. Inspect the saved results, including any error counts. The app never substitutes demo answers when a live call fails.

### Option B - Local Ollama model

1. Install Ollama from https://ollama.com/ and start it.
2. In a terminal, download a small model:

```powershell
ollama pull llama3.2:3b
```

3. In **Run Experiments**, choose **Ollama** and enter the exact tag you downloaded in both model fields.
4. Run the comparison. A local model requires enough RAM/disk space and may be slow on CPU.
5. API billing is not required for this local route. Model installation still requires a download.

### Add open-ended AI judging

After the initial live run works, include **Truthfulness**, **Code generation** and **Repository reasoning**.
Enable **Grade open-ended answers with an LLM rubric judge** to score those answers against their references and rubrics.
This adds inference calls and can add cost/latency. The same answering model serves as that experiment's judge.
Different judge models can disagree; their scores are not automatically interchangeable.
Some small local models may fail structured-output judging; judge failures remain ungraded and are visible in result details.

## 7. Enable LangSmith - essential for the complete topic demonstration

The offline demonstration alone does not demonstrate the hosted LangSmith integration.

1. Create or sign into your LangSmith account at https://smith.langchain.com.
2. Create a LangSmith API key in your account settings.
3. Add it to `.env` and enable tracing:

```dotenv
LANGSMITH_API_KEY=your_actual_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=llm-evaluation-lab
```

4. If your workspace uses a regional endpoint, set `LANGSMITH_ENDPOINT` to its documented endpoint as well.
5. Restart the app so the configuration loads.
6. Run a new live comparison. The real LangChain model calls will be traced in LangSmith.
7. Open **LangSmith & Guide** in the application.
8. Select a successful live experiment and click **Publish experiment to LangSmith**.
9. Publish the other experiment too. Identical dataset versions share the same LangSmith dataset.
10. Open LangSmith's dataset/experiment comparison view to inspect the recorded outputs and evaluator scores side by side.

Publishing sends the selected dataset and results to your workspace. It replays already-generated outputs and does not call the LLM again.
The replay trace duration is not model inference latency; original generation latency is uploaded separately as a metric.
To inspect real model-call traces, enable tracing **before** generation as described above.
Scripted demo experiments cannot be published as live-model experiments.

## 8. Optional functional testing for generated Python code

This feature adds a practical HumanEval-inspired component. It uses five self-authored functions and small test suites, not official HumanEval.
If you do not install Docker, the rest of the project still works. Code answers can be assessed by a human or rubric judge,
but those assessments must not be presented as functional test results.

1. Install and start Docker Desktop. Complete any WSL2 setup Docker requires on Windows.
2. Prepare the container image once:

```powershell
docker pull python:3.12-slim
```

3. Include **Code generation** in your experiment categories.
4. Check **Run code-generation tests in Docker** and start a run.

The application runs code only inside a disposable container with no network, no host-folder mount,
read-only root filesystem, non-root user, resource limits and a timeout. Never replace this with host-side `exec()`.
Container isolation reduces risk; it is not a hardened public code-execution service. Run this project locally.
The tests are small and visible in the dataset; passing them is not a proof of full correctness.

## 9. Run the tests

From the project terminal:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover dataset validation, reference leakage prevention, scoring, failed-call handling,
paired comparisons, report escaping, fake-model LangChain execution, mocked LangSmith uploads,
mocked Docker execution, and Streamlit page rendering.
They do not need API keys. Live provider calls and real Docker execution require your local configuration.

## 10. Demonstrate the project

Follow `docs/DEMO_SCRIPT.md`. For a final submission, record actual live results and show at least one real LangSmith experiment.
Do not present the canned 60%/80% demo values as findings about an LLM.
The study's five benchmark names describe the motivation; our custom tasks do not reproduce their official scores or difficulty.

## Common problems

| Problem | Action |
|---|---|
| Python not recognized / Microsoft Store opens | Install Python and enable PATH, then reopen VS Code. |
| `No module named ...` | Use `.\.venv\Scripts\python.exe` and rerun the installation command. |
| Browser does not open | Visit http://localhost:8501 while the terminal is running. |
| Port 8501 is already used | Add `--server.port 8502` to the Streamlit command and open localhost:8502. |
| OpenAI AuthenticationError | Check the API key in `.env` and restart the app. Never paste the key into a chat or screenshot. |
| OpenAI RateLimitError | Check API quota/billing and request limits; retry a smaller category selection. |
| Ollama connection error | Start Ollama and verify the model name with `ollama list`. |
| LangSmith authentication or connection failure | Check the key, account/workspace access and regional endpoint, then restart. |
| Docker prerequisite error | Start Docker Desktop and run the image-pull command. Alternatively leave code tests unchecked. |
| AI judge score is Not scored | Enable live judging; inspect `judge_error` if a judge call failed. Demo never runs a judge. |

All experiments and human reviews are stored in the local `results` folder and survive restarting the app.
This is a local single-user academic application. Authentication, hosted deployment, online monitoring,
Prompt Hub integration, official benchmark ingestion and LangSmith annotation queues are not implemented.

**Do not upload `.env`, API keys, `.venv`, or private results when you later publish source code.**
