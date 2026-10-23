"""Optional container-isolated functional tests. Never execute model code on the host."""
import json
import re
import subprocess
from uuid import uuid4

IMAGE = "python:3.12-slim"
# Tests and generated functions execute inside a disposable Docker container.
HARNESS = r'''
import contextlib, io, json, sys
payload = json.load(sys.stdin)
try:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload["code"], "candidate.py", "exec"), namespace)
        function = namespace[payload["entry_point"]]
        passed = sum(function(*case["args"]) == case["expected"] for case in payload["tests"])
    print(json.dumps({"passed": passed, "total": len(payload["tests"])}))
except BaseException as error:
    print(json.dumps({"passed": 0, "total": len(payload["tests"]), "error": type(error).__name__}))
'''


def extract_code(answer):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S | re.I)
    return blocks[0].strip() if blocks else answer.strip()


def check_docker():
    try:
        result = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=8)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def evaluate_code(task, answer):
    container_name = "evaluation-lab-" + uuid4().hex
    payload = {"code": extract_code(answer), "entry_point": task["entry_point"], "tests": task["tests"]}
    command = ["docker", "run", "--rm", "--name", container_name, "-i", "--network", "none", "--read-only",
               "--memory", "128m", "--memory-swap", "128m", "--cpus", "0.5", "--pids-limit", "32",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", IMAGE, "python", "-I", "-c", HARNESS]
    try:
        result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=15)
        if result.returncode:
            return {"code_score": 0, "code_reason": "Container execution failed."}
        output = json.loads(result.stdout)
        passed, total = output["passed"], output["total"]
        if total != len(task["tests"]) or not isinstance(passed, int) or not 0 <= passed <= total:
            raise ValueError("Invalid harness output")
        return {"code_score": int(passed == total), "code_reason": f"{passed}/{total} teaching tests passed."}
    except subprocess.TimeoutExpired:
        # --rm removes a container on normal exit; a timeout requires explicit cleanup.
        return {"code_score": 0, "code_reason": "Execution exceeded the time limit."}
    except (FileNotFoundError, ValueError, KeyError):
        return {"code_score": None, "code_reason": "Docker unavailable or invalid harness output; ungraded."}
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
