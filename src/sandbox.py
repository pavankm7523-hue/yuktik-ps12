import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid

from src.quixbugs import BUGGY_DIR, JSON_DIR, QUIXBUGS, SANDBOX_DIR, TEST_DIR


def _text(x):
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


def _run_pytest(workdir, target, timeout, fail_fast):
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=short", "-p", "no:cacheprovider"]
    if fail_fast:
        cmd.append("-x")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
        code, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, out, timed_out = -1, _text(exc.stdout), True
        err = _text(exc.stderr) + f"\nkilled after {timeout:.1f}s without a result (likely an infinite loop)"
    ms = (time.perf_counter() - start) * 1000.0
    return {
        "exit_code": code,
        "passed": code == 0,
        "timed_out": timed_out,
        "runtime_ms": round(ms, 1),
        "stdout": out,
        "stderr": err,
        "command": " ".join(cmd[1:]),
    }


_overhead = None


def startup_overhead():
    """seconds pytest needs just to start and collect on this machine. measured once per process."""
    global _overhead
    if _overhead is None:
        work = _fresh_dir()
        try:
            (work / "test_noop.py").write_text("def test_noop():\n    pass\n", encoding="utf-8")
            _overhead = _run_pytest(work, "test_noop.py", 60.0, False)["runtime_ms"] / 1000.0
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return _overhead


def _budget(timeout):
    # the timeout is a budget for the tests themselves. interpreter and pytest start-up can
    # take a second or two on windows, so it is added on top instead of eating the budget.
    # doubled because several sandboxes start at once during search and benchmarking.
    return timeout + 2.0 * startup_overhead()


def _fresh_dir():
    path = SANDBOX_DIR / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    return path


def run_quixbugs_tests(program_name, code, timeout=3.0, fail_fast=False):
    """run vendor/QuixBugs/python_testcases/test_<name>.py against `code` in a throwaway copy of the repo layout."""
    work = _fresh_dir()
    try:
        (work / "python_programs").mkdir()
        (work / "python_testcases").mkdir()
        (work / "json_testcases").mkdir()
        shutil.copy(QUIXBUGS / "conftest.py", work / "conftest.py")
        shutil.copy(BUGGY_DIR / "node.py", work / "python_programs" / "node.py")
        (work / "python_programs" / f"{program_name}.py").write_text(code, encoding="utf-8")
        for helper in ("load_testdata.py", "node.py", f"test_{program_name}.py"):
            shutil.copy(TEST_DIR / helper, work / "python_testcases" / helper)
        cases = JSON_DIR / f"{program_name}.json"
        if cases.exists():
            shutil.copy(cases, work / "json_testcases" / cases.name)
        return _run_pytest(work, f"python_testcases/test_{program_name}.py", _budget(timeout), fail_fast)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_custom_tests(code, assert_block, timeout=3.0, fail_fast=False):
    """run user supplied assert statements against pasted code."""
    work = _fresh_dir()
    try:
        (work / "candidate.py").write_text(code, encoding="utf-8")
        body = textwrap.indent(textwrap.dedent(assert_block).strip() or "pass", "    ")
        test_src = "from candidate import *\n\n\ndef test_custom():\n" + body + "\n"
        (work / "test_custom.py").write_text(test_src, encoding="utf-8")
        return _run_pytest(work, "test_custom.py", _budget(timeout), fail_fast)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_tests(code, program_name=None, assert_block=None, timeout=3.0, fail_fast=False):
    if program_name:
        return run_quixbugs_tests(program_name, code, timeout, fail_fast)
    return run_custom_tests(code, assert_block or "", timeout, fail_fast)
