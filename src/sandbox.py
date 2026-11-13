import json
import os
import queue
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

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


# fast python runner. pytest import and collection cost ~600 ms per run on windows. this one
# keeps a pool of bare interpreters, loads the test files against a tiny pytest stand-in that
# records parametrize cases, and calls every case itself. a run that overruns is killed with its worker.
_PYTEST_STUB = '''use_correct = False
run_slow = False


class Skipped(Exception):
    pass


def skip(msg=""):
    raise Skipped(msg)


def approx(expected, rel=1e-6, abs=1e-12):
    import math

    class _Approx:
        def __eq__(self, other):
            return math.isclose(other, expected, rel_tol=rel, abs_tol=abs)

        def __repr__(self):
            return f"approx({expected!r})"
    return _Approx()


class _Mark:
    def parametrize(self, names, values, **_):
        def deco(fn):
            fn._cases = (names, list(values))
            return fn
        return deco

    def __getattr__(self, _name):
        return lambda *a, **k: (lambda fn: fn)


mark = _Mark()


def fail(msg=""):
    raise AssertionError(msg)


class raises:
    def __init__(self, exc, *a, **k):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError(f"did not raise {self.exc}")
        return issubclass(kind, self.exc)
'''

_WORKER = '''import importlib.util, io, json, os, sys, traceback, types
sys.setrecursionlimit(10000)
stub = types.ModuleType("pytest")
exec(compile(STUB, "pytest_stub", "exec"), stub.__dict__)
sys.modules["pytest"] = stub
reply = sys.stdout
base_modules = set(sys.modules)
base_path = list(sys.path)
base_cwd = os.getcwd()


def run_job(targets, fail_fast):
    results = []
    for path in targets:
        spec = importlib.util.spec_from_file_location("t_" + os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            results.append([path, False, traceback.format_exc(limit=4)])
            continue
        for attr, fn in list(vars(mod).items()):
            if not attr.startswith("test") or not callable(fn) or isinstance(fn, type):
                continue
            cases = getattr(fn, "_cases", None)
            calls = [{}]
            if cases:
                names, values = cases
                names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
                calls = [dict(zip(names, v)) if len(names) > 1 else {names[0]: v} for v in values]
            for idx, kwargs in enumerate(calls):
                test_id = os.path.basename(path) + "::" + attr + (f"[{idx}]" if cases else "")
                try:
                    fn(**kwargs)
                    results.append([test_id, True, ""])
                except stub.Skipped:
                    results.append([test_id, True, ""])
                except BaseException:
                    results.append([test_id, False, traceback.format_exc(limit=4)])
                if fail_fast and not results[-1][1]:
                    return results
    return results


for line in sys.stdin:
    job = json.loads(line)
    os.chdir(job["cwd"])
    sys.path[:] = [job["cwd"]] + base_path
    stub.use_correct, stub.run_slow = False, False
    captured = io.StringIO()
    sys.stdout = sys.stderr = captured
    try:
        res = run_job(job["targets"], job["fail_fast"])
    except BaseException:
        res = [["worker", False, traceback.format_exc(limit=4)]]
    finally:
        sys.stdout, sys.stderr = reply, sys.__stderr__
        # forget every module the job imported so the next candidate loads fresh
        for name in set(sys.modules) - base_modules:
            del sys.modules[name]
        os.chdir(base_cwd)
    reply.write(json.dumps({"results": res, "output": captured.getvalue()[-4000:]}) + "\\n")
    reply.flush()
'''


class _Worker:
    """one long-lived interpreter. a run that overruns its budget kills the process."""

    def __init__(self):
        SANDBOX_DIR.mkdir(exist_ok=True)
        script = SANDBOX_DIR / "_fast_worker.py"
        if not script.exists():
            script.write_text("STUB = " + repr(_PYTEST_STUB) + "\n" + _WORKER, encoding="utf-8")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-I", "-S", "-u", str(script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     encoding="utf-8", cwd=SANDBOX_DIR, env=env)
        self.replies = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def run(self, cwd, targets, fail_fast, timeout):
        self.proc.stdin.write(json.dumps({"cwd": str(cwd), "targets": targets, "fail_fast": fail_fast}) + "\n")
        self.proc.stdin.flush()
        try:
            line = self.replies.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise TimeoutError
        if line is None:
            raise RuntimeError("worker crashed (stack overflow or hard exit in the candidate)")
        return json.loads(line)

    def alive(self):
        return self.proc.poll() is None

    def kill(self):
        try:
            self.proc.kill()
        except OSError:
            pass


_idle = queue.Queue()


def _take_worker():
    while True:
        try:
            w = _idle.get_nowait()
        except queue.Empty:
            return _Worker()
        if w.alive():
            return w


def _run_fast(workdir, targets, timeout, fail_fast):
    start = time.perf_counter()
    timed_out, crash, results, output = False, "", [], ""
    worker = _take_worker()
    try:
        reply = worker.run(workdir, targets, fail_fast, timeout)
        results, output = reply["results"], reply["output"]
        _idle.put(worker)
    except TimeoutError:
        timed_out = True
        crash = f"killed after {timeout:.1f}s without a result (likely an infinite loop)"
    except Exception as exc:
        worker.kill()
        crash = f"{type(exc).__name__}: {exc}"
    ms = (time.perf_counter() - start) * 1000.0
    failed = [r for r in results if not r[1]]
    passed = bool(results) and not failed and not timed_out and not crash
    lines = [f"FAILED {tid}" for tid, ok, _ in failed][:20]
    summary = f"{len(results) - len(failed)} passed, {len(failed)} failed" + (", timed out" if timed_out else "")
    trace = failed[0][2].strip() if failed else ""
    return {
        "exit_code": -1 if (timed_out or crash) else (0 if passed else 1),
        "passed": passed,
        "timed_out": timed_out,
        "runtime_ms": round(ms, 1),
        "stdout": "\n".join(lines + [summary] + ([output.strip()] if output.strip() else [])),
        "stderr": "\n".join(x for x in (trace, crash) if x),
        "command": "fast-runner " + " ".join(targets),
        "tests_run": len(results),
        "tests_failed": len(failed),
    }


def _quixbugs_layout(program_name, code):
    work = _fresh_dir()
    (work / "python_programs").mkdir()
    (work / "python_testcases").mkdir()
    (work / "json_testcases").mkdir()
    shutil.copy(BUGGY_DIR / "node.py", work / "python_programs" / "node.py")
    (work / "python_programs" / f"{program_name}.py").write_text(code, encoding="utf-8")
    for helper in ("load_testdata.py", "node.py", f"test_{program_name}.py"):
        shutil.copy(TEST_DIR / helper, work / "python_testcases" / helper)
    cases = JSON_DIR / f"{program_name}.json"
    if cases.exists():
        shutil.copy(cases, work / "json_testcases" / cases.name)
    return work


def run_quixbugs_fast(program_name, code, timeout=3.0, fail_fast=False):
    work = _quixbugs_layout(program_name, code)
    try:
        return _run_fast(work, [f"python_testcases/test_{program_name}.py"], timeout + 1.0, fail_fast)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_custom_fast(code, assert_block, timeout=3.0, fail_fast=False):
    work = _fresh_dir()
    try:
        (work / "candidate.py").write_text(code, encoding="utf-8")
        body = textwrap.indent(textwrap.dedent(assert_block).strip() or "pass", "    ")
        (work / "test_custom.py").write_text("from candidate import *\n\n\ndef test_custom():\n" + body + "\n",
                                             encoding="utf-8")
        return _run_fast(work, ["test_custom.py"], timeout + 1.0, fail_fast)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# brace languages


def _tool(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def toolchain():
    """which compilers and runtimes are on PATH. java needs javac, a bare jre cannot compile."""
    return {
        "javascript": _tool("node"),
        "java": _tool("javac") and _tool("java"),
        "c": _tool("gcc", "clang", "cc"),
        "cpp": _tool("g++", "clang++", "c++"),
    }


def _exec(cmd, cwd, timeout, label):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        code, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, out, timed_out = -1, _text(exc.stdout), True
        err = _text(exc.stderr) + f"\nkilled after {timeout:.1f}s without a result (likely an infinite loop)"
    return {
        "exit_code": code, "passed": code == 0 and not timed_out, "timed_out": timed_out,
        "runtime_ms": round((time.perf_counter() - start) * 1000.0, 1),
        "stdout": out, "stderr": err, "command": label,
    }


def _js_harness(code, assert_block):
    return ("'use strict';\nconst assert = require('assert');\n" + code + "\n\n// tests\n"
            + (assert_block or "").strip() + "\nconsole.log('all assertions passed');\n")


def _java_class(code):
    m = re.search(r"\bpublic\s+(?:final\s+|abstract\s+)?class\s+(\w+)", code) or re.search(r"\bclass\s+(\w+)", code)
    return m.group(1) if m else "Main"


def _java_harness(code, assert_block):
    # the candidate keeps its own class. the asserts run from a separate main with -ea.
    code = re.sub(r"^\s*package\s+[\w.]+;\s*", "", code, flags=re.M)
    body = textwrap.indent(textwrap.dedent(assert_block or "").strip(), "        ")
    harness = ("import java.util.*;\n\npublic class Harness {\n    public static void main(String[] args) {\n"
               + body + "\n        System.out.println(\"all assertions passed\");\n    }\n}\n")
    return code, harness


def _c_harness(code, assert_block, lang):
    head = "#include <assert.h>\n#include <stdio.h>\n" if lang == "c" else "#include <cassert>\n#include <cstdio>\n"
    body = textwrap.indent(textwrap.dedent(assert_block or "").strip(), "    ")
    return (head + code + "\n\nint main(void) {\n" + body
            + "\n    printf(\"all assertions passed\\n\");\n    return 0;\n}\n")


def _balanced(code):
    """bracket balance outside strings and comments. the static check used when no compiler is on PATH."""
    clean = re.sub(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", "", code, flags=re.S)
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    for ch in clean:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _static_result(code, lang, why):
    ok = _balanced(code)
    return {
        "exit_code": None, "passed": None, "timed_out": False, "runtime_ms": 0.0,
        "stdout": f"static check: brackets {'balanced' if ok else 'NOT balanced'}. tests were not executed.",
        "stderr": why, "command": f"static-check ({lang})", "static_only": True, "syntax_ok": ok,
    }


def run_brace_tests(code, lang, assert_block, timeout=3.0):
    tools = toolchain()
    if not (assert_block or "").strip():
        return _static_result(code, lang, "no test assertions given")
    if not tools.get(lang):
        need = {"java": "javac (a jdk)", "c": "gcc or clang", "cpp": "g++ or clang++", "javascript": "node"}[lang]
        return _static_result(code, lang, f"{need} is not on PATH, so the tests could not be compiled or run")
    work = _fresh_dir()
    try:
        if lang == "javascript":
            (work / "candidate.js").write_text(_js_harness(code, assert_block), encoding="utf-8")
            return _exec([tools["javascript"], "candidate.js"], work, timeout + 2.0, "node candidate.js")
        if lang == "java":
            src, harness = _java_harness(code, assert_block)
            (work / f"{_java_class(src)}.java").write_text(src, encoding="utf-8")
            (work / "Harness.java").write_text(harness, encoding="utf-8")
            built = _exec([shutil.which("javac"), "-nowarn", f"{_java_class(src)}.java",
                           "Harness.java"], work, 60.0, "javac")
            if not built["passed"]:
                built["command"] = "javac (compile failed)"
                return built
            return _exec([shutil.which("java"), "-ea", "-cp", ".", "Harness"], work, timeout + 2.0, "java -ea Harness")
        ext = ".c" if lang == "c" else ".cpp"
        (work / f"candidate{ext}").write_text(_c_harness(code, assert_block, lang), encoding="utf-8")
        exe = work / ("candidate.exe" if os.name == "nt" else "candidate")
        built = _exec([tools[lang], f"candidate{ext}", "-o", str(exe)], work, 60.0, f"{Path(tools[lang]).name} compile")
        if not built["passed"]:
            built["command"] += " (compile failed)"
            return built
        return _exec([str(exe)], work, timeout + 1.0, exe.name)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_tests(code, program_name=None, assert_block=None, timeout=3.0, fail_fast=False, lang="python", engine="fast"):
    """engine "fast" uses the lightweight runner, "pytest" the original pytest subprocess."""
    if lang != "python":
        return run_brace_tests(code, lang, assert_block, timeout)
    if program_name:
        if engine == "pytest":
            return run_quixbugs_tests(program_name, code, timeout, fail_fast)
        return run_quixbugs_fast(program_name, code, timeout, fail_fast)
    if engine == "pytest":
        return run_custom_tests(code, assert_block or "", timeout, fail_fast)
    return run_custom_fast(code, assert_block or "", timeout, fail_fast)
