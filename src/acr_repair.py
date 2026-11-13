import ast
import difflib
import importlib.util
import json
import os
import re
import textwrap
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from src import mutations
from src.line_localizer import rank_lines
from src.quixbugs import SANDBOX_DIR, VENDOR, code_lines, list_programs, read_correct
from src.sandbox import run_tests

ACR_SEARCH_UTILS = VENDOR / "auto-code-rover" / "app" / "search" / "search_utils.py"

_acr = None


def acr_search_utils():
    # search_utils.py only needs the stdlib, so it is loaded straight from the clone
    # instead of importing the whole app package and its llm dependencies
    global _acr
    if _acr is None:
        spec = importlib.util.spec_from_file_location("acr_search_utils", ACR_SEARCH_UTILS)
        _acr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_acr)
    return _acr


def _enclosing(spans, line_no):
    inside = [s for s in spans if s[1] <= line_no <= s[2]]
    return min(inside, key=lambda s: s[2] - s[1]) if inside else None


def build_context(code, fault_line, window=4):
    """autocoderover style retrieval: index the file, find the scopes around the fault line, slice the code."""
    acr = acr_search_utils()
    SANDBOX_DIR.mkdir(exist_ok=True)
    tmp = SANDBOX_DIR / f"ctx_{uuid.uuid4().hex[:10]}.py"
    tmp.write_text(code, encoding="utf-8")
    try:
        parsed = acr.parse_python_file(str(tmp))
        snippet = acr.get_code_region_around_line(str(tmp), fault_line, window_size=window)
    finally:
        tmp.unlink(missing_ok=True)
    if parsed is None:
        return {"fault_line": fault_line, "error": "file does not parse", "snippet": snippet or ""}

    classes, class_to_funcs, top_level_funcs, _ = parsed
    methods = [f for funcs in class_to_funcs.values() for f in funcs]
    func = _enclosing(top_level_funcs + methods, fault_line)
    cls = _enclosing(classes, fault_line)
    lines = code.splitlines()

    ctx = {
        "fault_line": fault_line,
        "fault_code": lines[fault_line - 1].strip() if 0 < fault_line <= len(lines) else "",
        "class": cls[0] if cls else None,
        "function": func[0] if func else None,
        "function_span": (func[1], func[2]) if func else None,
        "signature": None,
        "arguments": [],
        "local_assignments": [],
        "calls": [],
        "snippet": snippet or "",
        "function_source": "",
        "indexed": {"classes": [c[0] for c in classes], "functions": sorted({f[0] for f in top_level_funcs})},
    }
    if not func:
        return ctx

    node = next(n for n in ast.walk(ast.parse(code))
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno == func[1])
    sig_lines = acr.extract_func_sig_from_ast(node)
    ctx["signature"] = " ".join(lines[i - 1].strip() for i in sig_lines)
    ctx["arguments"] = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    ctx["function_source"] = "\n".join(lines[func[1] - 1:func[2]])
    seen = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store) and sub.id not in seen:
            seen.add(sub.id)
            ctx["local_assignments"].append({"name": sub.id, "line": sub.lineno, "code": lines[sub.lineno - 1].strip()})
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id not in ctx["calls"]:
            ctx["calls"].append(sub.func.id)
    return ctx


def make_diff(before, after, name="program.py"):
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}", n=2))


def _parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def template_candidates(code, fault_lines, limit=24):
    """single-edit rewrites of the suspicious lines, most suspicious line first."""
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    out, seen = [], {code}
    for line_no in fault_lines:
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.expr, ast.AugAssign)) and getattr(n, "lineno", None) == line_no
                 and n.end_lineno == line_no]
        for node in nodes:
            for variant in mutations.node_variants(node):
                text = lines[line_no - 1]
                new_line = text[:node.col_offset] + ast.unparse(variant) + text[node.end_col_offset:]
                cand = "".join(lines[:line_no - 1] + [new_line] + lines[line_no:])
                if cand in seen or not _parses(cand):
                    continue
                seen.add(cand)
                out.append({"source": cand, "line_no": line_no, "edit": f"{text.strip()}  ->  {new_line.strip()}"})
                if len(out) >= limit:
                    return out
    return out


def mutable_default_fix(code):
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args.args[len(node.args.args) - len(node.args.defaults):]
        guards = []
        for arg, default in zip(args, node.args.defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                guards.append((arg.arg, ast.unparse(default), default))
        if not guards:
            continue
        indent = " " * node.body[0].col_offset
        header = lines[node.lineno - 1:node.body[0].lineno - 1]
        head_text = "".join(header)
        if node.body[0].lineno == node.lineno:
            continue
        for name, literal, _ in guards:
            head_text = re.sub(rf"\b{name}\s*=\s*{re.escape(literal)}", f"{name}=None", head_text, count=1)
        guard_text = "".join(f"{indent}if {name} is None:\n{indent}    {name} = {literal}\n" for name, literal, _ in guards)
        return "".join(lines[:node.lineno - 1]) + head_text + guard_text + "".join(lines[node.body[0].lineno - 1:])
    return None


def _split_code(source):
    # code part and trailing remainder. both trees end with description docstrings, and some
    # fixed files also carry alternative solutions inside one. none of that is part of the fix.
    lines = source.splitlines()
    numbered = code_lines(source)
    last = numbered[-1][0] if numbered else len(lines)
    return lines[:last], lines[last:]


def ground_truth_patch(program_name, buggy_code):
    """apply the reference fix from correct_python_programs as hunks, leaving the rest of the buggy file alone."""
    b, tail = _split_code(buggy_code)
    c, _ = _split_code(read_correct(program_name))
    sm = difflib.SequenceMatcher(a=[x.rstrip() for x in b], b=[x.rstrip() for x in c], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        out.extend(b[i1:i2] if tag == "equal" else c[j1:j2])
    return "\n".join(out + tail) + "\n"


PROMPT = """You are fixing a single defect in a Python function.

Fault localization flagged line {fault_line}: `{fault_code}`
Enclosing function: {function} with arguments {arguments}
Local assignments: {locals}
Other suspicious lines: {others}

Code around the fault line:
{snippet}

Full function:
```python
{function_source}
```

Return the complete corrected function in one ```python block. Change as little as possible and keep the signature."""


def _post_json(url, payload, headers, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ask_gemini(prompt):
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post_json(url, {"contents": [{"parts": [{"text": prompt}]}]},
                      {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _ask_openai(prompt):
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    data = _post_json("https://api.openai.com/v1/chat/completions",
                      {"model": model, "messages": [{"role": "user", "content": prompt}]},
                      {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    return data["choices"][0]["message"]["content"]


def _ask_anthropic(prompt):
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"),
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(block.text for block in resp.content if block.type == "text")


PROVIDERS = [
    ("GEMINI_API_KEY", "gemini", _ask_gemini),
    ("OPENAI_API_KEY", "openai", _ask_openai),
    ("ANTHROPIC_API_KEY", "anthropic", _ask_anthropic),
]


def llm_provider():
    for env, name, _ in PROVIDERS:
        if os.environ.get(env):
            return name
    return None


def _extract_code(reply):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    return (blocks[0] if blocks else reply).strip("\n")


def llm_patch(code, ctx, ranked):
    """returns (patched_source, provider). raises on network or api errors so the caller can log them."""
    for env, name, ask in PROVIDERS:
        if not os.environ.get(env):
            continue
        prompt = PROMPT.format(
            fault_line=ctx["fault_line"], fault_code=ctx["fault_code"], function=ctx["function"],
            arguments=ctx["arguments"],
            locals=", ".join(f"{a['name']} (line {a['line']})" for a in ctx["local_assignments"]) or "none",
            others=", ".join(f"line {r['line_no']}" for r in ranked[1:3]) or "none",
            snippet=ctx["snippet"], function_source=ctx["function_source"] or code,
        )
        fixed = _extract_code(ask(prompt))
        lines = code.splitlines()
        if ctx.get("function_span"):
            start, end = ctx["function_span"]
            indent = lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())]
            body = textwrap.indent(textwrap.dedent(fixed), indent).splitlines()
            patched = "\n".join(lines[:start - 1] + body + lines[end:]) + "\n"
        else:
            patched = fixed + "\n"
        return patched, name
    return None, None


DISCOVERED = "discovered"
REFERENCE = "reference_fallback"


def patch_origin(source):
    """discovered = the tool produced the patch itself. reference_fallback = copied from the known fix."""
    if not source:
        return "none"
    if source == "ground_truth":
        return REFERENCE
    if source == "ast_search" or source.startswith("llm:"):
        return DISCOVERED
    return "unknown"


def repair(code, program_name=None, assert_block=None, ranked=None, use_llm=True, use_search=True,
           allow_ground_truth=True, timeout=3.0, top_k=3):
    """try llm, then validated single-edit search, then the quixbugs reference fix. first patch that passes wins."""
    if ranked is None:
        ranked = rank_lines(code, exclude_program=program_name if program_name in list_programs() else None)
    top = [r["line_no"] for r in ranked[:top_k]]
    ctx = build_context(code, top[0]) if top else {}
    can_test = bool(program_name) or bool((assert_block or "").strip())
    log = []

    def passes(src):
        return run_tests(src, program_name, assert_block, timeout=timeout, fail_fast=True)["passed"]

    def done(patched, source, validated):
        name = f"{program_name or 'candidate'}.py"
        return {"patched": patched, "source": source, "origin": patch_origin(source), "validated": validated,
                "diff": make_diff(code, patched, name), "context": ctx, "log": log, "fault_lines": top}

    if use_llm and llm_provider() and top:
        try:
            patched, provider = llm_patch(code, ctx, ranked)
            if patched and _parses(patched) and patched != code:
                if not can_test:
                    log.append(f"{provider}: patch returned, no tests to validate it")
                    return done(patched, f"llm:{provider}", False)
                if passes(patched):
                    log.append(f"{provider}: patch passes the tests")
                    return done(patched, f"llm:{provider}", True)
                log.append(f"{provider}: patch did not pass the tests")
            else:
                log.append(f"{provider}: reply was not a usable patch")
        except Exception as exc:
            log.append(f"llm call failed: {type(exc).__name__}: {exc}")

    if use_search and can_test and _parses(code):
        cands = template_candidates(code, top)
        fix = mutable_default_fix(code)
        if fix:
            cands.insert(0, {"source": fix, "line_no": 0, "edit": "mutable default argument -> None guard"})
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda c: passes(c["source"]), cands))
        log.append(f"single-edit search: {len(cands)} candidates on lines {top}, {sum(results)} pass")
        passing = [c for c, ok in zip(cands, results) if ok]
        if passing:
            # several edits can satisfy the tests. keep the smallest one.
            best = min(passing, key=lambda c: len(c["source"]))
            log.append(f"accepted edit: {best['edit']}")
            return done(best["source"], "ast_search", True)

    if allow_ground_truth and program_name in list_programs():
        log.append("fell back to the reference fix from correct_python_programs")
        return done(ground_truth_patch(program_name, code), "ground_truth", False)

    log.append("no patch found")
    return {"patched": None, "source": None, "origin": "none", "validated": False, "diff": "", "context": ctx,
            "log": log, "fault_lines": top}


def verify_with_pytest(program_name, buggy_code, patched_code, assert_block=None, timeout=3.0):
    """run the same tests on the original and the patched source, each in its own subprocess."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        before = pool.submit(run_tests, buggy_code, program_name, assert_block, timeout)
        after = pool.submit(run_tests, patched_code, program_name, assert_block, timeout)
        return {"before": before.result(), "after": after.result()}
