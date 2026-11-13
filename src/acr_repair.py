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
from src.quixbugs import EXT, SANDBOX_DIR, VENDOR, code_lines, list_programs, read_correct
from src.sandbox import _balanced, run_tests, toolchain

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


def build_context(code, fault_line, window=4, lang="python"):
    """autocoderover style retrieval: index the file, find the scopes around the fault line, slice the code."""
    if lang != "python":
        return brace_context(code, fault_line, window)
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


def _brace_depths(code):
    """per line: brace depth at line start, ignoring braces inside strings and comments."""
    clean = re.sub(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
                   lambda m: re.sub(r"[^\n]", " ", m.group(0)), code, flags=re.S)
    depth, out = 0, []
    for line in clean.splitlines():
        out.append((depth, line))
        depth += line.count("{") - line.count("}")
    return out


def _block_span(depths, line_no, want):
    """1-based start and end line of the block opened at depth `want` that contains line_no."""
    start = None
    for i in range(line_no - 1, -1, -1):
        d, text = depths[i]
        if d == want and "{" in text:
            start = i + 1
            break
        if d < want:
            break
    if start is None:
        return None
    for j in range(start, len(depths)):
        # the block closes on the first later line that starts back at its own depth
        if depths[j][0] <= want:
            return start, j
    return start, len(depths)


_SIG = re.compile(r"([A-Za-z_$][\w$]*)\s*\(([^()]*)\)\s*(?:const\s*)?(?:throws\s+[\w.,\s]+)?\{?\s*$")
_CTRL = {"if", "for", "while", "switch", "catch", "else", "do", "try", "synchronized", "return", "sizeof"}


def brace_context(code, fault_line, window=4):
    """brace-balanced method slicing for java, c, cpp and javascript."""
    lines = code.splitlines()
    depths = _brace_depths(code)
    lo, hi = max(1, fault_line - window), min(len(lines), fault_line + window)
    snippet = "\n".join(f"{i:>4} {lines[i - 1]}" for i in range(lo, hi + 1))
    ctx = {"fault_line": fault_line, "fault_code": lines[fault_line - 1].strip() if 0 < fault_line <= len(lines) else "",
           "class": None, "function": None, "function_span": None, "signature": None, "arguments": [],
           "local_assignments": [], "calls": [], "snippet": snippet, "function_source": "",
           "indexed": {"classes": re.findall(r"\b(?:class|struct|interface)\s+(\w+)", code),
                       "functions": sorted({m for m in re.findall(r"\b(\w+)\s*\([^;{()]*\)\s*(?:throws[\w.,\s]+)?\{", code)}
                                           - _CTRL)}}
    if not 0 < fault_line <= len(lines):
        return ctx
    # walk outwards through the enclosing blocks until one opens with a function signature
    level = depths[fault_line - 1][0]
    if "{" in depths[fault_line - 1][1]:
        level += 1
    while level > 0:
        level -= 1
        span = _block_span(depths, fault_line, level)
        if not span:
            continue
        head = " ".join(depths[i][1].strip() for i in range(max(0, span[0] - 3), span[0])).rstrip().rstrip("{")
        head = re.split(r"[;{}]", head)[-1].strip()
        m = _SIG.search(head)
        if m and m.group(1) not in _CTRL and ctx["function"] is None:
            ctx["function"], ctx["function_span"] = m.group(1), span
            ctx["signature"] = head.strip().rstrip("{").strip()
            ctx["arguments"] = [a.split()[-1].strip("&*[]") for a in m.group(2).split(",") if a.strip()]
            ctx["function_source"] = "\n".join(lines[span[0] - 1:span[1]])
        cls = re.search(r"\b(?:class|struct|interface)\s+(\w+)", head)
        if cls and not ctx["class"]:
            ctx["class"] = cls.group(1)
    if ctx["function_span"]:
        first = ctx["function_span"][0]
        seen = set()
        for i, text in enumerate(ctx["function_source"].splitlines()[1:], start=first + 1):
            m = re.match(r"\s*(?:(?:final|const|let|var|auto|static|unsigned)\s+)*[\w<>\[\],.*&]+\s+\**(\w+)\s*(?:=|;|\[)",
                         text) or re.match(r"\s*(\w+)\s*(?:[-+*/%^&|]?=)(?!=)", text)
            if m and m.group(1) not in seen | _CTRL:
                seen.add(m.group(1))
                ctx["local_assignments"].append({"name": m.group(1), "line": i, "code": text.strip()})
        body = "\n".join(ctx["function_source"].splitlines()[1:])
        ctx["calls"] = sorted({c for c in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)} - _CTRL)
    return ctx


# operator swaps for brace languages, longest token first so "<=" is never read as "<"
_SWAPS = [
    ("===", ["!=="]), ("!==", ["==="]), ("<<=", [">>="]), (">>=", ["<<="]),
    ("&&", ["||"]), ("||", ["&&"]), ("==", ["!="]), ("!=", ["=="]), ("<=", ["<", ">="]),
    (">=", [">", "<="]), ("^=", ["&=", "|="]), ("&=", ["|=", "^="]), ("|=", ["&=", "^="]),
    ("+=", ["-="]), ("-=", ["+="]), ("*=", ["/="]), ("/=", ["*="]), ("++", ["--"]), ("--", ["++"]),
    ("<<", [">>"]), (">>", ["<<"]), ("<", ["<=", ">"]), (">", [">=", "<"]), ("^", ["&", "|"]),
    ("&", ["^", "|"]), ("|", ["^", "&"]), ("+", ["-"]), ("-", ["+"]), ("*", ["/"]), ("/", ["*"]), ("%", ["/"]),
]
_SWAP = dict(_SWAPS)
_SWAP_RE = re.compile("|".join(re.escape(a) for a, _ in _SWAPS))


def _code_part(line):
    """the line with string literals blanked and any trailing // comment cut off."""
    text = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", lambda m: " " * len(m.group(0)), line)
    cut = text.find("//")
    return text if cut < 0 else text[:cut]


def brace_candidates(code, fault_lines, lang, limit=60):
    """single-token rewrites of the suspicious lines: operator swaps, +-1 on bounds, == to .equals in java."""
    lines = code.splitlines(keepends=True)
    out, seen = [], {code}

    def add(line_no, new_line, what):
        cand = "".join(lines[:line_no - 1] + [new_line] + lines[line_no:])
        if cand not in seen and _balanced(cand):
            seen.add(cand)
            out.append({"source": cand, "line_no": line_no, "edit": f"line {line_no}: {what}"})

    for line_no in fault_lines:
        if not 0 < line_no <= len(lines):
            continue
        text = lines[line_no - 1]
        scan = _code_part(text)
        for m in _SWAP_RE.finditer(scan):
            # "->" and "::" are member access, not arithmetic
            if scan[m.end():m.end() + 1] in (">", ":") and m.group(0) == "-":
                continue
            for new_tok in _SWAP[m.group(0)]:
                add(line_no, text[:m.start()] + new_tok + text[m.end():], f"{m.group(0)} -> {new_tok}")
        for m in re.finditer(r"(?<![\w.])(\w+(?:\.length|\.size\(\))?)(?=\s*[;)\]])", scan):
            if m.group(1) in ("return", "break", "continue", "null", "true", "false"):
                continue
            for delta in ("- 1", "+ 1"):
                add(line_no, text[:m.start(1)] + f"{m.group(1)} {delta}" + text[m.end(1):],
                    f"{m.group(1)} -> {m.group(1)} {delta}")
        if lang == "java":
            for m in re.finditer(r"([\w.]+(?:\([^()]*\))?)\s*(==|!=)\s*([\w.\"]+(?:\([^()]*\))?)", scan):
                left, op, right = m.groups()
                if "null" in (left, right) or right.replace(".", "").isdigit() or left.replace(".", "").isdigit():
                    continue
                neg = "!" if op == "!=" else ""
                add(line_no, text[:m.start()] + f"{neg}{left}.equals({right})" + text[m.end():],
                    f"{m.group(0)} -> {neg}{left}.equals({right})")
        if lang == "javascript":
            for m in re.finditer(r"(?<![=!<>])(==|!=)(?!=)", scan):
                add(line_no, text[:m.start()] + m.group(1) + "=" + text[m.end():], f"{m.group(1)} -> {m.group(1)}=")
        if lang in ("c", "cpp"):
            for m in re.finditer(r"\b(if|while)\s*\(([^=!<>()]*[^=!<>()\s])\s*=\s*([^=)][^)]*)\)", scan):
                add(line_no, text[:m.start()] + f"{m.group(1)} ({m.group(2).strip()} == {m.group(3).strip()})"
                    + text[m.end():], "assignment in condition -> comparison")
            for m in re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", scan):
                add(line_no, text[:m.start()] + f"{m.group(1)}.0 / {m.group(2)}" + text[m.end():],
                    f"{m.group(0)} -> floating division")
    return out[:limit]


def make_diff(before, after, name="program.py"):
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}", n=2))


def _parses(src, lang="python"):
    if lang != "python":
        return _balanced(src)
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


def _split_code(source, lang="python"):
    # code part and trailing remainder. both trees end with description docstrings, and some
    # fixed files also carry alternative solutions inside one. none of that is part of the fix.
    lines = source.splitlines()
    numbered = code_lines(source, lang)
    last = numbered[-1][0] if numbered else len(lines)
    return lines[:last], lines[last:]


def ground_truth_patch(program_name, buggy_code, lang="python"):
    """apply the reference fix from correct_<lang>_programs as hunks, leaving the rest of the buggy file alone."""
    b, tail = _split_code(buggy_code, lang)
    c, _ = _split_code(read_correct(program_name, lang), lang)
    # the java fixes live in another package, keep the buggy file's own declaration
    c = [x.replace("package correct_java_programs;", "package java_programs;") for x in c]
    sm = difflib.SequenceMatcher(a=[x.rstrip() for x in b], b=[x.rstrip() for x in c], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        out.extend(b[i1:i2] if tag == "equal" else c[j1:j2])
    return "\n".join(out + tail) + "\n"


LANG_NAMES = {"python": "Python", "java": "Java", "javascript": "JavaScript", "c": "C", "cpp": "C++"}
LANG_HINTS = {
    "python": "Watch for off-by-one bounds, wrong comparison operators and mutable default arguments.",
    "java": "Watch for == on String or boxed objects instead of .equals(), off-by-one loop bounds, ^ used for &, "
            "null dereferences and List.get index bounds.",
    "javascript": "Watch for == instead of ===, off-by-one loop bounds over .length, and async callbacks in forEach.",
    "c": "Watch for = inside if conditions, off-by-one < vs <= bounds, uninitialised variables and integer division.",
    "cpp": "Watch for = inside if conditions, off-by-one < vs <= bounds, uninitialised variables, integer division "
           "and new/delete mismatches.",
}

PROMPT = """You are fixing a single defect in a {lang_name} function.
{hint}

Fault localization flagged line {fault_line}: `{fault_code}`
Enclosing function: {function} with arguments {arguments}
Local assignments: {locals}
Other suspicious lines: {others}

Code around the fault line:
{snippet}

Full function:
```{fence}
{function_source}
```

Return the complete corrected function in one ```{fence} block. Change as little as possible and keep the signature."""


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
    blocks = re.findall(r"```(?:python|py|java|javascript|js|cpp|c\+\+|c)?\s*\n(.*?)```", reply, flags=re.S)
    return (blocks[0] if blocks else reply).strip("\n")


def llm_patch(code, ctx, ranked, lang="python"):
    """returns (patched_source, provider). raises on network or api errors so the caller can log them."""
    for env, name, ask in PROVIDERS:
        if not os.environ.get(env):
            continue
        prompt = PROMPT.format(
            lang_name=LANG_NAMES[lang], hint=LANG_HINTS[lang], fence="cpp" if lang == "cpp" else lang,
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
            body = (textwrap.indent(textwrap.dedent(fixed), indent) if lang == "python" else fixed).splitlines()
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


def can_execute(lang, program_name=None, assert_block=None):
    """true when this input has tests that can actually run on this machine."""
    if lang == "python":
        return bool(program_name) or bool((assert_block or "").strip())
    return bool((assert_block or "").strip()) and bool(toolchain().get(lang))


def repair(code, program_name=None, assert_block=None, ranked=None, use_llm=True, use_search=True,
           allow_ground_truth=True, timeout=3.0, top_k=3, lang="python"):
    """try llm, then validated single-edit search, then the quixbugs reference fix. first patch that passes wins."""
    quix = program_name if program_name in list_programs(lang) else None
    # java quixbugs tests are junit files, so java programs run with user asserts only
    test_name = quix if lang == "python" else None
    if ranked is None:
        ranked = rank_lines(code, exclude_program=quix if lang == "python" else None, lang=lang)
    top = [r["line_no"] for r in ranked[:top_k]]
    ctx = build_context(code, top[0], lang=lang) if top else {}
    can_test = can_execute(lang, test_name, assert_block)
    log = []
    if not can_test and lang != "python":
        log.append(f"no runnable tests for {lang} here: " + (
            "add assert lines" if not (assert_block or "").strip() else f"no {lang} compiler or runtime on PATH"))

    def passes(src):
        return run_tests(src, test_name, assert_block, timeout=timeout, fail_fast=True, lang=lang)["passed"] is True

    def done(patched, source, validated):
        name = f"{program_name or 'candidate'}{EXT[lang]}"
        return {"patched": patched, "source": source, "origin": patch_origin(source), "validated": validated,
                "diff": make_diff(code, patched, name), "context": ctx, "log": log, "fault_lines": top, "lang": lang}

    if use_llm and llm_provider() and top:
        try:
            patched, provider = llm_patch(code, ctx, ranked, lang)
            if patched and _parses(patched, lang) and patched != code:
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

    if use_search and can_test and _parses(code, lang):
        if lang == "python":
            cands = template_candidates(code, top)
            fix = mutable_default_fix(code)
            if fix:
                cands.insert(0, {"source": fix, "line_no": 0, "edit": "mutable default argument -> None guard"})
        else:
            cands = brace_candidates(code, top, lang)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda c: passes(c["source"]), cands))
        log.append(f"single-edit search: {len(cands)} candidates on lines {top}, {sum(results)} pass")
        passing = [c for c, ok in zip(cands, results) if ok]
        if passing:
            # several edits can satisfy the tests. keep the smallest one.
            best = min(passing, key=lambda c: len(c["source"]))
            log.append(f"accepted edit: {best['edit']}")
            return done(best["source"], "ast_search", True)

    if allow_ground_truth and quix:
        log.append(f"fell back to the reference fix from correct_{lang}_programs")
        return done(ground_truth_patch(quix, code, lang), "ground_truth", False)

    log.append("no patch found")
    return {"patched": None, "source": None, "origin": "none", "validated": False, "diff": "", "context": ctx,
            "log": log, "fault_lines": top, "lang": lang}


def verify_with_pytest(program_name, buggy_code, patched_code, assert_block=None, timeout=3.0, lang="python"):
    """run the same tests on the original and the patched source, each in its own sandbox."""
    test_name = program_name if lang == "python" else None
    with ThreadPoolExecutor(max_workers=2) as pool:
        before = pool.submit(run_tests, buggy_code, test_name, assert_block, timeout, False, lang)
        after = pool.submit(run_tests, patched_code, test_name, assert_block, timeout, False, lang)
        return {"before": before.result(), "after": after.result()}
