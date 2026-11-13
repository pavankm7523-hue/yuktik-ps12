import ast
import io
import keyword
import math
import re
import tokenize

from src.quixbugs import docstring_lines

# column names follow the nasa jm1 csv in vendor/jm1-defect-prediction/data/jm1.csv
JM1_FEATURES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]
# python-only structure counts that the jm1 schema has no column for
PY_FEATURES = [
    "max_nesting", "ast_depth", "ast_nodes", "loops", "conditionals", "compares",
    "subscripts", "calls", "returns", "bool_ops", "aug_assigns",
]
FEATURES = JM1_FEATURES + PY_FEATURES

FEATURE_LABELS = {
    "loc": "lines of code (span)",
    "v(g)": "cyclomatic complexity",
    "ev(g)": "essential complexity",
    "iv(g)": "design complexity",
    "n": "halstead length",
    "v": "halstead volume",
    "l": "halstead level",
    "d": "halstead difficulty",
    "i": "halstead intelligence",
    "e": "halstead effort",
    "b": "halstead bug estimate",
    "t": "halstead time (s)",
    "lOCode": "code lines",
    "uniq_Op": "unique operators",
    "uniq_Opnd": "unique operands",
    "total_Op": "total operators",
    "total_Opnd": "total operands",
    "branchCount": "branch count",
    "max_nesting": "max block nesting depth",
    "ast_depth": "max ast node depth",
    "ast_nodes": "ast node count",
    "loops": "for/while loops",
    "conditionals": "if statements and expressions",
    "compares": "comparison nodes",
    "subscripts": "subscript nodes",
    "calls": "call nodes",
    "returns": "return/yield nodes",
    "bool_ops": "and/or nodes",
    "aug_assigns": "augmented assignments",
}

_CLOSERS = {")", "]", "}"}
_SKIP_TOKENS = {
    tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
    tokenize.COMMENT, tokenize.ENDMARKER, tokenize.ENCODING,
}
_DECISIONS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler, ast.Assert, ast.match_case)
_BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.FunctionDef,
           ast.AsyncFunctionDef, ast.ClassDef, ast.Match)


def _halstead_tokens(source):
    skip = docstring_lines(source)
    ops, opnds = [], []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _SKIP_TOKENS or tok.start[0] in skip:
            continue
        if tok.type == tokenize.OP:
            # brackets are counted once per pair
            if tok.string not in _CLOSERS:
                ops.append(tok.string)
        elif tok.type == tokenize.NAME:
            if keyword.iskeyword(tok.string) or keyword.issoftkeyword(tok.string):
                ops.append(tok.string)
            else:
                opnds.append(tok.string)
        elif tok.type in (tokenize.NUMBER, tokenize.STRING) or tok.type == getattr(tokenize, "FSTRING_START", -1):
            opnds.append(tok.string)
    return ops, opnds


def _decision_count(node):
    count = 0
    for sub in ast.walk(node):
        if isinstance(sub, _DECISIONS):
            count += 1
        elif isinstance(sub, ast.BoolOp):
            count += len(sub.values) - 1
        elif isinstance(sub, ast.comprehension):
            count += len(sub.ifs)
    return count


def _has_call(node):
    return any(isinstance(s, ast.Call) for s in ast.walk(node))


def _unstructured_exits(func):
    # rough essential complexity: jumps out of the middle of loops and early returns
    exits = 0
    for sub in ast.walk(func):
        if isinstance(sub, (ast.For, ast.While, ast.AsyncFor)):
            for inner in ast.walk(sub):
                if isinstance(inner, (ast.Break, ast.Continue, ast.Return)):
                    exits += 1
    returns = [s for s in ast.walk(func) if isinstance(s, ast.Return)]
    body_last = func.body[-1] if func.body else None
    exits += sum(1 for r in returns if r is not body_last)
    return exits


def _max_nesting(node, depth=0):
    best = depth
    for child in ast.iter_child_nodes(node):
        d = depth + 1 if isinstance(child, _BLOCKS) else depth
        best = max(best, _max_nesting(child, d))
    return best


def _ast_depth(node, depth=1):
    return max([depth] + [_ast_depth(c, depth + 1) for c in ast.iter_child_nodes(node)])


def _count(nodes, *types):
    return sum(1 for n in nodes if isinstance(n, types))


def _units(tree):
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return funcs or [tree]


def extract_metrics(source):
    tree = ast.parse(source)
    skip = docstring_lines(source)
    raw_lines = source.splitlines()

    code_idx = [i for i, line in enumerate(raw_lines, start=1)
                if line.strip() and not line.strip().startswith("#") and i not in skip]
    lo_code = len(code_idx)
    loc = (code_idx[-1] - code_idx[0] + 1) if code_idx else 0

    ops, opnds = _halstead_tokens(source)
    n1, n2 = len(set(ops)), len(set(opnds))
    N1, N2 = len(ops), len(opnds)
    length = N1 + N2
    vocab = n1 + n2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    difficulty = (n1 / 2.0) * (N2 / n2) if n2 else 0.0
    level = 1.0 / difficulty if difficulty else 0.0
    effort = difficulty * volume

    units = _units(tree)
    # the trailing description docstrings are not code, keep them out of the node counts
    nodes = [n for n in ast.walk(tree) if getattr(n, "lineno", 0) not in skip]
    decisions = _decision_count(tree)
    vg = 1 + decisions
    evg = min(vg, 1 + sum(_unstructured_exits(u) for u in units if not isinstance(u, ast.Module)))
    design = sum(1 for s in ast.walk(tree) if isinstance(s, _DECISIONS) and _has_call(s))
    ivg = min(vg, 1 + design)

    vals = {
        "loc": loc,
        "v(g)": vg,
        "ev(g)": evg,
        "iv(g)": ivg,
        "n": length,
        "v": round(volume, 2),
        "l": round(level, 3),
        "d": round(difficulty, 2),
        "i": round(level * volume, 2),
        "e": round(effort, 2),
        "b": round(volume / 3000.0, 3),
        "t": round(effort / 18.0, 2),
        "lOCode": lo_code,
        "uniq_Op": n1,
        "uniq_Opnd": n2,
        "total_Op": N1,
        "total_Opnd": N2,
        "branchCount": 2 * decisions + 1,
        "max_nesting": _max_nesting(tree),
        "ast_depth": _ast_depth(tree),
        "ast_nodes": len(nodes),
        "loops": _count(nodes, ast.For, ast.AsyncFor, ast.While),
        "conditionals": _count(nodes, ast.If, ast.IfExp),
        "compares": _count(nodes, ast.Compare),
        "subscripts": _count(nodes, ast.Subscript),
        "calls": _count(nodes, ast.Call),
        "returns": _count(nodes, ast.Return, ast.Yield, ast.YieldFrom),
        "bool_ops": _count(nodes, ast.BoolOp),
        "aug_assigns": _count(nodes, ast.AugAssign),
    }
    return {k: vals[k] for k in FEATURES}


# brace languages: lexical halstead and mccabe, no parser needed
_KEYWORDS = {
    "java": "abstract assert boolean break byte case catch char class continue default do double else enum extends "
            "final finally float for if implements import instanceof int interface long new package private "
            "protected public return short static super switch this throw throws try void while null true false",
    "c": "auto break case char const continue default do double else enum extern float for goto if int long "
         "register return short signed sizeof static struct switch typedef union unsigned void volatile while NULL",
    "cpp": "auto bool break case catch char class const continue default delete do double else enum false float for "
           "if int long namespace new nullptr private protected public return short sizeof static struct switch "
           "template this throw true try typedef typename unsigned using virtual void while std",
    "javascript": "async await break case catch class const continue default delete do else export extends false "
                  "finally for function if import in instanceof let new null return switch this throw true try "
                  "typeof undefined var void while yield of",
}
_KEYWORDS = {k: set(v.split()) for k, v in _KEYWORDS.items()}
_LEX = re.compile(
    r"(?P<str>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)"
    r"|(?P<num>\b0[xX][0-9a-fA-F]+\b|\b\d+(?:\.\d+)?[fFlLuU]*\b)"
    r"|(?P<id>[A-Za-z_$][\w$]*)"
    r"|(?P<op>>>>=|===|!==|>>>|<<=|>>=|\+\+|--|->|::|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>"
    r"|[-+*/%=<>!&|^~?:.,;(){}\[\]])"
)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_CMP = {"==", "!=", "===", "!==", "<", "<=", ">", ">="}
_AUG = {"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>="}


def _strip_comments(code):
    # keep line breaks so line numbers stay aligned
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), code)


def _lex(code):
    return [(m.lastgroup, m.group(), code.count("\n", 0, m.start()) + 1) for m in _LEX.finditer(code)]


def _brace_metrics(code, lang):
    clean = _strip_comments(code)
    toks = _lex(clean)
    kw = _KEYWORDS[lang]
    ops, opnds = [], []
    for kind, text, _ in toks:
        if kind == "op":
            if text not in (")", "]", "}"):
                ops.append(text)
        elif kind == "id" and text in kw:
            ops.append(text)
        else:
            opnds.append(text)
    words = [t for k, t, _ in toks if k in ("id", "op")]

    decisions = sum(words.count(w) for w in ("if", "for", "while", "case", "catch", "?", "&&", "||"))
    exits = sum(words.count(w) for w in ("return", "break", "continue", "throw", "goto"))
    funcs = max(1, len(re.findall(r"\)\s*(?:throws\s+[\w.,\s]+)?\{", clean)) - words.count("if")
                - words.count("for") - words.count("while") - words.count("switch") - words.count("catch"))
    # a name followed by "(" is a call unless a type or name sits right before it (a definition)
    call_lines = [a[2] for p, a, b in zip(toks, toks[1:], toks[2:])
                  if a[0] == "id" and b[1] == "(" and a[1] not in kw
                  and not (p[0] == "id" and p[1] not in ("return", "new", "throw", "await"))]
    calls = len(call_lines)
    design = sum(1 for (_, t, ln) in toks if t in ("if", "while", "for") and ln in call_lines)

    depth = best = 0
    for _, t, _ in toks:
        if t == "{":
            depth += 1
            best = max(best, depth)
        elif t == "}":
            depth = max(0, depth - 1)

    lines = [i for i, line in enumerate(clean.splitlines(), start=1) if line.strip()]
    n1, n2, N1, N2 = len(set(ops)), len(set(opnds)), len(ops), len(opnds)
    length, vocab = N1 + N2, n1 + n2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    difficulty = (n1 / 2.0) * (N2 / n2) if n2 else 0.0
    level = 1.0 / difficulty if difficulty else 0.0
    effort = difficulty * volume
    vg = 1 + decisions
    # essential complexity: early exits beyond one per function
    evg = min(vg, 1 + max(0, exits - funcs))
    return {
        "loc": (lines[-1] - lines[0] + 1) if lines else 0,
        "v(g)": vg, "ev(g)": evg, "iv(g)": min(vg, 1 + design),
        "n": length, "v": round(volume, 2), "l": round(level, 3), "d": round(difficulty, 2),
        "i": round(level * volume, 2), "e": round(effort, 2), "b": round(volume / 3000.0, 3),
        "t": round(effort / 18.0, 2), "lOCode": len(lines),
        "uniq_Op": n1, "uniq_Opnd": n2, "total_Op": N1, "total_Opnd": N2,
        "branchCount": 2 * decisions + 1,
        # structure counts are lexical stand-ins for the python ast counts
        "max_nesting": max(0, best - (2 if lang in ("java", "cpp") and "class" in words else 1)),
        "ast_depth": best + 1, "ast_nodes": len(toks),
        "loops": words.count("for") + words.count("while"),
        "conditionals": words.count("if") + words.count("?"),
        "compares": sum(1 for w in words if w in _CMP),
        "subscripts": words.count("["), "calls": calls, "returns": words.count("return"),
        "bool_ops": words.count("&&") + words.count("||"),
        "aug_assigns": sum(1 for w in words if w in _AUG) + words.count("++") + words.count("--"),
    }


def extract_jm1_metrics(code_str, lang="python"):
    """jm1-schema metrics for python, java, c, cpp and javascript, same 29 columns for every language."""
    if lang == "python":
        return extract_metrics(code_str)
    if lang not in _KEYWORDS:
        raise ValueError(f"unsupported language: {lang}")
    vals = _brace_metrics(code_str, lang)
    return {k: vals[k] for k in FEATURES}
