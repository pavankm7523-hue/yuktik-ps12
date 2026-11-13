import ast
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.quixbugs import MODEL_DIR, code_lines, faulty_lines, list_programs, read_buggy, read_correct

TFIDF_PATH = MODEL_DIR / "tfidf_line.pkl"

# adapted from preprocess_code_line in vendor/DeepLineDP/script/preprocess_data.py.
# deeplinedp targets java and throws operators and subscripts away; python one-line bugs
# live in exactly those tokens, so literals are normalised but operators are kept.
_STR = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN = re.compile(r"<str>|<num>|[A-Za-z_$][\w$]*|===|!==|&&|\|\||\+\+|--|->|[-+*/%&|^]=|==|!=|<=|>=|//|\*\*|<<|>>"
                    r"|[-+*/%&|^~<>=\[\](){}.,:;?!]")

W_TOKEN = 0.5
W_AST = 0.5


def preprocess_line(line, lang="python"):
    line = _STR.sub("<str>", line)
    line = line.split("#" if lang == "python" else "//", 1)[0]
    line = _NUM.sub("<num>", line)
    return line.strip()


def line_features(line):
    toks = _TOKEN.findall(preprocess_line(line))
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def _training_lines(exclude=None):
    lines, labels = [], []
    for name in list_programs():
        if name == exclude:
            continue
        buggy = read_buggy(name)
        bad = set(faulty_lines(buggy, read_correct(name)))
        for no, text in code_lines(buggy):
            lines.append(text)
            labels.append(1 if no in bad else 0)
    return lines, np.array(labels)


def fit_token_model(exclude=None, smooth=0.02):
    """tf-idf over quixbugs code lines, then a per-token risk: share of its weight that sits on fixed lines."""
    lines, labels = _training_lines(exclude)
    vec = TfidfVectorizer(analyzer=line_features, min_df=1, sublinear_tf=True)
    X = vec.fit_transform(lines)
    bad = np.asarray(X[labels == 1].mean(axis=0)).ravel()
    ok = np.asarray(X[labels == 0].mean(axis=0)).ravel()
    risk = (bad + smooth) / (bad + ok + 2 * smooth)
    return {"vectorizer": vec, "risk": risk, "n_lines": len(lines), "n_faulty": int(labels.sum())}


def save_token_model():
    MODEL_DIR.mkdir(exist_ok=True)
    model = fit_token_model()
    joblib.dump(model, TFIDF_PATH)
    return model


def load_token_model():
    if not TFIDF_PATH.exists():
        return save_token_model()
    return joblib.load(TFIDF_PATH)


def token_score(line, model):
    row = model["vectorizer"].transform([line])
    if row.nnz == 0:
        return 0.5
    w = row.data
    return float(np.dot(w, model["risk"][row.indices]) / w.sum())


def _is_one(node):
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == 1


def _ast_signals(tree):
    """line_no -> {signal: weight}. each check is a pattern that keeps showing up in one-line defects."""
    signals = {}
    owner = {}

    def add(node, name, weight):
        signals.setdefault(node.lineno, {})[name] = weight

    def visit(node, func_name):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    add(default, "mutable_default", 0.6)
        if isinstance(node, ast.stmt):
            owner.setdefault(node.lineno, type(node).__name__)
        if isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                add(node, "boundary_compare", 0.35)
            elif any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                add(node, "equality_compare", 0.15)
        elif isinstance(node, ast.Subscript):
            add(node, "slice" if isinstance(node.slice, ast.Slice) else "index_subscript", 0.25)
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.BitAnd, ast.BitXor, ast.BitOr, ast.LShift, ast.RShift)):
                add(node, "bitwise_op", 0.4)
            elif isinstance(node.op, (ast.Add, ast.Sub)) and (_is_one(node.right) or _is_one(node.left)):
                add(node, "off_by_one", 0.3)
            elif isinstance(node.op, (ast.Mod, ast.FloorDiv, ast.Div)):
                add(node, "division", 0.15)
        elif isinstance(node, ast.AugAssign):
            add(node, "aug_assign", 0.15)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if func_name and node.func.id == func_name:
                add(node, "recursive_call", 0.35)
            elif node.func.id == "range":
                add(node, "range_bound", 0.25)
        elif isinstance(node, (ast.While, ast.If)):
            add(node, "loop_condition" if isinstance(node, ast.While) else "branch_condition", 0.2)
        elif isinstance(node, (ast.Return, ast.Yield)):
            add(node, "return_value", 0.15)
        for child in ast.iter_child_nodes(node):
            visit(child, func_name)

    visit(tree, None)
    return signals, owner


# language rules: (pattern, signal, weight). checked per line after literals are masked.
# NUM and STR stand for masked literals, see rule_signals.
_RULES = {
    "java": [
        (r"\bfor\s*\(.*;\s*\w+\s*<=\s*[\w.]+(?:\.length|\.size\(\))", "loop_bound_off_by_one", 0.35),
        (r"(?:STR|\b[A-Z]\w*\b|\w+\.\w+\(\))\s*[!=]=|[!=]=\s*(?:STR|\b[a-z]\w*\.(?:get|name|toString)\()",
         "reference_equality", 0.45),
        (r"\^", "bitwise_xor", 0.4),
        (r"\.get\(\s*\w+\s*(?:[-+]\s*\w+\s*)?\)", "list_get_bounds", 0.3),
        (r"\b[a-z]\w*\.\w+", "null_dereference", 0.1),
        (r"\[\s*\w+\s*(?:[-+]\s*\w+\s*)?\]", "array_index", 0.25),
    ],
    "c": [
        (r"\b(?:if|while)\s*\([^)]*[^=!<>]=[^=][^)]*\)", "assignment_in_condition", 0.6),
        (r"\bfor\s*\(.*;\s*\w+\s*<=\s*\w+", "loop_bound_off_by_one", 0.4),
        (r"\[\s*\w+\s*(?:[-+]\s*\w+\s*)?\]", "array_index", 0.25),
        (r"^\s*[\w:<>]+\s*\*\s*\w+\s*;|^\s*\w+\s+\w+\s*\[\s*NUM\s*\]\s*;|^\s*(?:int|double|float|char|long)\s+\w+\s*;",
         "uninitialised_storage", 0.3),
        (r"\bNUM\s*/\s*NUM\b|/\s*NUM\b", "integer_division", 0.35),
        (r"\^|[^&]&[^&=]|[^|]\|[^|=]", "bitwise_op", 0.3),
    ],
    "javascript": [
        (r"[^=!]==[^=]|!=[^=]", "loose_equality", 0.45),
        (r"\bfor\s*\(.*;\s*\w+\s*<=\s*[\w.]+\.length", "loop_bound_off_by_one", 0.45),
        (r"\.(?:forEach|map)\(\s*async|\bawait\b.*\bforEach\b", "async_in_iterator", 0.4),
        (r"\.(?:sort|reverse|splice|push|pop|shift)\(", "in_place_mutation", 0.2),
        (r"\[\s*\w+\s*(?:[-+]\s*\w+\s*)?\]", "array_index", 0.2),
        (r"\bvar\b", "function_scoped_var", 0.15),
    ],
}
_RULES["cpp"] = _RULES["c"] + [(r"\bdelete\b|\bnew\b", "manual_memory", 0.2)]
_SHARED = [
    (r"(?<![<>])[<>]=?(?![<>])", "boundary_compare", 0.3),
    (r"[-+]\s*NUM\b", "off_by_one", 0.2),
    (r"\breturn\b", "return_value", 0.15),
    (r"\b(?:if|while)\s*\(", "branch_condition", 0.15),
    (r"[-+*/%&|^]=|\+\+|--", "aug_assign", 0.1),
]
_OWNER = [("for", "For"), ("while", "While"), ("if", "If"), ("else", "If"), ("return", "Return"),
          ("case", "Case"), ("switch", "Switch"), ("throw", "Raise")]


def rule_signals(text, lang):
    """language rule hits for one source line, masked like the token model sees it."""
    line = preprocess_line(text, lang).replace("<num>", "NUM").replace("<str>", "STR")
    hits = {}
    for pattern, name, weight in _RULES.get(lang, []) + _SHARED:
        if re.search(pattern, line):
            hits[name] = max(weight, hits.get(name, 0.0))
    return hits


def _brace_owner(text):
    s = text.strip()
    for kw, name in _OWNER:
        if re.match(r"(?:\}\s*)?" + kw + r"\b", s):
            return name
    if s.endswith("{"):
        return "FunctionDef" if "(" in s else "ClassDef"
    return "Assign" if re.search(r"[^=!<>]=[^=]", s) else "Expr"


def rank_lines(code, model=None, exclude_program=None, lang="python"):
    """rank code lines by suspicion. pass exclude_program to score a quixbugs file without having trained on it."""
    if exclude_program and lang == "python":
        model = fit_token_model(exclude=exclude_program)
    model = model or load_token_model()
    if lang == "python":
        try:
            signals, owner = _ast_signals(ast.parse(code))
        except SyntaxError:
            signals, owner = {}, {}
    else:
        signals, owner = {}, {}
        for no, text in code_lines(code, lang):
            signals[no] = rule_signals(text, lang)
            owner[no] = _brace_owner(text)

    rows = []
    for no, text in code_lines(code, lang):
        sig = signals.get(no, {})
        ast_part = 1.0 - float(np.prod([1.0 - w for w in sig.values()])) if sig else 0.0
        tok_part = token_score(text if lang == "python" else preprocess_line(text, lang), model)
        score = W_TOKEN * tok_part + W_AST * ast_part
        rows.append({
            "line_no": no,
            "code": text.rstrip(),
            "suspicion_score": round(min(1.0, max(0.0, score)), 4),
            "ast_node_type": owner.get(no, "Continuation"),
            "signals": ", ".join(sorted(sig, key=sig.get, reverse=True)),
            "token_score": round(tok_part, 4),
            "ast_score": round(ast_part, 4),
        })
    rows.sort(key=lambda r: (-r["suspicion_score"], r["line_no"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


if __name__ == "__main__":
    m = save_token_model()
    print(f"wrote {TFIDF_PATH} ({m['n_lines']} lines, {m['n_faulty']} faulty)")
