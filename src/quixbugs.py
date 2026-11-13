import ast
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
QUIXBUGS = VENDOR / "QuixBugs"
BUGGY_DIR = QUIXBUGS / "python_programs"
CORRECT_DIR = QUIXBUGS / "correct_python_programs"
TEST_DIR = QUIXBUGS / "python_testcases"
JSON_DIR = QUIXBUGS / "json_testcases"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SANDBOX_DIR = ROOT / ".sandbox"

# helpers and graph drivers live next to the programs but are not benchmark subjects
_SKIP = {"node"}


def list_programs():
    names = []
    for p in sorted(BUGGY_DIR.glob("*.py")):
        name = p.stem
        if name in _SKIP or name.endswith("_test"):
            continue
        if (CORRECT_DIR / p.name).exists():
            names.append(name)
    return names


def read_buggy(name):
    return (BUGGY_DIR / f"{name}.py").read_text(encoding="utf-8")


def read_correct(name):
    return (CORRECT_DIR / f"{name}.py").read_text(encoding="utf-8")


def has_tests(name):
    return (TEST_DIR / f"test_{name}.py").exists()


def docstring_lines(source):
    """line numbers covered by bare string expressions (docstrings, the quixbugs trailer)."""
    lines = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def code_lines(source):
    """(line_no, text) for lines that carry code: no blanks, comments or docstrings."""
    skip = docstring_lines(source)
    out = []
    for i, line in enumerate(source.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in skip:
            continue
        out.append((i, line))
    return out


def faulty_lines(buggy_src, correct_src):
    """buggy-file line numbers touched by the fix, compared on code lines only."""
    b = code_lines(buggy_src)
    c = code_lines(correct_src)
    sm = difflib.SequenceMatcher(a=[t.strip() for _, t in b], b=[t.strip() for _, t in c], autojunk=False)
    hits = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            hits.update(b[k][0] for k in range(i1, i2))
        else:
            # pure insertion in the fix: blame the neighbouring buggy lines
            if i1 - 1 >= 0:
                hits.add(b[i1 - 1][0])
            if i1 < len(b):
                hits.add(b[i1][0])
    return sorted(hits)
