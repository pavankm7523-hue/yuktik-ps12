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
JAVA_BUGGY_DIR = QUIXBUGS / "java_programs"
JAVA_CORRECT_DIR = QUIXBUGS / "correct_java_programs"

LANGS = ["python", "java", "javascript", "c", "cpp"]
EXT = {"python": ".py", "java": ".java", "javascript": ".js", "c": ".c", "cpp": ".cpp"}
# quixbugs ships python and java versions of the same 40 programs
QUIXBUGS_DIRS = {"python": (BUGGY_DIR, CORRECT_DIR), "java": (JAVA_BUGGY_DIR, JAVA_CORRECT_DIR)}

# helpers and graph drivers live next to the programs but are not benchmark subjects
_SKIP = {"node", "Node", "WeightedEdge"}


def list_programs(lang="python"):
    if lang not in QUIXBUGS_DIRS:
        return []
    buggy_dir, correct_dir = QUIXBUGS_DIRS[lang]
    names = []
    for p in sorted(buggy_dir.glob(f"*{EXT[lang]}")):
        name = p.stem
        if name in _SKIP or name.lower().endswith("_test"):
            continue
        if (correct_dir / p.name).exists():
            names.append(name)
    return names


def program_path(name, lang="python", correct=False):
    return QUIXBUGS_DIRS[lang][1 if correct else 0] / f"{name}{EXT[lang]}"


def read_buggy(name, lang="python"):
    return program_path(name, lang).read_text(encoding="utf-8")


def read_correct(name, lang="python"):
    return program_path(name, lang, correct=True).read_text(encoding="utf-8")


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


def block_comment_lines(source):
    """line numbers that hold only /* */ comment text or // comments in brace languages."""
    lines = set()
    inside = False
    for i, line in enumerate(source.splitlines(), start=1):
        s = line.strip()
        if inside:
            lines.add(i)
            if "*/" in s:
                inside = False
                if s.split("*/", 1)[1].strip():
                    lines.discard(i)
            continue
        if s.startswith("/*"):
            lines.add(i)
            inside = "*/" not in s[2:]
        elif s.startswith("//"):
            lines.add(i)
    return lines


def code_lines(source, lang="python"):
    """(line_no, text) for lines that carry code: no blanks, comments or docstrings."""
    if lang != "python":
        skip = block_comment_lines(source)
        out = []
        for i, line in enumerate(source.splitlines(), start=1):
            s = line.strip()
            # package/import lines and lone braces carry no behaviour
            if not s or i in skip or s in ("{", "}", "};") or s.startswith(("package ", "import ", "#include")):
                continue
            out.append((i, line))
        return out
    skip = docstring_lines(source)
    out = []
    for i, line in enumerate(source.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in skip:
            continue
        out.append((i, line))
    return out


def faulty_lines(buggy_src, correct_src, lang="python"):
    """buggy-file line numbers touched by the fix, compared on code lines only."""
    b = code_lines(buggy_src, lang)
    c = code_lines(correct_src, lang)
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
