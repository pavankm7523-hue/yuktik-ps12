import re
import difflib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import benchmark, train_pipeline
from src.acr_repair import DISCOVERED, REFERENCE, build_context, can_execute, llm_provider, repair, verify_with_pytest
from src.ast_metrics import FEATURE_LABELS, FEATURES
from src.line_localizer import load_token_model, rank_lines
from src.quixbugs import EXT, faulty_lines, list_programs, program_path, read_buggy, read_correct
from src.sandbox import _balanced, run_tests, toolchain

ROOT = Path(__file__).resolve().parent
MODEL_CHOICES = {"XGBoost (JM1/Halstead)": "xgb", "Random Forest": "rf"}

LANGS = {"Python": "python", "Java": "java", "JavaScript": "javascript", "C": "c", "C++": "cpp"}
QUIX_TREES = {"Python (vendor/QuixBugs/python_programs)": "python", "Java (vendor/QuixBugs/java_programs)": "java"}
HIGHLIGHT = {"python": "python", "java": "java", "javascript": "javascript", "c": "c", "cpp": "cpp"}

# one buggy snippet and matching asserts per language, each with the same off-by-one loop bound
SAMPLES = {
    "python": ('''def find_max(values):
    best = values[0]
    for i in range(1, len(values) - 1):
        if values[i] > best:
            best = values[i]
    return best
''', '''assert find_max([1, 2, 3]) == 3
assert find_max([5, 1, 9]) == 9
assert find_max([4]) == 4
'''),
    "java": ('''public class MaxFinder {
    public static int findMax(int[] values) {
        int best = values[0];
        for (int i = 1; i < values.length - 1; i++) {
            if (values[i] > best) {
                best = values[i];
            }
        }
        return best;
    }
}
''', '''assert MaxFinder.findMax(new int[]{1, 2, 3}) == 3;
assert MaxFinder.findMax(new int[]{5, 1, 9}) == 9;
assert MaxFinder.findMax(new int[]{4}) == 4;
'''),
    "javascript": ('''function findMax(values) {
  let best = values[0];
  for (let i = 1; i < values.length - 1; i++) {
    if (values[i] > best) {
      best = values[i];
    }
  }
  return best;
}
''', '''assert.strictEqual(findMax([1, 2, 3]), 3);
assert.strictEqual(findMax([5, 1, 9]), 9);
assert.strictEqual(findMax([4]), 4);
'''),
    "c": ('''int find_max(const int *values, int n) {
    int best = values[0];
    for (int i = 1; i < n - 1; i++) {
        if (values[i] > best) {
            best = values[i];
        }
    }
    return best;
}
''', '''int a[] = {1, 2, 3};
int b[] = {5, 1, 9};
int c[] = {4};
assert(find_max(a, 3) == 3);
assert(find_max(b, 3) == 9);
assert(find_max(c, 1) == 4);
'''),
    "cpp": ('''#include <vector>

int findMax(const std::vector<int>& values) {
    int best = values[0];
    for (size_t i = 1; i < values.size() - 1; i++) {
        if (values[i] > best) {
            best = values[i];
        }
    }
    return best;
}
''', '''assert(findMax({1, 2, 3}) == 3);
assert(findMax({5, 1, 9}) == 9);
assert(findMax({4}) == 4);
'''),
}


st.set_page_config(page_title="PS12 code intelligence", layout="wide")


@st.cache_resource
def get_model(kind, stamp):
    return train_pipeline.load_model(kind)


@st.cache_resource
def get_explainer(kind, stamp):
    # building a tree explainer walks every tree, so it is built once per model file
    return train_pipeline.make_explainer(get_model(kind, stamp))


@st.cache_data(max_entries=256, show_spinner=False)
def get_explanation(code, kind, lang, stamp):
    return train_pipeline.explain_file(code, kind, model=get_model(kind, stamp), lang=lang,
                                       explainer=get_explainer(kind, stamp))


@st.cache_resource
def get_token_model(stamp):
    return load_token_model()


def model_stamp():
    files = list(train_pipeline.MODEL_FILES.values()) + [train_pipeline.METRICS_PATH]
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


def retrain():
    proc = subprocess.run([sys.executable, "-m", "src.train_pipeline"], cwd=ROOT, capture_output=True, text=True)
    st.cache_resource.clear()
    st.cache_data.clear()
    return proc


@st.cache_data(max_entries=128, show_spinner=False)
def first_run(code, program, tests, timeout, lang):
    """the failing run of the input as it is. its output guides localization and fills the status card."""
    return run_tests(code, program, tests, timeout=timeout, fail_fast=True, lang=lang)


# one-click judge presets for custom input mode: (python source, asserts)
QUICK_LOADS = {
    "Preset 1: Off-by-One Loop Bug": ('''def sum_up_to(n):
    total = 0
    for i in range(n):
        total += i
    return total
''', '''assert sum_up_to(0) == 0
assert sum_up_to(1) == 1
assert sum_up_to(3) == 6
assert sum_up_to(10) == 55
'''),
    "Preset 2: Mutable Default Bug": ('''def append_item(item, items=[]):
    items.append(item)
    return items
''', '''assert append_item(1) == [1]
assert append_item(2) == [2]
assert append_item(3, [0]) == [0, 3]
'''),
    # forgets n -= 1, so the sandbox has to kill it at the test budget
    "Preset 3: Infinite Loop Sandbox Stress-Test": ('''def countdown_sum(n):
    total = 0
    while n > 0:
        total += n
    return total
''', '''assert countdown_sum(0) == 0
assert countdown_sum(3) == 6
assert countdown_sum(5) == 15
'''),
}


def quick_load(name):
    """runs as a button callback, before the widgets are built, so the editors pick up the new text."""
    source, asserts = QUICK_LOADS[name]
    st.session_state["lang_select"] = "Python"
    st.session_state["code_python"] = source
    st.session_state["tests_python"] = asserts
    st.session_state["preset"] = name
    for key in ("fix", "runs", "timing"):
        st.session_state.pop(key, None)


def patch_path(program, lang):
    if program:
        return program_path(program, lang).relative_to(ROOT).as_posix()
    return f"candidate{EXT[lang]}"


def git_patch(before, after, path):
    """unified diff with a diff --git header and no-newline markers, so `git apply` accepts it."""
    out = [f"diff --git a/{path} b/{path}\n"]
    for line in difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                     fromfile=f"a/{path}", tofile=f"b/{path}", n=3):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out) if len(out) > 1 else ""


def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)


def run_summary(run):
    if not run:
        return None
    return {k: run.get(k) for k in ("passed", "timed_out", "exit_code", "runtime_ms", "command")}


def run_status(run):
    if run["passed"] is None:
        return "NOT RUN"
    return "PASS" if run["passed"] else ("TIMEOUT" if run["timed_out"] else "FAIL")


def run_block(col, title, run):
    status = run_status(run)
    banner = {"PASS": col.success, "NOT RUN": col.warning}.get(status, col.error)
    banner(f"{title}: {status}")
    col.caption(f"exit code {run['exit_code']}, {run['runtime_ms']:.0f} ms, {run['command']}")
    col.code((run["stdout"] + run["stderr"]).strip() or "(no output)", language="text")


# the theme lives in .streamlit/config.toml. this only tightens the top padding and the tab labels.
st.markdown("""<style>
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
.stTabs [data-baseweb="tab"] { padding: 0.5rem 0; }
.stTabs [data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 600; }
</style>""", unsafe_allow_html=True)

st.sidebar.header("Input")
mode = st.sidebar.radio("Source", ["QuixBugs program", "Custom Input"])
program = None
tests = None
if mode == "QuixBugs program":
    tree = st.sidebar.radio("QuixBugs tree", list(QUIX_TREES))
    lang = QUIX_TREES[tree]
    program = st.sidebar.selectbox("Program", list_programs(lang))
    code = read_buggy(program, lang)
    st.sidebar.caption(str(program_path(program, lang).relative_to(ROOT)))
else:
    lang = LANGS[st.sidebar.selectbox("Language", list(LANGS), key="lang_select")]
    code = None

model_label = st.sidebar.selectbox("Classifier", list(MODEL_CHOICES))
model_kind = MODEL_CHOICES[model_label]
timeout = st.sidebar.number_input("Test budget per pytest run (s)", min_value=1.0, max_value=30.0, value=3.0, step=0.5)

st.sidebar.header("Training")
if st.sidebar.button("Re-run src/train_pipeline.py"):
    with st.spinner("rebuilding verified Python variants, grouped metrics, and the full-data model"):
        result = retrain()
    st.sidebar.code((result.stdout + result.stderr).strip()[-1500:], language="text")

provider = llm_provider()
st.sidebar.header("Repair backend")
st.sidebar.text(f"LLM: {provider}" if provider else "LLM: none (no API key in env)")
st.sidebar.text("python: fast worker pool\n" + "\n".join(
    f"{k}: {'found' if v else 'not on PATH, static check only'}" for k, v in toolchain().items()))

st.title("Defect prediction, bug localization and program repair")

if mode == "Custom Input":
    sample_code, sample_tests = SAMPLES[lang]
    label = next(k for k, v in LANGS.items() if v == lang)
    st.caption("Judge presets (Python): load a known bug and its assert tests into the editors.")
    for col, name in zip(st.columns(len(QUICK_LOADS)), QUICK_LOADS):
        col.button(f"[{name}]", key=f"preset_{name}", on_click=quick_load, args=(name,), width="stretch")
    left, right = st.columns(2)
    # keyed per language so switching the selector loads that language's template
    st.session_state.setdefault(f"code_{lang}", sample_code)
    st.session_state.setdefault(f"tests_{lang}", sample_tests)
    code = left.text_area(f"{label} source", height=260, key=f"code_{lang}")
    tests = right.text_area("Assert tests (one per line)", height=260, key=f"tests_{lang}")

if not train_pipeline.METRICS_PATH.exists():
    st.warning("No trained models found. Use the sidebar button or run: python -m src.train_pipeline")
    st.stop()

if lang == "python":
    try:
        compile(code, "<input>", "exec")
    except SyntaxError as exc:
        st.error(f"Input does not parse: {exc}")
        st.stop()
elif not _balanced(code):
    st.error("Input has unbalanced brackets.")
    st.stop()

stamp = model_stamp()
metrics = train_pipeline.load_metrics()
test_program = program if lang == "python" else None
can_test = can_execute(lang, test_program, tests)
baseline = first_run(code, test_program, tests, timeout, lang) if can_test else None
failure = None
if baseline and not baseline["passed"]:
    failure = baseline["stdout"] + "\n" + baseline["stderr"]
ranked = rank_lines(code, model=get_token_model(stamp), exclude_program=test_program, lang=lang,
                    failure_output=failure)
info = get_explanation(code, model_kind, lang, stamp)

prob = info["probability"]
suspect = ranked[0] if ranked else None
m1, m2, m3, m4 = st.columns(4)
m1.metric("Defect probability", f"{prob:.3f}")
m1.caption(f"{model_label}, " + ("high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low") + " risk")
m2.metric("Top suspect line", f"L{suspect['line_no']}" if suspect else "n/a")
m2.caption((f"score {suspect['suspicion_score']:.3f}, " if suspect else "")
           + ("traceback guided" if failure else "static signals"))
m3.metric("Tests on input", run_status(baseline) if baseline else "n/a")
m3.caption(f"{baseline['runtime_ms']:.0f} ms" if baseline else "no runnable tests")
m4.metric("Repair backend", provider or "AST search")
m4.caption("llm then single-edit search" if provider else "no API key, single-edit search only")

tab_pred, tab_loc, tab_fix, tab_bench, tab_adr = st.tabs([
    "Defect Prediction & SHAP",
    "Line Localization (DeepLineDP + AST)",
    "Program Repair & Test Sandbox (AutoCodeRover)",
    "Benchmark Evaluation",
    "Architecture & ADR",
])

with tab_pred:
    if lang != "python":
        st.info("The classifiers were trained on Python AST metrics. For this language the same 29 columns come from "
                "a lexical extractor (Halstead tokens, McCabe branch keywords, brace depth), so the probability is "
                "a model applied outside the language it was trained on.")
    holdout = metrics["models"][model_kind]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Defect probability", f"{info['probability']:.3f}")
    c2.metric("Holdout precision", f"{holdout['precision']:.3f}")
    c3.metric("Holdout recall", f"{holdout['recall']:.3f}")
    c4.metric("Holdout F1", f"{holdout['f1']:.3f}")
    c5.metric("Holdout ROC-AUC", f"{holdout['roc_auc']:.3f}")
    pairs, variants, base = holdout["original_pairs"], holdout["generated_variants"], holdout.get("jm1_baseline")
    st.caption(
        f"{model_label}, trained on {metrics['train_rows']} rows of Python AST metrics ({metrics['training_data']}) and "
        f"scored on {metrics['test_rows']} rows from {len(metrics['test_programs'])} programs it never saw in any variant. "
        f"Accuracy {holdout['accuracy']:.3f}. The deployed model is refit on all {metrics['final_training_rows']} rows "
        f"across {metrics['final_training_programs']} programs. Grouped {metrics['cv_folds']}-fold out-of-fold "
        f"ROC-AUC {holdout['group_cv']['roc_auc']:.3f}, recall {holdout['group_cv']['recall']:.3f}."
    )
    breakdown = pd.DataFrame([
        {"holdout slice": "all holdout rows", **holdout},
        {"holdout slice": "original QuixBugs buggy/correct pairs", **pairs},
        {"holdout slice": "generated mutants and refactors", **variants},
    ] + ([{"holdout slice": "same rows, model trained on NASA JM1 (C code)", **base}] if base else []))
    st.dataframe(breakdown[["holdout slice", "rows", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(3),
                 hide_index=True, width="stretch")
    st.caption("An original buggy file and its fix differ by one token and usually have identical metrics, so that "
               "slice is the hard one. Most of the signal comes from the generated variants.")
    if program and lang == "python":
        side = "holdout" if program in metrics["test_programs"] else "training"
        st.caption(f"{program} is in the {side} programs of this split.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("JM1 and AST metrics" if lang == "python" else "JM1 metrics (lexical extractor)")
        table = pd.DataFrame({
            "metric": FEATURES,
            "meaning": [FEATURE_LABELS[f] for f in FEATURES],
            "value": [info["features"][f] for f in FEATURES],
            "shap": [round(info["shap"][f], 4) for f in FEATURES],
        })
        st.dataframe(table, hide_index=True, width="stretch", height=560)
    with right:
        st.subheader("SHAP attribution")
        shap_s = pd.Series(info["shap"]).sort_values(key=abs)
        fig, ax = plt.subplots(figsize=(7, 7.5), facecolor="white")
        ax.set_facecolor("white")
        ax.barh(shap_s.index, shap_s.values, color="#EA580C")
        ax.axvline(0, color="#18181B", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(f"SHAP value ({info['shap_units']}), positive pushes towards defective")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"TreeExplainer base value {info['base_value']:.3f} ({info['shap_units']}).")

        st.subheader("Holdout confusion matrix")
        cm = pd.DataFrame(holdout["confusion_matrix"], index=["actual clean", "actual defective"],
                          columns=["predicted clean", "predicted defective"])
        st.dataframe(cm)

with tab_loc:
    frame = pd.DataFrame(ranked)
    if frame.empty:
        st.info("No code lines to score.")
    else:
        st.subheader("Top 3 suspicious lines")
        st.dataframe(frame.head(3)[["rank", "line_no", "code", "suspicion_score", "ast_node_type", "signals"]],
                     hide_index=True, width="stretch")
        if program:
            truth = faulty_lines(code, read_correct(program, lang), lang)
            hit = [r["rank"] for r in ranked if r["line_no"] in truth]
            st.caption(f"Reference fix touches line(s) {truth}. Best rank of a fixed line: {min(hit) if hit else 'n/a'}. "
                       + ("Token weights were refitted without this program before scoring it." if lang == "python"
                          else "Token weights come from the Python programs; the Java rules add the language signals."))

        if failure:
            st.caption("The failing test run boosted lines named in the traceback and lines matching the "
                       "exception type.")
        st.subheader("Per-line suspicion")
        by_line = frame.sort_values("line_no")[
            ["line_no", "code", "suspicion_score", "token_score", "ast_score", "ast_node_type", "signals"]]
        st.dataframe(by_line, hide_index=True, width="stretch", height=min(700, 38 * (len(by_line) + 1)),
                     column_config={
                         "suspicion_score": st.column_config.ProgressColumn("Suspicion", min_value=0.0, max_value=1.0,
                                                                            format="%.2f"),
                         "token_score": st.column_config.NumberColumn(format="%.3f"),
                         "ast_score": st.column_config.NumberColumn(format="%.3f"),
                     })

        st.subheader("AST context slice")
        ctx = build_context(code, ranked[0]["line_no"], lang=lang)
        a, b = st.columns(2)
        a.json({k: ctx.get(k) for k in ("fault_line", "fault_code", "class", "function", "signature", "arguments", "calls")})
        b.dataframe(pd.DataFrame(ctx.get("local_assignments", [])), hide_index=True, width="stretch")
        slice_lines = ctx.get("snippet", "").splitlines()
        numbered = [re.match(r"\s*(\d+) ?(.*)$", line) for line in slice_lines]
        if numbered and all(numbered) and numbered[0].group(1) == "1":
            st.code("\n".join(m.group(2) for m in numbered), language=HIGHLIGHT[lang], line_numbers=True)
        else:
            # the slice starts mid-file and already carries its real line numbers
            st.code("\n".join(slice_lines), language=HIGHLIGHT[lang])

with tab_fix:
    if not can_test and lang != "python":
        st.info(f"No runnable tests for {lang} on this machine "
                + ("(add assert lines)." if not (tests or "").strip() else "(compiler or runtime not on PATH).")
                + " Patches are still proposed; the sandbox falls back to a static bracket check.")
    use_llm = st.checkbox("Use LLM when an API key is set", value=bool(provider), disabled=not provider)
    if st.button("Generate patch and run pytest"):
        st.session_state.pop("runs", None)
        start = time.perf_counter()
        with st.spinner("searching for a patch"):
            fix = repair(code, program_name=program, assert_block=tests, ranked=ranked, use_llm=use_llm,
                         timeout=timeout, lang=lang, failure_output=failure)
        repair_ms = (time.perf_counter() - start) * 1000.0
        st.session_state["fix"] = (code, fix)
        start = time.perf_counter()
        if fix["patched"]:
            with st.spinner("running the tests on both versions"):
                st.session_state["runs"] = verify_with_pytest(program, code, fix["patched"], tests, timeout, lang)
        elif can_test:
            # no patch, but still run the original in the sandbox so a hang or failure is visible
            with st.spinner("running the tests on the original"):
                st.session_state["runs"] = {"before": run_tests(code, test_program, tests, timeout, False, lang),
                                            "after": None}
        st.session_state["timing"] = {"repair_search_ms": round(repair_ms, 1),
                                      "verification_wall_ms": round((time.perf_counter() - start) * 1000.0, 1)}

    def audit_record(fix, runs):
        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": {"mode": mode, "program": program, "language": lang, "path": patch_path(program, lang),
                      "preset": st.session_state.get("preset") if mode == "Custom Input" else None},
            "classifier": {"model": model_label, "defect_probability": round(float(info["probability"]), 6)},
            "jm1_metrics": {f: info["features"][f] for f in FEATURES},
            "top3_suspicious_lines": [{k: r[k] for k in ("rank", "line_no", "code", "suspicion_score", "token_score",
                                                         "ast_score", "ast_node_type", "signals") if k in r}
                                      for r in ranked[:3]],
            "patch": {"origin": fix["origin"], "source": fix["source"], "validated_by_tests": fix["validated"],
                      "counts_as_autonomous_repair": fix["origin"] == DISCOVERED and bool(fix["validated"]),
                      "fault_lines_sent_to_repair": fix["fault_lines"], "log": fix["log"]},
            "pytest_execution_ms": {"test_budget_s": timeout, **st.session_state.get("timing", {}),
                                    "original": run_summary((runs or {}).get("before")),
                                    "patched": run_summary((runs or {}).get("after"))},
        }

    saved = st.session_state.get("fix")
    if saved and saved[0] == code:
        fix = saved[1]
        runs = st.session_state.get("runs")
        st.text("\n".join(fix["log"]))
        if not fix["patched"]:
            if lang != "python" and not can_test:
                st.warning(f"No patch produced. Candidate edits need a {lang} compiler or runtime to be checked "
                           "against the asserts, or an API key for the LLM path.")
            else:
                st.warning("No patch produced. Add assert tests, or set an API key for the LLM path.")
            if runs and runs.get("before"):
                st.subheader("Test sandbox")
                run_block(st, "Original", runs["before"])
                if runs["before"]["timed_out"]:
                    st.caption(f"The sandbox killed the run after the {timeout:.1f}s budget. The Streamlit process "
                               "was not blocked: tests execute in a separate interpreter.")
            st.download_button("Download Verification Audit (.json)",
                               json.dumps(audit_record(fix, runs), indent=2, default=_plain),
                               file_name="verification_audit.json", mime="application/json")
        else:
            if fix["origin"] == REFERENCE:
                st.warning("REFERENCE FALLBACK. The tool did not find this patch. It is the known fix copied from "
                           f"vendor/QuixBugs/correct_{lang}_programs and does not count as a repair.")
            elif fix["validated"]:
                st.success(f"DISCOVERED by {fix['source']} and validated by the tests. No reference file was used.")
            else:
                st.info(f"DISCOVERED by {fix['source']}, but there were no tests to validate it.")
            st.caption(f"Suspect lines sent to repair: {fix['fault_lines']}.")
            st.subheader("Unified diff")
            st.code(fix["diff"] or "(no textual change)", language="diff")
            path = patch_path(program, lang)
            d1, d2 = st.columns(2)
            d1.download_button("Download Unified Patch (.patch)", git_patch(code, fix["patched"], path),
                               file_name=f"{Path(path).stem}.patch", mime="text/x-diff", width="stretch",
                               help=f"Apply from the repository root with: git apply {Path(path).stem}.patch")
            d2.download_button("Download Verification Audit (.json)",
                               json.dumps(audit_record(fix, runs), indent=2, default=_plain),
                               file_name="verification_audit.json", mime="application/json", width="stretch")
            a, b = st.columns(2)
            a.subheader("Original")
            a.code(code, language=HIGHLIGHT[lang], line_numbers=True)
            b.subheader("Patched")
            b.code(fix["patched"], language=HIGHLIGHT[lang], line_numbers=True)

            st.subheader("Test sandbox")
            if not runs:
                st.info("No tests available for this input, so the patch was not executed.")
            else:
                a, b = st.columns(2)
                run_block(a, "Original", runs["before"])
                run_block(b, "Patched", runs["after"])
    else:
        st.info("Patch generation order: LLM (if a key is set), validated single-edit AST search, each retried once "
                "with the failing test output, then the QuixBugs reference fix as a fallback. A fallback is labelled as such and is not a repair.")

with tab_bench:
    bench_lang = lang if lang in ("python", "java") else "python"
    names = list_programs(bench_lang)
    st.caption(f"Benchmark over vendor/QuixBugs/{bench_lang}_programs. Programs run in parallel "
               f"({benchmark.WORKERS} workers) and results are cached in data/benchmark_cache.json by "
               "program, language, code sha1 and llm backend.")
    picked = st.multiselect("Programs", names, default=names[:8], key=f"bench_{bench_lang}")
    c1, c2, c3 = st.columns(3)
    search = c1.checkbox("Try single-edit search before the reference fix", value=True)
    bench_llm = c2.checkbox("Use LLM in benchmark", value=False, disabled=not provider)
    use_cache = c3.checkbox("Use cached benchmark results (instant) / Force fresh run", value=True)
    if st.button("Run benchmark") and picked:
        bar = st.progress(0.0)
        df = benchmark.run_benchmark(picked, use_llm=bench_llm, use_search=search, timeout=timeout, lang=bench_lang,
                                     use_cache=use_cache, progress=lambda i, n, name: bar.progress(i / n, text=name))
        st.session_state["bench"] = df
    df = st.session_state.get("bench")
    if df is not None:
        s = benchmark.summarize(df)
        n = s["programs"]
        cached = int(df["cached"].sum())
        st.caption(f"{cached} rows from cache, {n - cached} computed fresh in {df.attrs.get('seconds', 0.0):.2f}s.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Top-1 line hit", f"{s['top1_hit']}/{n}")
        c2.metric("Top-3 line hit", f"{s['top3_hit']}/{n}")
        c3.metric("Discovered and passing", f"{s['discovered_fix']}/{n}")
        c4.metric("Reference fallback", f"{s['reference_fallback']}/{n}")
        c5.metric("Unresolved", f"{s['unresolved']}/{n}")
        st.caption(
            f"Dynamic patches found: {s['discovered_by_ast_search']} by AST search and {s['discovered_by_llm']} by LLM; "
            f"{s['discovered_fix']} passed independent pytest verification. The reference file was used for "
            f"{s['reference_fallback']} programs ({s['reference_fallback_passes']} passed pytest); these fallbacks "
            "are excluded from discovered-repair counts."
            + (f" {s['not_executed']} rows were not executed: no runnable tests for this language on this machine."
               if s["not_executed"] else "")
        )
        st.subheader("Patch origin breakdown")
        st.dataframe(benchmark.summarize_origins(df), hide_index=True, width="stretch")

        def shade(row):
            colour = {DISCOVERED: "background-color: #d9ead3", REFERENCE: "background-color: #fff2cc"}
            return [colour.get(row["patch_origin"], "") if col in ("patch_origin", "repair_source") else ""
                    for col in row.index]

        st.dataframe(df.style.apply(shade, axis=1), hide_index=True, width="stretch")
        st.caption("Green = discovered by the tool. Yellow = reference fallback. Python line hits use leave-one-out "
                   "token weights.")

with tab_adr:
    st.subheader("Pipeline")
    st.graphviz_chart('''
digraph pipeline {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded", fontname="Helvetica", fontsize=11, color="#A1A1AA", fontcolor="#18181B",
          margin="0.18,0.10"];
    edge [color="#EA580C", arrowsize=0.7, fontname="Helvetica", fontsize=9, fontcolor="#555555"];
    src [label="Source file\n(QuixBugs or custom)", shape=note];
    s1 [label="1. AST Telemetry\n18 JM1 columns (Halstead, McCabe, LOC)\n+ 11 Python AST structure counts"];
    s2 [label="2. XGBoost / Random Forest\n+ SHAP TreeExplainer\nfile-level defect probability"];
    s3 [label="3. DeepLineDP Token Localizer\n+ AST rule signals\nTop-3 suspicious lines"];
    s4 [label="4. AST Context Slicer\n+ Isolated Pytest Subprocess\npatch search, per-run time budget"];
    out [label="Verified patch + audit\n(.patch, .json)", shape=note];
    src -> s1 -> s2 -> s3 -> s4 -> out;
    s4 -> s4 [label="  candidate edits"];
}
''', width="stretch")
    st.caption("Stages 1 to 3 are static and deterministic. Stage 4 is the only place code executes, and it runs "
               "in a separate interpreter that is killed when it exceeds the test budget.")

    st.subheader("Architecture Decision Record")
    st.markdown('''
| ID | Decision | Context | Choice | Trade-off accepted |
|---|---|---|---|---|
| ADR-1 | Hybrid AST + ML instead of an end-to-end LLM wrapper | A raw LLM wrapper returns a patch with no evidence for where the bug is, no reproducible score, and no guarantee the patch runs. Output changes between calls. | Deterministic AST features feed tree ensembles whose SHAP values explain each score. The localizer narrows repair to three lines and the AST slicer sends only that context. Every patch, including an optional LLM patch, must pass tests in the sandbox before it is accepted. | Single-edit search cannot express large multi-line rewrites. The optional LLM step covers those, and it is held to the same test gate. |
| ADR-2 | Bridge the C-to-Python NASA JM1 domain shift | JM1 metrics were extracted from C modules. Python has different operators, no braces, and different idioms, so a model fit on C rows does not transfer. | Re-implement the 18 JM1 column definitions over the Python AST so the schema is shared, add 11 Python-only structure counts, and train on QuixBugs buggy/correct pairs plus test-verified mutants and refactors. Splits are grouped by program so no variant of a holdout program is seen in training. The JM1-on-C model is kept only as a reported baseline on the same holdout. | The Python training set is far smaller than JM1, and original buggy/fixed pairs often share identical metrics. The holdout table reports that hard slice separately instead of hiding it in the average. |
| ADR-3 | Separate autonomous repairs from reference fallbacks | For QuixBugs programs the known fix exists on disk. Copying it would make the repair rate look close to 100 percent. | A patch is DISCOVERED only when the LLM or AST search produced it without the reference file. The reference fix is labelled REFERENCE FALLBACK, shown in a separate column, and excluded from discovered-repair counts in the benchmark and the audit JSON. | Headline repair numbers are lower, but every counted repair is reproducible and was verified by tests. |
''')
