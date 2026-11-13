import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import benchmark, train_pipeline
from src.acr_repair import DISCOVERED, REFERENCE, build_context, llm_provider, repair, verify_with_pytest
from src.ast_metrics import FEATURE_LABELS, FEATURES
from src.line_localizer import load_token_model, rank_lines
from src.quixbugs import BUGGY_DIR, faulty_lines, has_tests, list_programs, read_buggy, read_correct

ROOT = Path(__file__).resolve().parent
MODEL_CHOICES = {"XGBoost (JM1/Halstead)": "xgb", "Random Forest": "rf"}

CUSTOM_CODE = '''def find_max(values):
    best = values[0]
    for i in range(1, len(values) - 1):
        if values[i] > best:
            best = values[i]
    return best
'''
CUSTOM_TESTS = '''assert find_max([1, 2, 3]) == 3
assert find_max([5, 1, 9]) == 9
assert find_max([4]) == 4
'''

st.set_page_config(page_title="PS12 code intelligence", layout="wide")


@st.cache_resource
def get_model(kind, stamp):
    return train_pipeline.load_model(kind)


@st.cache_resource
def get_token_model(stamp):
    return load_token_model()


def model_stamp():
    files = list(train_pipeline.MODEL_FILES.values()) + [train_pipeline.METRICS_PATH]
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


def retrain():
    proc = subprocess.run([sys.executable, "-m", "src.train_pipeline"], cwd=ROOT, capture_output=True, text=True)
    st.cache_resource.clear()
    return proc


def run_block(title, run):
    status = "PASS" if run["passed"] else ("TIMEOUT" if run["timed_out"] else "FAIL")
    st.markdown(f"**{title}: {status}**")
    st.caption(f"exit code {run['exit_code']}, {run['runtime_ms']:.0f} ms, python -m {run['command']}")
    st.code((run["stdout"] + run["stderr"]).strip() or "(no output)", language="text")


st.sidebar.header("Input")
mode = st.sidebar.radio("Source", ["QuixBugs program", "Custom Python Input"])
program = None
tests = None
if mode == "QuixBugs program":
    program = st.sidebar.selectbox("File in vendor/QuixBugs/python_programs", list_programs())
    code = read_buggy(program)
    st.sidebar.caption(str((BUGGY_DIR / f"{program}.py").relative_to(ROOT)))
else:
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

st.title("Defect prediction, bug localization and program repair")

if mode == "Custom Python Input":
    left, right = st.columns(2)
    code = left.text_area("Python source", CUSTOM_CODE, height=220)
    tests = right.text_area("Assert tests (one per line)", CUSTOM_TESTS, height=220)

if not train_pipeline.METRICS_PATH.exists():
    st.warning("No trained models found. Use the sidebar button or run: python -m src.train_pipeline")
    st.stop()

try:
    compile(code, "<input>", "exec")
except SyntaxError as exc:
    st.error(f"Input does not parse: {exc}")
    st.stop()

stamp = model_stamp()
metrics = train_pipeline.load_metrics()
ranked = rank_lines(code, model=get_token_model(stamp), exclude_program=program)

tab_pred, tab_loc, tab_fix, tab_bench = st.tabs([
    "Defect Prediction & SHAP",
    "Line Localization (DeepLineDP + AST)",
    "Program Repair & Pytest Sandbox (AutoCodeRover)",
    "Benchmark Evaluation",
])

with tab_pred:
    info = train_pipeline.explain_file(code, model_kind, model=get_model(model_kind, stamp))
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
    if program:
        side = "holdout" if program in metrics["test_programs"] else "training"
        st.caption(f"{program} is in the {side} programs of this split.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("JM1 and Python AST metrics")
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
        fig, ax = plt.subplots(figsize=(7, 7.5))
        ax.barh(shap_s.index, shap_s.values, color=["#b3412f" if v > 0 else "#2f6db3" for v in shap_s.values])
        ax.axvline(0, color="black", linewidth=0.8)
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
            truth = faulty_lines(code, read_correct(program))
            hit = [r["rank"] for r in ranked if r["line_no"] in truth]
            st.caption(f"Reference fix touches line(s) {truth}. Best rank of a fixed line: {min(hit) if hit else 'n/a'}. "
                       "Token weights were refitted without this program before scoring it.")

        st.subheader("Per-line suspicion")
        by_line = frame.sort_values("line_no")[
            ["line_no", "code", "suspicion_score", "token_score", "ast_score", "ast_node_type", "signals"]]
        styled = by_line.style.background_gradient(subset=["suspicion_score"], cmap="Reds", vmin=0.0, vmax=1.0)
        styled = styled.format({"suspicion_score": "{:.3f}", "token_score": "{:.3f}", "ast_score": "{:.3f}"})
        st.dataframe(styled, hide_index=True, width="stretch", height=min(700, 38 * (len(by_line) + 1)))

        st.subheader("AST context slice")
        ctx = build_context(code, ranked[0]["line_no"])
        a, b = st.columns(2)
        a.json({k: ctx.get(k) for k in ("fault_line", "fault_code", "class", "function", "signature", "arguments", "calls")})
        b.dataframe(pd.DataFrame(ctx.get("local_assignments", [])), hide_index=True, width="stretch")
        st.code(ctx.get("snippet", ""), language="text")

with tab_fix:
    can_test = bool(program and has_tests(program)) or bool(tests and tests.strip())
    use_llm = st.checkbox("Use LLM when an API key is set", value=bool(provider), disabled=not provider)
    if st.button("Generate patch and run pytest"):
        with st.spinner("searching for a patch"):
            fix = repair(code, program_name=program, assert_block=tests, ranked=ranked, use_llm=use_llm, timeout=timeout)
        st.session_state["fix"] = (code, fix)
        if fix["patched"]:
            with st.spinner("running pytest on both versions"):
                st.session_state["runs"] = verify_with_pytest(program, code, fix["patched"], tests, timeout) if can_test else None

    saved = st.session_state.get("fix")
    if saved and saved[0] == code:
        fix = saved[1]
        st.text("\n".join(fix["log"]))
        if not fix["patched"]:
            st.warning("No patch produced. Add assert tests, or set an API key for the LLM path.")
        else:
            if fix["origin"] == REFERENCE:
                st.warning("REFERENCE FALLBACK. The tool did not find this patch. It is the known fix copied from "
                           "vendor/QuixBugs/correct_python_programs and does not count as a repair.")
            elif fix["validated"]:
                st.success(f"DISCOVERED by {fix['source']} and validated by pytest. No reference file was used.")
            else:
                st.info(f"DISCOVERED by {fix['source']}, but there were no tests to validate it.")
            st.caption(f"Suspect lines sent to repair: {fix['fault_lines']}.")
            st.subheader("Unified diff")
            st.code(fix["diff"] or "(no textual change)", language="diff")
            a, b = st.columns(2)
            a.subheader("Original")
            a.code(code, language="python", line_numbers=True)
            b.subheader("Patched")
            b.code(fix["patched"], language="python", line_numbers=True)

            st.subheader("Pytest sandbox")
            runs = st.session_state.get("runs")
            if not runs:
                st.info("No tests available for this input, so the patch was not executed.")
            else:
                a, b = st.columns(2)
                with a:
                    run_block("Original", runs["before"])
                with b:
                    run_block("Patched", runs["after"])
    else:
        st.info("Patch generation order: LLM (if a key is set), validated single-edit AST search, "
                "then the QuixBugs reference fix as a fallback. A fallback is labelled as such and is not a repair.")

with tab_bench:
    names = list_programs()
    picked = st.multiselect("Programs", names, default=names[:8])
    c1, c2 = st.columns(2)
    search = c1.checkbox("Try single-edit AST search before the reference fix", value=True)
    bench_llm = c2.checkbox("Use LLM in benchmark", value=False, disabled=not provider)
    if st.button("Run benchmark") and picked:
        bar = st.progress(0.0)
        df = benchmark.run_benchmark(picked, use_llm=bench_llm, use_search=search, timeout=timeout,
                                     progress=lambda i, n, name: bar.progress(i / n, text=name))
        st.session_state["bench"] = df
    df = st.session_state.get("bench")
    if df is not None:
        s = benchmark.summarize(df)
        n = s["programs"]
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
        )
        st.subheader("Patch origin breakdown")
        st.dataframe(benchmark.summarize_origins(df), hide_index=True, width="stretch")

        def shade(row):
            colour = {DISCOVERED: "background-color: #d9ead3", REFERENCE: "background-color: #fff2cc"}
            return [colour.get(row["patch_origin"], "") if col in ("patch_origin", "repair_source") else "" for col in row.index]

        st.dataframe(df.style.apply(shade, axis=1), hide_index=True, width="stretch")
        st.caption("Green = discovered by the tool. Yellow = reference fallback. Line hits use leave-one-out token weights.")
