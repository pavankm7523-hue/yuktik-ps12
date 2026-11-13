import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from src import mutations
from src.ast_metrics import FEATURES, JM1_FEATURES, extract_metrics
from src.line_localizer import TFIDF_PATH, save_token_model
from src.quixbugs import DATA_DIR, MODEL_DIR, ROOT, VENDOR, list_programs, read_buggy, read_correct
from src.sandbox import run_quixbugs_tests

CSV_PATH = DATA_DIR / "quixbugs_jm1_features.csv"
CACHE_PATH = DATA_DIR / "variant_test_cache.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
JM1_CSV = VENDOR / "jm1-defect-prediction" / "data" / "jm1.csv"
MODEL_FILES = {"xgb": MODEL_DIR / "xgb_model.pkl", "rf": MODEL_DIR / "rf_model.pkl"}

VARIANTS_PER_KIND = 4
CANDIDATES_PER_KIND = 8
FOLDS = 5


def _candidates(name, correct_src, seed):
    rng = random.Random(f"{seed}:{name}")
    seen = {correct_src.strip()}
    out = []
    for kind, make in (("mutant", mutations.inject_bug), ("refactor", mutations.refactor)):
        found, tries = 0, 0
        while found < CANDIDATES_PER_KIND and tries < 60:
            tries += 1
            src = make(correct_src, rng)
            if src is None or src.strip() in seen:
                continue
            seen.add(src.strip())
            out.append({"program": name, "kind": kind, "source": src})
            found += 1
    return out


def _key(cand):
    return hashlib.sha1(f"{cand['program']}\n{cand['source']}".encode("utf-8")).hexdigest()


def _confirm_all(cands, timeout, workers):
    # a mutant only counts as defective if the real quixbugs tests fail, a refactor only counts as
    # clean if they pass. verdicts are cached by source hash so a retrain does not rerun pytest.
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    todo = [c for c in cands if _key(c) not in cache]

    def check(cand):
        res = run_quixbugs_tests(cand["program"], cand["source"], timeout=timeout, fail_fast=True)
        return _key(cand), res["passed"]

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cache.update(pool.map(check, todo))
        DATA_DIR.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    for c in cands:
        c["tests_pass"] = cache[_key(c)]
    return len(todo)


def build_dataset(seed=42, verify=True, test_timeout=3.0, workers=6):
    rows = []
    cands = []
    for name in list_programs():
        correct = read_correct(name)
        rows.append({"program": name, "variant": "quixbugs_buggy", "source": read_buggy(name), "defects": 1})
        rows.append({"program": name, "variant": "quixbugs_correct", "source": correct, "defects": 0})
        cands.extend(_candidates(name, correct, seed))

    if verify:
        _confirm_all(cands, test_timeout, workers)

    kept = {}
    for c in cands:
        want_pass = c["kind"] == "refactor"
        if verify and c["tests_pass"] != want_pass:
            continue
        key = (c["program"], c["kind"])
        kept[key] = kept.get(key, 0) + 1
        if kept[key] > VARIANTS_PER_KIND:
            continue
        rows.append({
            "program": c["program"],
            "variant": f"{c['kind']}_{kept[key]}",
            "source": c["source"],
            "defects": 0 if want_pass else 1,
        })

    records = []
    for r in rows:
        feats = extract_metrics(r["source"])
        records.append({"program": r["program"], "variant": r["variant"], **feats, "defects": r["defects"]})
    df = pd.DataFrame(records)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def _score_probabilities(y, proba):
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= 0.5).astype(int)
    both = len(set(y)) == 2
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _score(model, X, y):
    return _score_probabilities(y, model.predict_proba(X)[:, 1])


def load_jm1():
    """the nasa jm1 rows shipped in vendor/jm1-defect-prediction, exact duplicate rows dropped."""
    df = pd.read_csv(JM1_CSV).drop_duplicates().reset_index(drop=True)
    df["defects"] = df["defects"].astype(int)
    return df


def _make_models():
    return {
        "xgb": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                     random_state=42, n_jobs=-1),
    }


def _jm1_baselines(jm1):
    # the previous setup: fit on c modules from nasa, apply to python files unchanged
    ratio = float((jm1["defects"] == 0).sum() / (jm1["defects"] == 1).sum())
    models = _make_models()
    models["xgb"].set_params(scale_pos_weight=ratio)
    for model in models.values():
        model.fit(jm1[JM1_FEATURES].astype(float), jm1["defects"])
    return models


def train(df):
    """Evaluate with program-grouped splits, then refit deployment models on all Python examples."""
    X = df[FEATURES].astype(float)
    y = df["defects"].astype(int)
    groups = df["program"]
    originals = df["variant"].str.startswith("quixbugs_")

    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError("QuixBugs training data must contain at least two distinct programs.")
    n_folds = min(FOLDS, n_groups)
    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y, groups))
    train_idx, test_idx = folds[0]
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    test_orig = originals.iloc[test_idx].to_numpy()

    baselines = _jm1_baselines(load_jm1()) if JM1_CSV.exists() else {}
    MODEL_DIR.mkdir(exist_ok=True)
    report = {
        "training_data": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
        "split": f"fold 1 of {FOLDS}, stratified by label and grouped by program",
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "final_training_rows": int(len(df)),
        "final_training_programs": n_groups,
        "cv_folds": n_folds,
        "test_programs": sorted(groups.iloc[test_idx].unique().tolist()),
        "defective_rows": int(y.sum()),
        "variants": df["variant"].str.replace(r"_\d+$", "", regex=True).value_counts().to_dict(),
        "features": FEATURES,
        "models": {},
    }
    for key, model in _make_models().items():
        holdout_model = clone(model).fit(X_train, y_train)
        m = _score(holdout_model, X_test, y_test)
        # the two halves of the holdout behave very differently, so they are reported apart
        m["original_pairs"] = _score(holdout_model, X_test[test_orig], y_test[test_orig])
        m["generated_variants"] = _score(holdout_model, X_test[~test_orig], y_test[~test_orig])

        aucs = []
        oof_probabilities = np.full(len(y), np.nan, dtype=float)
        for tr, te in folds:
            fold_model = clone(model).fit(X.iloc[tr], y.iloc[tr])
            probabilities = fold_model.predict_proba(X.iloc[te])[:, 1]
            oof_probabilities[te] = probabilities
            fold_auc = _score_probabilities(y.iloc[te], probabilities)["roc_auc"]
            if fold_auc is not None:
                aucs.append(fold_auc)
        m["group_cv"] = _score_probabilities(y, oof_probabilities)
        m["group_cv"]["folds"] = n_folds
        m["cv_roc_auc_mean"] = float(np.mean(aucs)) if aucs else None
        m["cv_roc_auc_std"] = float(np.std(aucs)) if aucs else None

        # The saved model is for inference, so use all available programs after evaluation.
        deployment_model = clone(model).fit(X, y)
        joblib.dump(deployment_model, MODEL_FILES[key], compress=3)

        if key in baselines:
            m["jm1_baseline"] = _score(baselines[key], X_test[JM1_FEATURES], y_test)
        report["models"][key] = m
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_model(model_type="xgb"):
    return joblib.load(MODEL_FILES[model_type])


def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _positive_class_shap(explainer, X):
    values = np.asarray(explainer.shap_values(X))
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    # random forest gives one slice per class, in either (class, row, feat) or (row, feat, class) order
    if values.ndim == 3:
        values = values[1] if values.shape[0] == 2 and values.shape[1] == len(X) else values[:, :, 1]
    return values[0], float(base[-1])


def explain_file(code_str, model_type="xgb", model=None):
    import shap

    model = model or load_model(model_type)
    feats = extract_metrics(code_str)
    X = pd.DataFrame([feats], columns=FEATURES).astype(float)
    proba = float(model.predict_proba(X)[0, 1])
    values, base = _positive_class_shap(shap.TreeExplainer(model), X)
    return {
        "probability": proba,
        "features": feats,
        "shap": {f: float(v) for f, v in zip(FEATURES, values)},
        "base_value": base,
        # xgboost shap values are in log-odds, random forest ones in probability
        "shap_units": "log-odds" if model_type == "xgb" else "probability",
    }


def _line(tag, m):
    auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
    return (f"  {tag:<22} rows={m['rows']:<4} acc={m['accuracy']:.3f} precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1']:.3f} roc_auc={auc} cm={m['confusion_matrix']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the quixbugs python feature csv and train both classifiers on it")
    ap.add_argument("--no-verify", action="store_true", help="skip running quixbugs tests on generated variants")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df = build_dataset(seed=args.seed, verify=not args.no_verify)
    report = train(df)
    tok = save_token_model()
    print(f"wrote {TFIDF_PATH} ({tok['n_lines']} code lines, {tok['n_faulty']} faulty)")
    print(f"wrote {CSV_PATH} ({report['rows']} rows, {report['defective_rows']} defective)")
    print(f"variants: {report['variants']}")
    print(f"split: {report['train_rows']} train / {report['test_rows']} holdout rows, {report['split']}")
    for key, m in report["models"].items():
        cv = m["group_cv"]
        print(f"{key}: grouped {cv['folds']}-fold roc_auc {m['cv_roc_auc_mean']:.3f} +/- "
              f"{m['cv_roc_auc_std']:.3f}, recall={cv['recall']:.3f}, f1={cv['f1']:.3f}")
        print(_line("holdout", m))
        print(_line("holdout original pairs", m["original_pairs"]))
        print(_line("holdout variants", m["generated_variants"]))
        if "jm1_baseline" in m:
            print(_line("jm1-trained baseline", m["jm1_baseline"]))


if __name__ == "__main__":
    main()
