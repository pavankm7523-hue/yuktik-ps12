import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.acr_repair import DISCOVERED, REFERENCE, can_execute, llm_provider, repair, verify_with_pytest
from src.line_localizer import load_token_model, rank_lines
from src.sandbox import run_tests
from src.quixbugs import DATA_DIR, faulty_lines, list_programs, read_buggy, read_correct

CACHE_PATH = DATA_DIR / "benchmark_cache.json"
WORKERS = 6
# bump when the row layout or the repair logic changes, old cache entries are then ignored
CACHE_VERSION = 3

_lock = threading.Lock()


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_key(program, lang, code, llm_backend, use_search=True):
    return "|".join([program, lang, _sha1(code), llm_backend or "none", "search" if use_search else "nosearch"])


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("rows", {}) if data.get("version") == CACHE_VERSION else {}


def save_cache(rows):
    DATA_DIR.mkdir(exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "rows": rows}, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def evaluate_program(name, use_llm=False, use_search=True, timeout=3.0, lang="python"):
    buggy = read_buggy(name, lang)
    truth = set(faulty_lines(buggy, read_correct(name, lang), lang))
    # leave-one-out: the token model never sees the program it is scored on (it is trained on python only)
    # the failing run of the buggy program feeds traceback-guided localization, as autocoderover does
    failure = None
    if can_execute(lang, name):
        first = run_tests(buggy, name, timeout=timeout, fail_fast=True, lang=lang)
        failure = "" if first["passed"] else (first["stdout"] + "\n" + first["stderr"])
    ranked = rank_lines(buggy, exclude_program=name if lang == "python" else None,
                        model=None if lang == "python" else load_token_model(), lang=lang,
                        failure_output=failure)
    order = [r["line_no"] for r in ranked]
    first_hit = next((i for i, no in enumerate(order, start=1) if no in truth), None)

    fix = repair(buggy, program_name=name, ranked=ranked, use_llm=use_llm, use_search=use_search,
                 timeout=timeout, lang=lang, failure_output=failure)
    if can_execute(lang, name):
        runs = verify_with_pytest(name, buggy, fix["patched"], timeout=timeout, lang=lang)
        before, after = runs["before"], runs["after"]
        original_fails, passed = not before["passed"], bool(after["passed"])
        before_ms, after_ms = before["runtime_ms"], after["runtime_ms"]
    else:
        # no runnable tests for this language on this machine: report it, do not guess
        original_fails, passed, before_ms, after_ms = None, None, None, None
    return {
        "program": name,
        "lang": lang,
        "fault_lines": ", ".join(map(str, sorted(truth))),
        "top3_predicted": ", ".join(map(str, order[:3])),
        "rank_of_fault": first_hit,
        "top1_hit": first_hit == 1,
        "top3_hit": first_hit is not None and first_hit <= 3,
        "repair_source": fix["source"],
        "patch_origin": fix["origin"],
        "original_fails": original_fails,
        "patched_passes": passed,
        # only a patch the tool produced itself and that passes the tests counts as a repair
        "discovered_fix": bool(passed) and fix["origin"] == DISCOVERED,
        "reference_fallback": fix["origin"] == REFERENCE,
        "original_ms": before_ms,
        "patched_ms": after_ms,
    }


def run_benchmark(programs=None, use_llm=False, use_search=True, timeout=3.0, progress=None, lang="python",
                  use_cache=True, workers=WORKERS):
    """programs run concurrently. cached rows are reused unless use_cache is false; fresh rows are always stored."""
    programs = programs or list_programs(lang)
    backend = llm_provider() if use_llm else None
    cache = load_cache()
    keys = {name: cache_key(name, lang, read_buggy(name, lang), backend, use_search) for name in programs}
    rows, todo = {}, []
    for name in programs:
        hit = cache.get(keys[name]) if use_cache else None
        if hit is not None:
            rows[name] = dict(hit, cached=True)
        else:
            todo.append(name)
    done = len(rows)
    if progress and done:
        progress(done, len(programs), "cache")
    start = time.perf_counter()
    if todo:
        # the token model is fitted once per program for leave-one-out, warm the shared one first
        load_token_model()
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            futures = {pool.submit(evaluate_program, n, use_llm, use_search, timeout, lang): n for n in todo}
            for fut in as_completed(futures):
                name = futures[fut]
                row = fut.result()
                rows[name] = dict(row, cached=False)
                with _lock:
                    cache[keys[name]] = row
                done += 1
                if progress:
                    progress(done, len(programs), name)
        save_cache(cache)
    df = pd.DataFrame([rows[n] for n in programs])
    df.attrs["seconds"] = round(time.perf_counter() - start, 2)
    df.attrs["fresh"] = len(todo)
    return df


def summarize(df):
    found = df[df["discovered_fix"]]
    executed = df["patched_passes"].notna()
    discovered = df[df["patch_origin"] == DISCOVERED]
    return {
        "programs": len(df),
        "top1_hit": int(df["top1_hit"].sum()),
        "top3_hit": int(df["top3_hit"].sum()),
        "original_fails": int(df["original_fails"].fillna(False).astype(bool).sum()),
        "discovered_fix": len(found),
        "discovered_by_ast_search": int((discovered["repair_source"] == "ast_search").sum()),
        "discovered_by_llm": int(discovered["repair_source"].fillna("").str.startswith("llm:").sum()),
        "discovered_passes": int(found["patched_passes"].fillna(False).astype(bool).sum()),
        "reference_fallback": int(df["reference_fallback"].sum()),
        "reference_fallback_passes": int((df["reference_fallback"] & (df["patched_passes"] == True)).sum()),  # noqa: E712
        "unresolved": int((executed & (df["patched_passes"] != True)).sum()),  # noqa: E712
        "not_executed": int((~executed).sum()),
    }


def summarize_origins(df):
    """Show how many programs used each patch source and which patches passed pytest."""
    sources = [
        ("ast_search", "AST/search discovery", True),
        ("llm", "LLM-generated discovery", True),
        ("ground_truth", "Reference fallback (ground truth)", False),
        (None, "No patch", False),
    ]
    rows = []
    for source, label, counts_as_discovered in sources:
        if source == "llm":
            mask = df["repair_source"].fillna("").str.startswith("llm:")
        elif source is None:
            mask = df["repair_source"].isna()
        else:
            mask = df["repair_source"] == source
        group = df.loc[mask]
        rows.append({
            "patch_origin": label,
            "programs": int(len(group)),
            "patches_passing_pytest": int(group["patched_passes"].fillna(False).astype(bool).sum()),
            "counts_as_discovered": counts_as_discovered,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="run the quixbugs benchmark and fill data/benchmark_cache.json")
    ap.add_argument("--lang", default="python", choices=["python", "java"])
    ap.add_argument("--fresh", action="store_true", help="ignore cached rows")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    names = list_programs(args.lang)[:args.limit or None]
    out = run_benchmark(names, lang=args.lang, use_cache=not args.fresh,
                        progress=lambda i, n, name: print(f"[{i}/{n}] {name}", flush=True))
    print(out.drop(columns=["fault_lines"]).to_string(index=False))
    print(summarize(out))
    print(f"{out.attrs['fresh']} fresh rows in {out.attrs['seconds']}s")
