import pandas as pd

from src.acr_repair import DISCOVERED, REFERENCE, repair, verify_with_pytest
from src.line_localizer import rank_lines
from src.quixbugs import faulty_lines, list_programs, read_buggy, read_correct


def evaluate_program(name, use_llm=False, use_search=True, timeout=3.0):
    buggy = read_buggy(name)
    truth = set(faulty_lines(buggy, read_correct(name)))
    # leave-one-out: the token model never sees the program it is scored on
    ranked = rank_lines(buggy, exclude_program=name)
    order = [r["line_no"] for r in ranked]
    first_hit = next((i for i, no in enumerate(order, start=1) if no in truth), None)

    fix = repair(buggy, program_name=name, ranked=ranked, use_llm=use_llm, use_search=use_search, timeout=timeout)
    runs = verify_with_pytest(name, buggy, fix["patched"], timeout=timeout)
    passed = runs["after"]["passed"]
    return {
        "program": name,
        "fault_lines": ", ".join(map(str, sorted(truth))),
        "top3_predicted": ", ".join(map(str, order[:3])),
        "rank_of_fault": first_hit,
        "top1_hit": first_hit == 1,
        "top3_hit": first_hit is not None and first_hit <= 3,
        "repair_source": fix["source"],
        "patch_origin": fix["origin"],
        "original_fails": not runs["before"]["passed"],
        "patched_passes": passed,
        # only a patch the tool produced itself and that passes pytest counts as a repair
        "discovered_fix": passed and fix["origin"] == DISCOVERED,
        "reference_fallback": fix["origin"] == REFERENCE,
        "original_ms": runs["before"]["runtime_ms"],
        "patched_ms": runs["after"]["runtime_ms"],
    }


def run_benchmark(programs=None, use_llm=False, use_search=True, timeout=3.0, progress=None):
    programs = programs or list_programs()
    rows = []
    for i, name in enumerate(programs, start=1):
        rows.append(evaluate_program(name, use_llm, use_search, timeout))
        if progress:
            progress(i, len(programs), name)
    return pd.DataFrame(rows)


def summarize(df):
    found = df[df["discovered_fix"]]
    discovered = df[df["patch_origin"] == DISCOVERED]
    return {
        "programs": len(df),
        "top1_hit": int(df["top1_hit"].sum()),
        "top3_hit": int(df["top3_hit"].sum()),
        "original_fails": int(df["original_fails"].sum()),
        "discovered_fix": len(found),
        "discovered_by_ast_search": int((discovered["repair_source"] == "ast_search").sum()),
        "discovered_by_llm": int(discovered["repair_source"].fillna("").str.startswith("llm:").sum()),
        "discovered_passes": int(found["patched_passes"].sum()),
        "reference_fallback": int(df["reference_fallback"].sum()),
        "reference_fallback_passes": int((df["reference_fallback"] & df["patched_passes"]).sum()),
        "unresolved": int((~df["patched_passes"]).sum()),
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
            "patches_passing_pytest": int(group["patched_passes"].sum()),
            "counts_as_discovered": counts_as_discovered,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run_benchmark(progress=lambda i, n, name: print(f"[{i}/{n}] {name}", flush=True))
    print(out.to_string(index=False))
    print(summarize(out))
