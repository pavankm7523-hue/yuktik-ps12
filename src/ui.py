import html

# dark engineering theme for the custom html blocks. streamlit widgets keep their own theme.
CSS = """
<style>
.ps-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 4px 0 18px 0; }
.ps-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px; }
.ps-label { color: #94a3b8; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
.ps-value { color: #f1f5f9; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 24px;
            font-weight: 600; margin-top: 4px; line-height: 1.2; }
.ps-sub { color: #64748b; font-size: 12px; margin-top: 4px; font-family: ui-monospace, Consolas, monospace; }
.ps-pill { display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 11px; font-weight: 600;
           letter-spacing: 0.06em; font-family: ui-monospace, Consolas, monospace; margin-top: 8px; }
.ps-red { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }
.ps-green { background: #052e16; color: #86efac; border: 1px solid #14532d; }
.ps-amber { background: #451a03; color: #fcd34d; border: 1px solid #78350f; }
.ps-slate { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
.ps-code { background: #0b1220; border: 1px solid #1e293b; border-radius: 8px; overflow-x: auto; padding: 6px 0;
           font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
.ps-row { display: flex; align-items: center; white-space: pre; line-height: 1.55; border-left: 3px solid transparent; }
.ps-row.ps-top { border-left-color: #ef4444; }
.ps-no { color: #475569; width: 44px; text-align: right; padding-right: 10px; flex: none; user-select: none; }
.ps-score { color: #94a3b8; width: 46px; text-align: right; padding-right: 8px; flex: none; font-size: 11px; }
.ps-bar { width: 70px; height: 7px; background: #1e293b; border-radius: 4px; flex: none; margin-right: 12px; }
.ps-bar span { display: block; height: 7px; border-radius: 4px; background: #ef4444; }
.ps-rank { width: 30px; flex: none; color: #fca5a5; font-size: 11px; font-weight: 700; }
.ps-text { color: #e2e8f0; padding-right: 16px; }
.ps-diff-add { background: #052e16; color: #bbf7d0; }
.ps-diff-del { background: #450a0a; color: #fecaca; }
.ps-diff-hunk { color: #7dd3fc; }
.ps-diff-meta { color: #64748b; }
.ps-diff-line { white-space: pre; line-height: 1.55; padding: 0 14px; color: #cbd5e1; }
.ps-term { background: #0b1220; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px; }
.ps-term.ps-fail { border-top: 3px solid #dc2626; }
.ps-term.ps-pass { border-top: 3px solid #16a34a; }
.ps-term.ps-skip { border-top: 3px solid #64748b; }
.ps-term-head { font-family: ui-monospace, Consolas, monospace; font-size: 15px; font-weight: 700; }
.ps-term-head.ps-fail { color: #f87171; }
.ps-term-head.ps-pass { color: #4ade80; }
.ps-term-head.ps-skip { color: #94a3b8; }
.ps-term-meta { color: #64748b; font-size: 12px; font-family: ui-monospace, Consolas, monospace; margin: 4px 0 8px 0; }
.ps-term pre { background: transparent; color: #cbd5e1; font-size: 12px; margin: 0; white-space: pre-wrap;
               max-height: 260px; overflow-y: auto; }
</style>
"""

PILLS = {"red": "ps-red", "green": "ps-green", "amber": "ps-amber", "slate": "ps-slate"}


def _e(text):
    return html.escape(str(text), quote=True)


def kpi_strip(cards):
    """cards: dicts with label, value and optional sub, pill, tone (red, green, amber, slate)."""
    out = ['<div class="ps-kpis">']
    for c in cards:
        out.append('<div class="ps-card">')
        out.append(f'<div class="ps-label">{_e(c["label"])}</div><div class="ps-value">{_e(c["value"])}</div>')
        if c.get("sub"):
            out.append(f'<div class="ps-sub">{_e(c["sub"])}</div>')
        if c.get("pill"):
            out.append(f'<span class="ps-pill {PILLS.get(c.get("tone"), "ps-slate")}">{_e(c["pill"])}</span>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def heatmap(code, ranked, top_k=3):
    """every source line with its number, suspicion bar and a red tint proportional to the score."""
    by_line = {r["line_no"]: r for r in ranked}
    out = ['<div class="ps-code">']
    for no, text in enumerate(code.splitlines(), start=1):
        row = by_line.get(no)
        score = row["suspicion_score"] if row else 0.0
        top = bool(row) and row["rank"] <= top_k
        # tint: up to 55% red for the most suspicious line
        tint = f"background: rgba(239, 68, 68, {0.55 * score:.3f});" if row else ""
        rank = f"#{row['rank']}" if top else ""
        title = _e(row["signals"]) if row and row["signals"] else ""
        out.append(
            f'<div class="ps-row{" ps-top" if top else ""}" style="{tint}" title="{title}">'
            f'<span class="ps-no">{no}</span>'
            f'<span class="ps-score">{f"{score:.2f}" if row else ""}</span>'
            f'<span class="ps-bar"><span style="width: {100 * score:.0f}%"></span></span>'
            f'<span class="ps-rank">{rank}</span>'
            f'<span class="ps-text">{_e(text) or " "}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def diff_view(diff):
    """unified diff with removed lines on dark red and added lines on dark green."""
    out = ['<div class="ps-code">']
    for line in (diff or "(no textual change)").splitlines():
        if line.startswith(("+++", "---")):
            cls = "ps-diff-meta"
        elif line.startswith("@@"):
            cls = "ps-diff-hunk"
        elif line.startswith("+"):
            cls = "ps-diff-add"
        elif line.startswith("-"):
            cls = "ps-diff-del"
        else:
            cls = ""
        out.append(f'<div class="ps-diff-line {cls}">{_e(line) or " "}</div>')
    out.append("</div>")
    return "".join(out)


def run_status(run):
    if run["passed"] is None:
        return "NOT RUN", "skip"
    if run["passed"]:
        return "PASS", "pass"
    return ("TIMEOUT" if run["timed_out"] else "FAIL"), "fail"


def terminal_card(title, run):
    """one verification run as a terminal style card: status, runtime and captured output."""
    status, tone = run_status(run)
    body = ((run.get("stdout") or "") + "\n" + (run.get("stderr") or "")).strip() or "(no output)"
    return (
        f'<div class="ps-term ps-{tone}">'
        f'<div class="ps-term-head ps-{tone}">{_e(title.upper())}: {status}</div>'
        f'<div class="ps-term-meta">{run["runtime_ms"]:.0f} ms | exit code {_e(run["exit_code"])} | {_e(run["command"])}</div>'
        f"<pre>{_e(body[-3000:])}</pre></div>"
    )
