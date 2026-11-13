import html
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import benchmark, train_pipeline
from src.acr_repair import DISCOVERED, REFERENCE, build_context, can_execute, llm_provider, repair, verify_with_pytest
from src.ast_metrics import FEATURE_LABELS, FEATURES
from src.line_localizer import load_token_model, rank_lines
from src.quixbugs import faulty_lines, list_programs, program_path, read_buggy, read_correct
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

# three python presets for one-click judge testing of the custom input path
PRESETS = {
    "Load Off-by-One Bug": SAMPLES["python"],
    "Load Mutable Default Bug": ("""def append_item(item, bucket=[]):
    bucket.append(item)
    return bucket
""", """assert append_item(1) == [1]
assert append_item(2) == [2]
assert append_item(3, [1, 2]) == [1, 2, 3]
"""),
    "Load Infinite Loop Timeout Test": ("""def count_set_bits(n):
    count = 0
    while n:
        n ^= n - 1
        count += 1
    return count
""", """assert count_set_bits(5) == 2
assert count_set_bits(8) == 1
assert count_set_bits(0) == 0
"""),
}

LIGHT_LABEL = "Light (White & Orange)"
DARK_LABEL = "Dark (Obsidian & Orange)"
THEMES = {
    LIGHT_LABEL: {
        "bg": "#F8FAFC", "surface": "#FFFFFF", "border": "#E2E8F0",
        "accent": "#EA580C", "accent2": "#F97316", "tint": "#FFF7ED", "tint-border": "#FDBA74",
        "text": "#0F172A", "muted": "#475569", "code-bg": "#0F172A", "code-fg": "#F8FAFC",
        "pass": "#16A34A", "fail": "#DC2626", "warn": "#D97706",
        "accent-rgb": "234, 88, 12", "pass-rgb": "22, 163, 74", "fail-rgb": "220, 38, 38", "warn-rgb": "217, 119, 6",
        "track": "#E2E8F0", "shadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
    },
    DARK_LABEL: {
        "bg": "#0B0F17", "surface": "#111827", "border": "#1F2937",
        "accent": "#F97316", "accent2": "#FB923C", "tint": "rgba(249, 115, 22, 0.12)",
        "tint-border": "rgba(249, 115, 22, 0.35)",
        "text": "#F8FAFC", "muted": "#94A3B8", "code-bg": "#06090F", "code-fg": "#E2E8F0",
        "pass": "#22C55E", "fail": "#EF4444", "warn": "#F59E0B",
        "accent-rgb": "249, 115, 22", "pass-rgb": "34, 197, 94", "fail-rgb": "239, 68, 68", "warn-rgb": "245, 158, 11",
        "track": "#1F2937", "shadow": "0 6px 18px rgba(0, 0, 0, 0.45)",
    },
}
SLATE = "#64748B"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
:root { --mono: 'JetBrains Mono', Consolas, monospace; }

@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes growWidth { from { width: 0; } }
@keyframes enginePulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); opacity: 1; }
[data-testid="stSidebar"] h2 { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }

.stTextArea textarea, .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div,
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
  background: var(--surface); color: var(--text); border-color: var(--border); }
.stTextArea textarea { font-family: var(--mono); font-size: 0.82rem; line-height: 1.55; }
.stNumberInput button { background: var(--surface); color: var(--text); border-color: var(--border); }
[data-baseweb="select"] svg { fill: var(--muted); }
[data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"] { background: var(--surface); }
[role="option"] { background: var(--surface); color: var(--text); }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--tint); }
[data-baseweb="tag"] { background: var(--tint); border: 1px solid var(--tint-border); }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--accent); fill: var(--accent); }

.stButton button, [data-testid="stBaseButton-secondary"] {
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px; font-weight: 600;
  transition: border-color 0.2s cubic-bezier(0.16, 1, 0.3, 1), color 0.2s cubic-bezier(0.16, 1, 0.3, 1),
              background 0.2s cubic-bezier(0.16, 1, 0.3, 1); }
.stButton button p { color: inherit; }
.stButton button:hover, .stButton button:focus-visible { border-color: var(--accent2); color: var(--accent); background: var(--tint); }
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] { background: var(--accent); border-color: var(--accent); color: #FFFFFF; }
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: var(--accent2); border-color: var(--accent2); color: #FFFFFF; }
.stButton button:disabled { opacity: 0.5; }

[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] { color: var(--muted); background: transparent; transition: color 0.2s ease; }
[data-baseweb="tab"] p { color: inherit; font-weight: 600; font-size: 0.9rem; }
[data-baseweb="tab"]:hover { color: var(--accent); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
[data-baseweb="tab-highlight"] { background: var(--accent); }
[data-baseweb="tab-border"] { background: var(--border); }
[data-baseweb="tab-panel"] > div { animation: fadeSlideUp 0.24s cubic-bezier(0.16, 1, 0.3, 1) both; }
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text); }
[data-testid="stProgress"] [role="progressbar"] > div > div { background: var(--accent); }

.ps-card, .kpi, .term, .ps-head, .ps-note, .codebox, .tbl-wrap, .stat {
  animation: fadeSlideUp 0.24s cubic-bezier(0.16, 1, 0.3, 1) both; }
.ps-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
.ps-h { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 18px 0 8px; }
.ps-sub { color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin: 6px 0 12px; }
.mono { font-family: var(--mono); }

.ps-head { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px 16px; margin-bottom: 16px; }
.ps-head-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.ps-brand { display: flex; align-items: center; gap: 12px; }
.ps-mark { width: 34px; height: 34px; border-radius: 9px; background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #FFFFFF; font-family: var(--mono); font-weight: 700; font-size: 0.8rem; display: flex; align-items: center; justify-content: center; }
.ps-title { font-size: 1.22rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text); line-height: 1.2; }
.ps-tag { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
.ps-badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.badge { font-family: var(--mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border); color: var(--muted); background: var(--bg); white-space: nowrap; }
.badge.accent { color: var(--accent); background: var(--tint); border-color: var(--tint-border); }
.engine { display: inline-flex; align-items: center; gap: 7px; }
.engine-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent2); animation: enginePulse 2.5s ease-in-out infinite; }

.pipe { display: flex; align-items: center; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border); }
.pipe-step { display: flex; align-items: center; gap: 9px; flex: 0 0 auto; }
.pipe-num { font-family: var(--mono); font-size: 0.7rem; font-weight: 700; width: 26px; height: 26px; border-radius: 7px;
  display: flex; align-items: center; justify-content: center; border: 1px solid var(--border); color: var(--muted); background: var(--bg);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
.pipe-name { font-size: 0.8rem; font-weight: 600; color: var(--muted); white-space: nowrap; }
.pipe-step.on .pipe-num { background: var(--accent); border-color: var(--accent); color: #FFFFFF; }
.pipe-step.on .pipe-name { color: var(--text); }
.pipe-line { flex: 1 1 24px; height: 2px; min-width: 18px; margin: 0 12px; background: var(--border); border-radius: 2px;
  transition: background 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
.pipe-line.on { background: linear-gradient(90deg, var(--accent), var(--accent2)); }

.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .pipe { flex-wrap: wrap; row-gap: 10px; } }
.kpi { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 17px 16px 14px;
  overflow: hidden; min-height: 132px;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease; }
.kpi::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.kpi:hover { transform: translateY(-1px); border-color: #F97316; box-shadow: var(--shadow); }
.kpi-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); }
.kpi-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-top: 8px; }
.kpi-value { font-family: var(--mono); font-size: 1.7rem; font-weight: 700; color: var(--text); line-height: 1.15; white-space: nowrap; }
.kpi-value.small { font-size: 1.12rem; }
.kpi-foot { font-size: 0.74rem; color: var(--muted); margin-top: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-foot.mono { font-size: 0.72rem; }

.pill { display: inline-block; font-family: var(--mono); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); background: var(--bg); white-space: nowrap; }
.pill.pass { color: var(--pass); background: rgba(var(--pass-rgb), 0.12); border-color: rgba(var(--pass-rgb), 0.4); }
.pill.fail { color: var(--fail); background: rgba(var(--fail-rgb), 0.12); border-color: rgba(var(--fail-rgb), 0.4); }
.pill.warn { color: var(--warn); background: rgba(var(--warn-rgb), 0.13); border-color: rgba(var(--warn-rgb), 0.45); }
.pill.accent { color: var(--accent); background: var(--tint); border-color: var(--tint-border); }

.bar { height: 6px; border-radius: 999px; background: var(--track); overflow: hidden; }
.bar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2));
  animation: growWidth 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }
.bar.fail > i { background: linear-gradient(90deg, var(--accent2), var(--fail)); }
.bar.idle > i { background: var(--muted); opacity: 0.45; }
.kpi .bar { margin-top: 11px; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 12px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.stat-label { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.stat-value { font-family: var(--mono); font-size: 1.25rem; font-weight: 700; color: var(--text); margin-top: 4px; }

.ps-note { border: 1px solid var(--border); border-left: 3px solid var(--muted); background: var(--surface); color: var(--text);
  border-radius: 8px; padding: 11px 14px; font-size: 0.86rem; line-height: 1.5; margin: 8px 0 14px; }
.ps-note b { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; margin-right: 8px; }
.ps-note.info { border-left-color: var(--accent2); background: var(--tint); border-color: var(--tint-border); }
.ps-note.info b { color: var(--accent); }
.ps-note.warn { border-left-color: var(--warn); background: rgba(var(--warn-rgb), 0.1); border-color: rgba(var(--warn-rgb), 0.4); }
.ps-note.warn b { color: var(--warn); }
.ps-note.fail { border-left-color: var(--fail); background: rgba(var(--fail-rgb), 0.09); border-color: rgba(var(--fail-rgb), 0.4); }
.ps-note.fail b { color: var(--fail); }
.ps-note.pass { border-left-color: var(--pass); background: rgba(var(--pass-rgb), 0.09); border-color: rgba(var(--pass-rgb), 0.4); }
.ps-note.pass b { color: var(--pass); }

.tbl-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: auto; margin-bottom: 12px; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 0; }
table.tbl th { position: sticky; top: 0; z-index: 1; background: var(--bg); color: var(--muted); font-size: 0.64rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; text-align: left; padding: 9px 12px; border: 0; border-bottom: 1px solid var(--border); white-space: nowrap; }
table.tbl td { padding: 7px 12px; color: var(--text); border: 0; border-bottom: 1px solid var(--border); vertical-align: top; }
table.tbl tr:last-child td { border-bottom: 0; }
table.tbl tr { transition: background 0.2s ease; }
table.tbl tbody tr:hover { background: var(--tint); }
table.tbl td.num { font-family: var(--mono); text-align: right; white-space: nowrap; }
table.tbl td.code { font-family: var(--mono); white-space: pre; }
table.tbl td.pos { color: var(--accent); }
table.tbl td.neg { color: #64748B; }
table.tbl tr.row-pass td.mark { color: var(--pass); background: rgba(var(--pass-rgb), 0.1); font-weight: 600; }
table.tbl tr.row-warn td.mark { color: var(--warn); background: rgba(var(--warn-rgb), 0.12); font-weight: 600; }

.heat-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: auto; margin-bottom: 14px;
  animation: fadeSlideUp 0.24s cubic-bezier(0.16, 1, 0.3, 1) both; }
.heat-bar-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
table.heat { width: 100%; border-collapse: collapse; margin: 0; font-family: var(--mono); font-size: 0.8rem; }
table.heat td { border: 0; padding: 3px 10px; vertical-align: middle; color: var(--text); }
table.heat tr { border-left: 3px solid transparent; transition: background 0.2s ease; }
table.heat tr:hover { background: var(--tint); }
table.heat td.ln { width: 48px; text-align: right; color: var(--muted); user-select: none; }
table.heat td.sb { width: 170px; }
table.heat td.src { white-space: pre; }
table.heat tr.blank td.src { color: var(--muted); }
.sb-in { display: flex; align-items: center; gap: 8px; }
.sb-in .bar { flex: 1; height: 7px; }
.sb-in span { width: 38px; text-align: right; font-size: 0.7rem; color: var(--muted); }
table.heat tr.top .sb-in span { color: var(--text); font-weight: 700; }
.rank { display: inline-block; margin-left: 10px; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.06em; padding: 1px 6px; border-radius: 4px;
  color: var(--accent); background: var(--tint); border: 1px solid var(--tint-border); vertical-align: middle; }
.rank.hot { color: var(--fail); background: rgba(var(--fail-rgb), 0.12); border-color: rgba(var(--fail-rgb), 0.4); }

.codebox { background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 14px; }
.codebox-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 14px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2); font-family: var(--mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: #94A3B8; }
.codebox-body { overflow: auto; padding: 10px 0; font-family: var(--mono); font-size: 0.79rem; line-height: 1.6; color: var(--code-fg); max-height: 520px; }
.cl { display: flex; white-space: pre; padding: 0 14px; border-left: 3px solid transparent; min-width: max-content; }
.cl .n { flex: 0 0 34px; text-align: right; margin-right: 14px; color: #64748B; user-select: none; }
.cl.add { background: rgba(34, 197, 94, 0.16); border-left-color: #22C55E; }
.cl.del { background: rgba(239, 68, 68, 0.16); border-left-color: #EF4444; }
.cl.hunk { color: #FB923C; }
.cl.meta { color: #94A3B8; }

.term { background: var(--code-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 14px; }
.term.pass { border-top: 3px solid var(--pass); }
.term.fail { border-top: 3px solid var(--fail); }
.term.warn { border-top: 3px solid var(--warn); }
.term-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 14px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }
.term-title { font-family: var(--mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #F8FAFC; }
.term-pills { display: flex; gap: 6px; }
.term .pill { background: rgba(148, 163, 184, 0.12); border-color: rgba(148, 163, 184, 0.3); color: #CBD5E1; }
.term .pill.pass { color: #4ADE80; background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.45); }
.term .pill.fail { color: #F87171; background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.45); }
.term .pill.warn { color: #FBBF24; background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.45); }
.term-cmd { padding: 8px 14px 0; font-family: var(--mono); font-size: 0.72rem; color: #FB923C; white-space: pre-wrap; word-break: break-all; }
.term-out, .term-out * { font-family: var(--mono) !important; }
.term-out { margin: 0; padding: 8px 14px 14px; font-size: 0.75rem; line-height: 1.55; color: var(--code-fg);
  white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto; background: transparent; }

.side-list { font-family: var(--mono); font-size: 0.74rem; color: var(--text); line-height: 1.9; }
.side-list .k { color: var(--muted); display: inline-block; width: 92px; }
.side-list .ok { color: var(--pass); }
.side-list .off { color: var(--warn); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def esc(text):
    # newlines become entities so markdown never sees a blank line inside an html block
    return html.escape(str(text)).replace("\n", "&#10;")


def show(markup, target=None):
    (target or st).markdown(markup, unsafe_allow_html=True)


def inject_theme(theme):
    variables = "".join(f"--{k}: {v};" for k, v in theme.items())
    show(f"<style>:root {{{variables}}}{CSS}</style>")


def sync_native_theme(theme, dark):
    """point streamlit's own widgets (radios, selects, checkboxes) at the active palette. true when it changed."""
    wanted = {
        "theme.base": "dark" if dark else "light",
        "theme.primaryColor": theme["accent"],
        "theme.backgroundColor": theme["bg"],
        "theme.secondaryBackgroundColor": theme["surface"],
        "theme.textColor": theme["text"],
    }
    changed = False
    try:
        for key, value in wanted.items():
            if st.get_option(key) != value:
                st.config.set_option(key, value)
                changed = True
    except Exception:
        # the css variables still theme every custom component if the runtime refuses the option
        return False
    return changed


def heading(text, target=None):
    show(f'<div class="ps-h">{esc(text)}</div>', target)


def note(kind, label, text, target=None):
    show(f'<div class="ps-note {kind}"><b>{esc(label)}</b>{esc(text)}</div>', target)


def pill(text, kind=""):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def bar(fraction, kind=""):
    pct = max(0.0, min(1.0, float(fraction))) * 100.0
    return f'<div class="bar {kind}"><i style="width:{pct:.1f}%"></i></div>'


def stat_row(items, target=None):
    cells = "".join(f'<div class="stat"><div class="stat-label">{esc(k)}</div><div class="stat-value">{esc(v)}</div></div>'
                    for k, v in items)
    show(f'<div class="stat-grid">{cells}</div>', target)


def cell_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def html_table(rows, columns, code_cols=(), max_height=None, row_class=None, mark_cols=(), cell_class=None, target=None):
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            classes = []
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                classes.append("num")
            if col in code_cols:
                classes.append("code")
            if col in mark_cols:
                classes.append("mark")
            if cell_class:
                classes.append(cell_class(col, value) or "")
            cells.append(f'<td class="{" ".join(classes)}">{esc(cell_text(value))}</td>')
        body.append(f'<tr class="{row_class(row) if row_class else ""}">{"".join(cells)}</tr>')
    style = f' style="max-height:{max_height}px"' if max_height else ""
    show(f'<div class="tbl-wrap"{style}><table class="tbl"><thead><tr>{head}</tr></thead>'
         f'<tbody>{"".join(body)}</tbody></table></div>', target)


def code_box(title, text, target=None, right=""):
    lines = "".join(f'<div class="cl"><span class="n">{i}</span><span>{esc(line) or " "}</span></div>'
                    for i, line in enumerate(text.splitlines() or [""], start=1))
    show(f'<div class="codebox"><div class="codebox-head"><span>{esc(title)}</span><span>{right}</span></div>'
         f'<div class="codebox-body">{lines}</div></div>', target)


def diff_box(diff_text):
    out = []
    for line in (diff_text or "(no textual change)").splitlines():
        if line.startswith(("+++", "---")):
            kind = "meta"
        elif line.startswith("@@"):
            kind = "hunk"
        elif line.startswith("+"):
            kind = "add"
        elif line.startswith("-"):
            kind = "del"
        else:
            kind = ""
        out.append(f'<div class="cl {kind}"><span>{esc(line) or " "}</span></div>')
    show('<div class="codebox"><div class="codebox-head"><span>Unified diff</span><span>removed / added</span></div>'
         f'<div class="codebox-body">{"".join(out)}</div></div>')


def run_status(run):
    if run["passed"] is None:
        return "NOT RUN", "warn"
    if run["passed"]:
        return "PASS", "pass"
    return ("TIMEOUT", "warn") if run["timed_out"] else ("FAIL", "fail")


def terminal_card(title, run, target=None):
    status, kind = run_status(run)
    output = (run["stdout"] + run["stderr"]).strip() or "(no output)"
    show(f'<div class="term {kind}"><div class="term-head">'
         f'<span class="term-title">{esc(title)}: EXIT {esc(run["exit_code"])}</span>'
         f'<span class="term-pills">{pill(status, kind)}{pill(f"{run["runtime_ms"]:.0f} ms")}</span></div>'
         f'<div class="term-cmd">$ {esc(run["command"])}</div>'
         f'<pre class="term-out">{esc(output)}</pre></div>', target)


def header_bar(lang_label, provider, stage_done):
    backend = f"LLM: {provider.upper()}" if provider else "AST DETERMINISTIC"
    stages = ["AST Telemetry", "JM1 / SHAP Risk", "Token & AST Localizer", "Pytest Sandbox"]
    parts = []
    for i, name in enumerate(stages):
        on = "on" if stage_done[i] else ""
        if i:
            parts.append(f'<div class="pipe-line {on}"></div>')
        parts.append(f'<div class="pipe-step {on}"><div class="pipe-num">{i + 1:02d}</div>'
                     f'<div class="pipe-name">{esc(name)}</div></div>')
    return ('<div class="ps-head"><div class="ps-head-top"><div class="ps-brand"><div class="ps-mark">PS12</div><div>'
            '<div class="ps-title">Predictive Code Intelligence Workbench</div>'
            '<div class="ps-tag">Defect prediction, bug localization and automated program repair</div></div></div>'
            f'<div class="ps-badges"><span class="badge">LANG: {esc(lang_label.upper())}</span>'
            f'<span class="badge accent">{esc(backend)}</span>'
            '<span class="badge engine"><span class="engine-dot"></span>SANDBOX ENGINE ACTIVE</span></div></div>'
            f'<div class="pipe">{"".join(parts)}</div></div>')


def kpi_strip(probability, ranked, fix, runs, baseline=None, guided=False):
    high = probability >= 0.5
    risk = pill("HIGH RISK", "fail") if high else pill("MODERATE", "warn") if probability >= 0.25 else pill("LOW RISK", "pass")
    card1 = (f'<div class="kpi"><div class="kpi-label">Defect probability</div><div class="kpi-row">'
             f'<div class="kpi-value">{probability:.3f}</div>{risk}</div>{bar(probability, "fail" if high else "")}'
             '<div class="kpi-foot">Classifier output for this file</div></div>')

    if ranked:
        top = ranked[0]
        score = top["suspicion_score"]
        card2 = (f'<div class="kpi"><div class="kpi-label">Top faulty line</div><div class="kpi-row">'
                 f'<div class="kpi-value">L{top["line_no"]}</div>{pill(f"SUSPICION {score:.3f}", "fail" if score >= 0.6 else "accent")}</div>'
                 f'{bar(score, "fail" if score >= 0.6 else "")}'
                 f'<div class="kpi-foot mono">{esc(top["code"].strip())}</div>'
                 f'<div class="kpi-foot">{"Traceback guided" if guided else "Static signals"}</div></div>')
    else:
        card2 = ('<div class="kpi"><div class="kpi-label">Top faulty line</div><div class="kpi-row">'
                 f'<div class="kpi-value">n/a</div>{pill("NO CODE LINES")}</div>{bar(0, "idle")}'
                 '<div class="kpi-foot">Nothing to score</div></div>')

    if not fix:
        origin_value, origin_pill, origin_foot = "Pending", pill("NOT RUN"), "Generate a patch in the repair tab"
    elif not fix["patched"]:
        origin_value, origin_pill, origin_foot = "No patch", pill("UNRESOLVED", "fail"), "Search and LLM produced no patch"
    elif fix["origin"] == REFERENCE:
        origin_value, origin_pill = "Reference Fallback", pill("FALLBACK", "warn")
        origin_foot = "Known fix copied, not counted as a repair"
    else:
        origin_value = "Discovered"
        origin_pill = pill("VALIDATED", "pass") if fix["validated"] else pill("UNVALIDATED", "warn")
        origin_foot = f"Source: {fix['source']}"
    card3 = (f'<div class="kpi"><div class="kpi-label">Patch origin</div><div class="kpi-row">'
             f'<div class="kpi-value small">{esc(origin_value)}</div>{origin_pill}</div>'
             f'<div class="kpi-foot" style="margin-top:22px">{esc(origin_foot)}</div></div>')

    if runs:
        before, bkind = run_status(runs["before"])
        after, akind = run_status(runs["after"])
        total = runs["before"]["runtime_ms"] + runs["after"]["runtime_ms"]
        card4 = (f'<div class="kpi"><div class="kpi-label">Sandbox verification</div><div class="kpi-row">'
                 f'<div class="kpi-value small">{pill(before, bkind)} -&gt; {pill(after, akind)}</div></div>'
                 f'<div class="kpi-row"><div class="kpi-value small">{runs["after"]["runtime_ms"]:.0f} ms</div>'
                 f'{pill("VERIFIED", "pass") if akind == "pass" else pill("NOT VERIFIED", akind)}</div>'
                 f'<div class="kpi-foot">Patched runtime, {total:.0f} ms for both runs</div></div>')
    elif baseline:
        status, kind = run_status(baseline)
        card4 = ('<div class="kpi"><div class="kpi-label">Sandbox verification</div><div class="kpi-row">'
                 f'<div class="kpi-value small">{pill(status, kind)} -&gt; {pill("PENDING")}</div></div>'
                 f'<div class="kpi-row"><div class="kpi-value small">{baseline["runtime_ms"]:.0f} ms</div>'
                 f'{pill("DEFECT CONFIRMED", "fail") if kind != "pass" else pill("NO FAILURE", "pass")}</div>'
                 '<div class="kpi-foot">Original run only, no patch verified yet</div></div>')
    else:
        card4 = ('<div class="kpi"><div class="kpi-label">Sandbox verification</div><div class="kpi-row">'
                 f'<div class="kpi-value small">Pending</div>{pill("NOT RUN")}</div>'
                 '<div class="kpi-foot" style="margin-top:22px">No sandbox run for this input yet</div></div>')
    return f'<div class="kpi-grid">{card1}{card2}{card3}{card4}</div>'


def heatmap_viewer(code, ranked):
    by_line = {r["line_no"]: r for r in ranked}
    rows = []
    for no, text in enumerate(code.splitlines(), start=1):
        row = by_line.get(no)
        if not row:
            rows.append(f'<tr class="blank"><td class="ln">{no}</td><td class="sb"></td><td class="src">{esc(text) or " "}</td></tr>')
            continue
        score = row["suspicion_score"]
        hot = score >= 0.6
        tone = "fail" if hot else "accent"
        style, cls, tag = "", "", ""
        if row["rank"] <= 3:
            cls = "top"
            style = (f' style="border-left-color: var(--{"fail" if hot else "accent2"}); '
                     f'background: rgba(var(--{tone}-rgb), {0.05 + 0.2 * score:.3f})"')
            tag = f'<span class="rank {"hot" if hot else ""}">TOP {row["rank"]}</span>'
        tip = esc(f"{row['ast_node_type']} | token {row['token_score']:.3f} | ast {row['ast_score']:.3f} | {row['signals'] or 'no signals'}")
        rows.append(f'<tr class="{cls}"{style} title="{tip}"><td class="ln">{no}</td>'
                    f'<td class="sb"><div class="sb-in">{bar(score, "fail" if hot else "")}<span>{score * 100:.0f}%</span></div></td>'
                    f'<td class="src">{esc(text) or " "}{tag}</td></tr>')
    show('<div class="heat-wrap"><div class="heat-bar-top"><span>Line / Suspicion 0-100% / Source</span>'
         f'<span>{len(ranked)} scored lines</span></div><table class="heat"><tbody>{"".join(rows)}</tbody></table></div>')


def shap_figure(info, theme):
    shap_s = pd.Series(info["shap"]).sort_values(key=abs)
    fig, ax = plt.subplots(figsize=(7, 7.5), facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.barh(shap_s.index, shap_s.values, color=["#F97316" if v > 0 else SLATE for v in shap_s.values], height=0.72)
    ax.axvline(0, color=theme["muted"], linewidth=0.8)
    ax.set_xlabel(f"SHAP value ({info['shap_units']}), positive pushes towards defective", color=theme["muted"], fontsize=9)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    ax.grid(axis="x", color=theme["border"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(theme["border"])
    for label in ax.get_yticklabels():
        label.set_color(theme["text"])
        label.set_fontfamily("monospace")
    fig.tight_layout()
    return fig


def load_preset(name):
    source, asserts = PRESETS[name]
    st.session_state["lang_select"] = "Python"
    st.session_state["code_python"] = source
    st.session_state["tests_python"] = asserts


st.set_page_config(page_title="PS12 Code Intelligence Workbench", layout="wide")


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


@st.cache_data(max_entries=128, show_spinner=False)
def first_run(code, program, tests, timeout, lang):
    """the failing run of the input as it is. its output guides localization and fills the status card."""
    return run_tests(code, program, tests, timeout=timeout, fail_fast=True, lang=lang)


def retrain():
    proc = subprocess.run([sys.executable, "-m", "src.train_pipeline"], cwd=ROOT, capture_output=True, text=True)
    st.cache_resource.clear()
    st.cache_data.clear()
    return proc


def toggle_theme():
    st.session_state["theme"] = LIGHT_LABEL if st.session_state.get("theme") == DARK_LABEL else DARK_LABEL


theme_label = st.session_state.setdefault("theme", LIGHT_LABEL)
theme = THEMES[theme_label]
st.sidebar.button("Switch to Light Mode" if theme_label == DARK_LABEL else "Switch to Dark Mode",
                  key="theme_toggle", on_click=toggle_theme, width="stretch")
st.sidebar.caption(f"Theme: {theme_label}")
if sync_native_theme(theme, theme_label == DARK_LABEL):
    st.rerun()
inject_theme(theme)

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
lang_label = next(k for k, v in LANGS.items() if v == lang)

model_label = st.sidebar.selectbox("Classifier", list(MODEL_CHOICES))
model_kind = MODEL_CHOICES[model_label]
timeout = st.sidebar.number_input("Test budget per pytest run (s)", min_value=1.0, max_value=30.0, value=3.0, step=0.5)

st.sidebar.header("Training")
if st.sidebar.button("Re-run src/train_pipeline.py"):
    with st.spinner("rebuilding verified Python variants, grouped metrics, and the full-data model"):
        result = retrain()
    code_box("train_pipeline output", (result.stdout + result.stderr).strip()[-1500:], target=st.sidebar)

provider = llm_provider()
st.sidebar.header("Repair backend")
backend_rows = [("llm", provider or "none (no API key in env)", bool(provider)), ("python", "fast worker pool", True)]
backend_rows += [(k, "found" if v else "not on PATH, static check only", bool(v)) for k, v in toolchain().items()]
show('<div class="side-list">' + "".join(
    f'<div><span class="k">{esc(k)}</span><span class="{"ok" if ok else "off"}">{esc(v)}</span></div>'
    for k, v, ok in backend_rows) + "</div>", st.sidebar)

header_slot = st.empty()
show(header_bar(lang_label, provider, [True, False, False, False]), header_slot)

if mode == "Custom Input":
    sample_code, sample_tests = SAMPLES[lang]
    heading("Quick-load judge tests (Python)")
    for column, name in zip(st.columns(len(PRESETS)), PRESETS):
        column.button(name, key=f"preset_{name}", on_click=load_preset, args=(name,), width="stretch")
    # keyed per language so switching the selector loads that language's template
    st.session_state.setdefault(f"code_{lang}", sample_code)
    st.session_state.setdefault(f"tests_{lang}", sample_tests)
    left, right = st.columns(2)
    code = left.text_area(f"{lang_label} source", height=260, key=f"code_{lang}")
    tests = right.text_area("Assert tests (one per line)", height=260, key=f"tests_{lang}")

if not train_pipeline.METRICS_PATH.exists():
    note("warn", "No models", "No trained models found. Use the sidebar button or run: python -m src.train_pipeline")
    st.stop()

if lang == "python":
    try:
        compile(code, "<input>", "exec")
    except SyntaxError as exc:
        note("fail", "Parse error", f"Input does not parse: {exc}")
        st.stop()
elif not _balanced(code):
    note("fail", "Parse error", "Input has unbalanced brackets.")
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


def current_fix():
    saved = st.session_state.get("fix")
    if saved and saved[0] == code:
        return saved[1], saved[2]
    return None, None


def render_top():
    fix_now, runs_now = current_fix()
    show(header_bar(lang_label, provider, [True, True, bool(ranked), bool(runs_now)]), header_slot)
    show(kpi_strip(info["probability"], ranked, fix_now, runs_now, baseline, bool(failure)), kpi_slot)


kpi_slot = st.empty()
render_top()

tab_pred, tab_loc, tab_fix, tab_bench = st.tabs([
    "Defect Prediction & SHAP",
    "Line Localization (DeepLineDP + AST)",
    "Program Repair & Test Sandbox (AutoCodeRover)",
    "Benchmark Evaluation",
])

with tab_pred:
    if lang != "python":
        note("info", "Out of language", "The classifiers were trained on Python AST metrics. For this language the same 29 "
             "columns come from a lexical extractor (Halstead tokens, McCabe branch keywords, brace depth), so the "
             "probability is a model applied outside the language it was trained on.")
    holdout = metrics["models"][model_kind]

    stat_row([
        ("Defect probability", f"{info['probability']:.3f}"),
        ("Holdout precision", f"{holdout['precision']:.3f}"),
        ("Holdout recall", f"{holdout['recall']:.3f}"),
        ("Holdout F1", f"{holdout['f1']:.3f}"),
        ("Holdout ROC-AUC", f"{holdout['roc_auc']:.3f}"),
    ])
    pairs, variants, base = holdout["original_pairs"], holdout["generated_variants"], holdout.get("jm1_baseline")
    st.caption(
        f"{model_label}, trained on {metrics['train_rows']} rows of Python AST metrics ({metrics['training_data']}) and "
        f"scored on {metrics['test_rows']} rows from {len(metrics['test_programs'])} programs it never saw in any variant. "
        f"Accuracy {holdout['accuracy']:.3f}. The deployed model is refit on all {metrics['final_training_rows']} rows "
        f"across {metrics['final_training_programs']} programs. Grouped {metrics['cv_folds']}-fold out-of-fold "
        f"ROC-AUC {holdout['group_cv']['roc_auc']:.3f}, recall {holdout['group_cv']['recall']:.3f}."
    )
    breakdown = [
        {"holdout slice": "all holdout rows", **holdout},
        {"holdout slice": "original QuixBugs buggy/correct pairs", **pairs},
        {"holdout slice": "generated mutants and refactors", **variants},
    ] + ([{"holdout slice": "same rows, model trained on NASA JM1 (C code)", **base}] if base else [])
    html_table(breakdown, ["holdout slice", "rows", "accuracy", "precision", "recall", "f1", "roc_auc"])
    st.caption("An original buggy file and its fix differ by one token and usually have identical metrics, so that "
               "slice is the hard one. Most of the signal comes from the generated variants.")
    if program and lang == "python":
        side = "holdout" if program in metrics["test_programs"] else "training"
        st.caption(f"{program} is in the {side} programs of this split.")

    left, right = st.columns([2, 3])
    with left:
        heading("JM1 and AST metrics" if lang == "python" else "JM1 metrics (lexical extractor)")
        table = [{"metric": f, "meaning": FEATURE_LABELS[f], "value": info["features"][f],
                  "shap": round(float(info["shap"][f]), 4)} for f in FEATURES]
        html_table(table, ["metric", "meaning", "value", "shap"], code_cols=("metric",), max_height=640,
                   cell_class=lambda col, v: ("pos" if v > 0 else "neg" if v < 0 else "") if col == "shap" else "")
    with right:
        heading("SHAP attribution")
        fig = shap_figure(info, theme)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Orange raises defect risk, slate lowers it. TreeExplainer base value {info['base_value']:.3f} "
                   f"({info['shap_units']}).")

        heading("Holdout confusion matrix")
        matrix = holdout["confusion_matrix"]
        html_table([
            {"": "actual clean", "predicted clean": matrix[0][0], "predicted defective": matrix[0][1]},
            {"": "actual defective", "predicted clean": matrix[1][0], "predicted defective": matrix[1][1]},
        ], ["", "predicted clean", "predicted defective"])

with tab_loc:
    if not ranked:
        note("info", "Empty", "No code lines to score.")
    else:
        heading("Inline source heatmap")
        heatmap_viewer(code, ranked)
        if program:
            truth = faulty_lines(code, read_correct(program, lang), lang)
            hit = [r["rank"] for r in ranked if r["line_no"] in truth]
            st.caption(f"Reference fix touches line(s) {truth}. Best rank of a fixed line: {min(hit) if hit else 'n/a'}. "
                       + ("Token weights were refitted without this program before scoring it." if lang == "python"
                          else "Token weights come from the Python programs; the Java rules add the language signals."))

        if failure:
            st.caption("The failing test run boosted lines named in the traceback and lines matching the "
                       "exception type. Hover a line to see its signals.")

        heading("Top 3 suspicious lines")
        html_table(ranked[:3], ["rank", "line_no", "code", "suspicion_score", "ast_node_type", "signals"], code_cols=("code",))

        with st.expander("Per-line score breakdown"):
            html_table(sorted(ranked, key=lambda r: r["line_no"]),
                       ["line_no", "code", "suspicion_score", "token_score", "ast_score", "ast_node_type", "signals"],
                       code_cols=("code",), max_height=520)

        heading("AST context slice")
        ctx = build_context(code, ranked[0]["line_no"], lang=lang)
        a, b = st.columns(2)
        html_table([{"field": k, "value": ctx.get(k)}
                    for k in ("fault_line", "fault_code", "class", "function", "signature", "arguments", "calls")],
                   ["field", "value"], code_cols=("value",), target=a)
        assignments = ctx.get("local_assignments", [])
        if assignments:
            html_table(assignments, list(assignments[0]), code_cols=tuple(assignments[0]), target=b)
        else:
            note("info", "Locals", "No local assignments in this slice.", target=b)
        code_box("Context snippet", ctx.get("snippet", ""))

with tab_fix:
    if not can_test and lang != "python":
        note("info", "Static only", f"No runnable tests for {lang} on this machine "
             + ("(add assert lines)." if not (tests or "").strip() else "(compiler or runtime not on PATH).")
             + " Patches are still proposed; the sandbox falls back to a static bracket check.")
    use_llm = st.checkbox("Use LLM when an API key is set", value=bool(provider), disabled=not provider)
    if st.button("Generate patch and run pytest", type="primary"):
        with st.spinner("searching for a patch"):
            fix = repair(code, program_name=program, assert_block=tests, ranked=ranked, use_llm=use_llm,
                         timeout=timeout, lang=lang, failure_output=failure)
        runs = None
        if fix["patched"]:
            with st.spinner("running the tests on both versions"):
                runs = verify_with_pytest(program, code, fix["patched"], tests, timeout, lang)
        st.session_state["fix"] = (code, fix, runs)
        render_top()

    fix, runs = current_fix()
    if fix:
        code_box("Repair log", "\n".join(fix["log"]))
        if not fix["patched"]:
            if lang != "python" and not can_test:
                note("warn", "No patch", f"No patch produced. Candidate edits need a {lang} compiler or runtime to be "
                     "checked against the asserts, or an API key for the LLM path.")
            else:
                note("warn", "No patch", "No patch produced. Add assert tests, or set an API key for the LLM path.")
            if baseline:
                # no patch to compare against, so show how the original behaves on its own (fail or timeout)
                heading("Test sandbox")
                terminal_card("ORIGINAL EXECUTION", baseline)
        else:
            if fix["origin"] == REFERENCE:
                note("warn", "Reference fallback", "The tool did not find this patch. It is the known fix copied from "
                     f"vendor/QuixBugs/correct_{lang}_programs and does not count as a repair.")
            elif fix["validated"]:
                note("pass", "Discovered", f"Found by {fix['source']} and validated by the tests. No reference file was used.")
            else:
                note("info", "Discovered", f"Found by {fix['source']}, but there were no tests to validate it.")
            st.caption(f"Suspect lines sent to repair: {fix['fault_lines']}.")
            heading("Unified diff")
            diff_box(fix["diff"])
            a, b = st.columns(2)
            code_box("Original", code, target=a, right=esc(HIGHLIGHT[lang]))
            code_box("Patched", fix["patched"], target=b, right=esc(HIGHLIGHT[lang]))

            heading("Test sandbox")
            if not runs:
                note("info", "Not executed", "No tests available for this input, so the patch was not executed.")
            else:
                a, b = st.columns(2)
                terminal_card("ORIGINAL EXECUTION", runs["before"], target=a)
                terminal_card("PATCHED EXECUTION", runs["after"], target=b)
    else:
        note("info", "Repair order", "LLM (if a key is set), validated single-edit AST search, each retried once "
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
    if st.button("Run benchmark", type="primary") and picked:
        progress = st.progress(0.0)
        df = benchmark.run_benchmark(picked, use_llm=bench_llm, use_search=search, timeout=timeout, lang=bench_lang,
                                     use_cache=use_cache, progress=lambda i, n, name: progress.progress(i / n, text=name))
        st.session_state["bench"] = df
    df = st.session_state.get("bench")
    if df is not None:
        s = benchmark.summarize(df)
        n = s["programs"]
        cached = int(df["cached"].sum())
        st.caption(f"{cached} rows from cache, {n - cached} computed fresh in {df.attrs.get('seconds', 0.0):.2f}s.")
        stat_row([
            ("Top-1 line hit", f"{s['top1_hit']}/{n}"),
            ("Top-3 line hit", f"{s['top3_hit']}/{n}"),
            ("Discovered and passing", f"{s['discovered_fix']}/{n}"),
            ("Reference fallback", f"{s['reference_fallback']}/{n}"),
            ("Unresolved", f"{s['unresolved']}/{n}"),
        ])
        st.caption(
            f"Dynamic patches found: {s['discovered_by_ast_search']} by AST search and {s['discovered_by_llm']} by LLM; "
            f"{s['discovered_fix']} passed independent pytest verification. The reference file was used for "
            f"{s['reference_fallback']} programs ({s['reference_fallback_passes']} passed pytest); these fallbacks "
            "are excluded from discovered-repair counts."
            + (f" {s['not_executed']} rows were not executed: no runnable tests for this language on this machine."
               if s["not_executed"] else "")
        )
        heading("Patch origin breakdown")
        origins = benchmark.summarize_origins(df)
        html_table(origins.to_dict("records"), list(origins.columns))

        heading("Per-program results")
        html_table(df.to_dict("records"), list(df.columns), max_height=560, mark_cols=("patch_origin", "repair_source"),
                   row_class=lambda r: {DISCOVERED: "row-pass", REFERENCE: "row-warn"}.get(r.get("patch_origin"), ""))
        st.caption("Green = discovered by the tool. Amber = reference fallback. Python line hits use leave-one-out "
                   "token weights.")
