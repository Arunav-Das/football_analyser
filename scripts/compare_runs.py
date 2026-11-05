#!/usr/bin/env python3
"""Compare every run stored under runs/<id>_<label>/ (see scripts/run_one.py).

Reads, per run: run_config.json, <root>/analysis/per_year_eval.csv, <root>/analysis/summary_by_model.csv (ground-truth
attribute accuracies, Ballon d'Or task only) and the per-cell result JSONs (parsed rows, tool calls, turns).
No model calls. Writes runs/comparison/:
  runs_overview.csv          one row per run: configuration + totals + scores
  per_year_by_run.csv        one row per run x edition
  dl_agreement_pairs.csv     domestic-league task (no ground truth): player-set agreement between runs, per edition
  comparison_report.txt      the written comparison
  *.png                      the plots listed in the report
Two tasks are never pooled: 'bdo' (Ballon d'Or / UEFA, scored against df_gt_football.csv) and 'dl' (domestic-league
best player, no ground truth).
"""
import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RUNS = Path("runs")
OUT = RUNS / "comparison"

# reference palette (dataviz skill): surface, ink, categorical order, blue ramp, status
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
STATUS = {"ok": "#0ca30c", "truncated": "#fab219", "empty/other": "#ec835a", "error": "#d03b3b"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def flag(cmd, name, default=None):
    if name in cmd:
        i = cmd.index(name)
        return cmd[i + 1] if i + 1 < len(cmd) and not cmd[i + 1].startswith("--") else True
    return default


def describe(cfg):
    cmd, env = cfg.get("command", []), cfg.get("env_overrides", {})
    script = Path(next((c for c in cmd if c.endswith(".py")), "?")).name
    task = "dl" if "dl_open" in cfg.get("output_root", "") or "dom" in script else "bdo"
    think_off = env.get("CLAUDE_CODE_DISABLE_THINKING") == "1" or flag(cmd, "--max_thinking_tokens") == "0"
    budget = flag(cmd, "--max_thinking_tokens")
    thinking = "off" if think_off else (f"budget {budget}" if budget else "default")
    tools = flag(cmd, "--enable_tools") or "none"
    years = flag(cmd, "--years")
    return {"run": f"{cfg['run_id']}_{cfg['label']}", "run_id": cfg["run_id"], "task": task, "script": script,
            "definition": flag(cmd, "--dl_definition", "-") if task == "dl" else "-",
            "rag": flag(cmd, "--rag_tier", "RAG0"), "directive_level": int(flag(cmd, "--directive_level", 1)),
            "no_tool_directive": "--no_tool_directive" in cmd, "thinking": thinking, "tools": tools,
            "max_turns": flag(cmd, "--max_turns", "-") if tools != "none" else "-",
            "cap": int(flag(cmd, "--max_new_tokens", 8000)), "years_requested": years if years else "all",
            "exit_code": cfg.get("exit_code"), "imported": bool(cfg.get("imported"))}


def short(run_label):
    return run_label if len(run_label) <= 46 else run_label[:45] + "…"


def name_key(s):
    s = re.sub(r"\(.*?\)", "", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def status_bucket(s):
    s = str(s)
    if s == "ok":
        return "ok"
    if s.startswith("truncated"):
        return "truncated"
    if s in ("error",):
        return "error"
    return "empty/other"


def thinking_blocks(rec, run_dir):
    """count thinking content blocks in the final attempt's raw CLI events (thinking text itself is not returned)"""
    files = [f for f in (rec.get("raw_files") or []) if str(f).endswith("_events.jsonl")]
    if not files:
        return np.nan
    p = run_dir / files[-1]
    if not p.exists():
        hits = list(run_dir.rglob(Path(files[-1]).name))
        if not hits:
            return np.nan
        p = hits[0]
    n = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if '"content_block_start"' in line and '"thinking"' in line:
                try:
                    e = json.loads(line)
                    if (e.get("event") or {}).get("content_block", {}).get("type") == "thinking":
                        n += 1
                except ValueError:
                    pass
    return n


def load():
    runs, years, players = [], [], {}
    for cfgp in sorted(RUNS.glob("*/run_config.json")):
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        root = cfgp.parent / Path(cfg.get("output_root", "")).name
        ev = root / "analysis" / "per_year_eval.csv"
        meta = describe(cfg)
        if not ev.exists():
            meta["note"] = "no analysis/per_year_eval.csv"
            runs.append(meta)
            continue
        d = pd.read_csv(ev)
        recs = {}
        for jf in root.glob("*/*.json"):
            if jf.parent.name.startswith("analysis"):
                continue
            try:
                for r in json.loads(jf.read_text(encoding="utf-8")).get("results", []):
                    recs[int(r.get("Year_Requested"))] = r
            except Exception:
                pass
        d["tool_calls"] = d["year"].map(lambda y: sum((recs.get(int(y), {}).get("tool_calls") or {}).values()))
        d["thinking_blocks"] = d["year"].map(lambda y: thinking_blocks(recs.get(int(y), {}), cfgp.parent))
        d["num_turns"] = d["year"].map(lambda y: recs.get(int(y), {}).get("num_turns"))
        d["status_bucket"] = d["status"].map(status_bucket)
        for c, v in meta.items():
            d[c] = v
        years.append(d)
        for y, r in recs.items():
            players[(meta["run"], y)] = {name_key(p.get("Player") or p.get("TargetPlayer")) for p in (r.get("Parsed_Rows") or [])} - {""}
        sm = root / "analysis" / "summary_by_model.csv"
        acc = pd.read_csv(sm).iloc[0].to_dict() if sm.exists() else {}
        gt = d[d["gt_available"] == True] if "gt_available" in d else d.iloc[0:0]
        tp, npred, ngt = gt["tp"].sum(), gt["n_pred"].sum(), gt["n_gt"].sum()
        P = tp / npred if npred else np.nan
        R = tp / ngt if ngt else np.nan
        meta.update({
            "editions_run": len(d), "status_ok": int((d.status_bucket == "ok").sum()),
            "status_truncated": int((d.status_bucket == "truncated").sum()),
            "status_error_or_empty": int(d.status_bucket.isin(["error", "empty/other"]).sum()),
            "rows_total": int(d["n_pred"].sum()), "output_tokens": float(d["completion_tokens"].sum()),
            "cost_usd": float(d["cost_usd"].sum()), "latency_s": float(d["latency_s"].sum()),
            "thinking_chars": float(d["thinking_chars"].sum()), "tool_calls": int(d["tool_calls"].sum()),
            "thinking_blocks": float(d["thinking_blocks"].sum()),
            "citation_complete_share": float(d["citation_complete"].mean()),
            "gt_scored_editions": int(len(gt)) if meta["task"] == "bdo" else 0,
            "micro_precision": P if meta["task"] == "bdo" else np.nan,
            "micro_recall": R if meta["task"] == "bdo" else np.nan,
            "micro_f1": (2 * P * R / (P + R) if P + R else 0.0) if meta["task"] == "bdo" and P == P and R == R else np.nan,
            "macro_f1": float(gt[gt.n_gt > 0]["f1"].mean()) if meta["task"] == "bdo" else np.nan,
            "award_acc": acc.get("gt_award_acc", np.nan) if meta["task"] == "bdo" else np.nan,
            "birthplace_acc": acc.get("gt_birthplace_acc", np.nan) if meta["task"] == "bdo" else np.nan,
            "pop_within_5pct": acc.get("gt_pop_within_5pct", np.nan) if meta["task"] == "bdo" else np.nan,
        })
        runs.append(meta)
    return pd.DataFrame(runs), (pd.concat(years, ignore_index=True) if years else pd.DataFrame()), players


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    return name


def hbar_metrics(ov, metrics, title, name, xlim=(0, 1)):
    """grouped horizontal bars, one group per run, one hue per metric (fixed order)"""
    n, k = len(ov), len(metrics)
    fig, ax = plt.subplots(figsize=(9, 0.55 * n * k / 1.6 + 1.6))
    h = 0.8 / k
    for j, (col, lab) in enumerate(metrics):
        y = np.arange(n) + (j - (k - 1) / 2) * h
        v = ov[col].astype(float).values
        ax.barh(y, v, height=h * 0.9, color=CAT[j], label=lab)
        for yi, vi in zip(y, v):
            if vi == vi:
                ax.text(vi + 0.01, yi, f"{vi:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels([short(r) for r in ov["run"]], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(*xlim); ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return save(fig, name)


def hbar_single(ov, col, title, name, fmt):
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(ov) + 1.2))
    v = ov[col].astype(float).values
    ax.barh(np.arange(len(ov)), v, height=0.6, color=CAT[0])
    for i, vi in enumerate(v):
        ax.text(vi, i, " " + fmt(vi), va="center", fontsize=7, color=INK2)
    ax.set_yticks(np.arange(len(ov))); ax.set_yticklabels([short(r) for r in ov["run"]], fontsize=8); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlim(0, (np.nanmax(v) if len(v) and np.nanmax(v) > 0 else 1) * 1.18)
    return save(fig, name)


def heat(df, value, title, name, fmt="{:.2f}", vmax=None):
    pv = df.pivot_table(index="run", columns="year", values=value, aggfunc="first", dropna=False)
    if pv.empty:
        return None
    pv = pv.reindex(sorted(pv.index))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * pv.shape[1] + 3.5), 0.45 * pv.shape[0] + 1.6))
    m = np.ma.masked_invalid(pv.values.astype(float))
    im = ax.imshow(m, aspect="auto", cmap=BLUE, vmin=0, vmax=vmax if vmax is not None else np.nanmax(pv.values))
    for (i, j), v in np.ndenumerate(pv.values):
        if v == v:
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                    color="#ffffff" if (vmax or np.nanmax(pv.values) or 1) and v > 0.6 * (vmax or np.nanmax(pv.values)) else INK)
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels([short(r) for r in pv.index], fontsize=7)
    ax.grid(False); ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    return save(fig, name)


def status_stack(ov, df, name):
    order = ["ok", "truncated", "empty/other", "error"]
    ct = df.groupby(["run", "status_bucket"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    ct = ct.reindex(sorted(ct.index))
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(ct) + 1.4))
    left = np.zeros(len(ct))
    for s in order:
        v = ct[s].values
        ax.barh(np.arange(len(ct)), v, left=left, height=0.6, color=STATUS[s], label=s, edgecolor=SURFACE, linewidth=2)
        for i, (l, vi) in enumerate(zip(left, v)):
            if vi:
                ax.text(l + vi / 2, i, str(int(vi)), ha="center", va="center", fontsize=7,
                        color=INK if s == "truncated" else "#ffffff")
        left += v
    ax.set_yticks(np.arange(len(ct))); ax.set_yticklabels([short(r) for r in ct.index], fontsize=8); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("editions")
    ax.set_title("Call outcome per run (count of editions)", loc="left", fontsize=10)
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    return save(fig, name)


def dl_agreement(players, dl_runs):
    rows = []
    for a, b in itertools.combinations(sorted(dl_runs), 2):
        for (ra, y), sa in players.items():
            if ra != a or (b, y) not in players:
                continue
            sb = players[(b, y)]
            u = sa | sb
            rows.append({"run_a": a, "run_b": b, "year": y, "n_a": len(sa), "n_b": len(sb), "shared": len(sa & sb),
                         "jaccard": len(sa & sb) / len(u) if u else 1.0})
    return pd.DataFrame(rows)


def agreement_matrix(ag, runs, name):
    if ag.empty:
        return None
    runs = sorted(runs)
    M = pd.DataFrame(np.nan, index=runs, columns=runs)
    for r in runs:
        M.loc[r, r] = 1.0
    for (a, b), g in ag.groupby(["run_a", "run_b"]):
        M.loc[a, b] = M.loc[b, a] = g["jaccard"].mean()
    fig, ax = plt.subplots(figsize=(0.7 * len(runs) + 5, 0.55 * len(runs) + 2.4))
    im = ax.imshow(M.values.astype(float), cmap=BLUE, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(M.values):
        if v == v:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.6 else INK)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([r.split("_")[0] for r in runs], fontsize=8)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels([short(r) for r in runs], fontsize=7)
    ax.grid(False)
    ax.set_title("Domestic-league task: mean Jaccard overlap of player sets\n(shared editions only; agreement, not correctness)",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    return save(fig, name)


def fmt_table(df, cols, floatfmt="{:.3f}"):
    out = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if v != v else floatfmt.format(v))
    return out.to_string(index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ov, py, players = load()
    if ov.empty:
        raise SystemExit("no runs found")
    ov = ov.sort_values("run_id").reset_index(drop=True)
    ov.to_csv(OUT / "runs_overview.csv", index=False)
    keep = [c for c in ["run", "task", "rag", "directive_level", "thinking", "tools", "cap", "year", "status",
                        "n_pred", "n_gt", "tp", "precision", "recall", "f1", "completion_tokens", "thinking_blocks",
                        "tool_calls", "num_turns", "cost_usd", "latency_s", "citation_complete", "missed_players",
                        "false_positive_players"] if c in py.columns]
    py[keep].sort_values(["run", "year"]).to_csv(OUT / "per_year_by_run.csv", index=False)

    plots = []
    bdo = ov[(ov.task == "bdo") & ov.get("micro_f1", pd.Series(dtype=float)).notna()]
    dl = ov[ov.task == "dl"]
    if len(bdo):
        plots.append(hbar_metrics(bdo, [("micro_precision", "precision"), ("micro_recall", "recall"), ("micro_f1", "F1")],
                                  "Ballon d'Or / UEFA task: pooled precision, recall, F1 vs ground truth", "bdo_prf1_by_run.png"))
        plots.append(hbar_metrics(bdo, [("award_acc", "award correct"), ("birthplace_acc", "birthplace correct"),
                                        ("pop_within_5pct", "population within 5%")],
                                  "Ballon d'Or / UEFA task: attribute accuracy on correctly found players", "bdo_attribute_accuracy_by_run.png"))
        plots.append(heat(py[py.task == "bdo"], "f1", "Ballon d'Or / UEFA task: F1 per edition (blank = no ground-truth players)",
                          "bdo_f1_heatmap.png", vmax=1))
    if len(dl):
        plots.append(heat(py[py.task == "dl"], "n_pred", "Domestic-league task: rows returned per edition", "dl_rows_heatmap.png", fmt="{:.0f}"))
    plots.append(status_stack(ov, py, "status_by_run.png"))
    plots.append(hbar_single(ov, "cost_usd", "Reported cost per run (USD, sum over editions)", "cost_by_run.png", lambda v: f"${v:.2f}"))
    plots.append(hbar_single(ov, "output_tokens", "Output tokens per run (sum over editions)", "output_tokens_by_run.png", lambda v: f"{v:,.0f}"))
    plots.append(hbar_single(ov, "latency_s", "Wall time per run (seconds, sum over editions)", "latency_by_run.png", lambda v: f"{v:,.0f}s"))
    if (ov["tool_calls"].fillna(0) > 0).any():
        plots.append(heat(py[py["tool_calls"].fillna(0) >= 0], "tool_calls", "Tool calls per edition", "tool_calls_heatmap.png", fmt="{:.0f}"))
    if (ov["thinking_blocks"].fillna(0) > 0).any():
        plots.append(heat(py, "thinking_blocks", "Thinking blocks per edition (from raw CLI events; thinking text is not returned)",
                          "thinking_blocks_heatmap.png", fmt="{:.0f}"))
    ag = dl_agreement(players, dl["run"].tolist())
    ag.to_csv(OUT / "dl_agreement_pairs.csv", index=False)
    plots.append(agreement_matrix(ag, dl["run"].tolist(), "dl_agreement_matrix.png"))
    plots = [p for p in plots if p]

    L = []
    L.append("RUN COMPARISON  (generated by scripts/compare_runs.py; no model calls)")
    L.append("=" * 110)
    L.append("Two different tasks are compared side by side but never pooled:")
    L.append("  bdo = Ballon d'Or / UEFA Men's Player of the Year question (football_analyzer_inf_mod.py), scored against df_gt_football.csv")
    L.append("  dl  = domestic-league best-player question (claude_football_inf_dom.py), NO ground truth: only health, cost and")
    L.append("        cross-run agreement can be compared, not correctness.")
    L.append("")
    L.append("1. RUN CONFIGURATIONS")
    L.append(fmt_table(ov, ["run_id", "task", "rag", "directive_level", "thinking", "tools", "max_turns", "cap",
                            "years_requested", "no_tool_directive", "exit_code"]))
    L.append("")
    L.append("2. HEALTH AND COST (sums over editions)")
    L.append(fmt_table(ov, ["run_id", "editions_run", "status_ok", "status_truncated", "status_error_or_empty", "rows_total",
                            "output_tokens", "thinking_blocks", "tool_calls", "cost_usd", "latency_s", "citation_complete_share"],
                       "{:.2f}"))
    if len(bdo):
        L.append("")
        L.append("3. BALLON D'OR / UEFA TASK - SCORES VS GROUND TRUTH")
        L.append(fmt_table(bdo, ["run_id", "gt_scored_editions", "micro_precision", "micro_recall", "micro_f1", "macro_f1",
                                 "award_acc", "birthplace_acc", "pop_within_5pct"]))
        base = bdo.iloc[0]
        L.append(f"   Differences vs the first run listed ({base['run']}):")
        for _, r in bdo.iloc[1:].iterrows():
            L.append(f"     {r['run']}: dF1 {r['micro_f1'] - base['micro_f1']:+.3f}  dP {r['micro_precision'] - base['micro_precision']:+.3f}"
                     f"  dR {r['micro_recall'] - base['micro_recall']:+.3f}  cost ${r['cost_usd']:.2f} vs ${base['cost_usd']:.2f}")
    if len(dl):
        L.append("")
        L.append("4. DOMESTIC-LEAGUE TASK - AGREEMENT BETWEEN RUNS (Jaccard of player sets; shared editions only)")
        if ag.empty:
            L.append("   fewer than two domestic-league runs share an edition")
        else:
            s = ag.groupby(["run_a", "run_b"]).agg(editions=("year", "size"), mean_jaccard=("jaccard", "mean"),
                                                   rows_a=("n_a", "sum"), rows_b=("n_b", "sum"), shared=("shared", "sum")).reset_index()
            L.append(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    L.append("")
    L.append("5. NOT RUN")
    L.append("   inf_mod RAG1, RAG2, RAG3, RAG6: need world_cup_squads_reference.csv, which is not in the repository.")
    L.append("   inf_dom refuses RAG3..RAG6 by design (they inject Ballon d'Or / UEFA evidence).")
    L.append("")
    L.append("FILES: runs_overview.csv, per_year_by_run.csv, dl_agreement_pairs.csv, " + ", ".join(plots))
    (OUT / "comparison_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
