#!/usr/bin/env python3
"""
Domestic-league variant of claude_football_inf.py  (no edits to that file are needed)
====================================================================================
Imports claude_football_inf.py (version 2026-11-04.8), replaces its prompt / strategy addenda / award handling by
module-attribute patching (the base functions read these globals at call time), then runs base.main().

Extra flag (consumed here, then removed from argv):   --dl_definition {open,pinned}   (default: open)
  open   : "domestic league" is NOT defined; the model must interpret it and state an INTERPRETATION line.
  pinned : "domestic league" = top-tier national club league of any country in which the player played.
All other flags are the base script's flags (--model, --per_year, --years, --strategy, --max_new_tokens, ...).

Safety rails
  * RAG3..RAG6 are refused: they inject Ballon d'Or / UEFA evidence that belongs to the original task.
  * The ground-truth file default points to df_gt_domestic_league.csv (does not exist) so the ORIGINAL df_gt_football.csv
    is never used to score this variant. Scoring stays OFF until you supply a ground truth for this task.
  * Output roots are separate: ./claude_code_football_player_dl_<open|pinned>_{strategies,by_year}
  * Merged-row recovery (rows without tab / pipe separators) is disabled: it needs known award names.
"""
import re
import sys
import json
import argparse
import logging
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import claude_football_inf as base

DL_VERSION = "2026-11-04.1-dl"
logger = logging.getLogger("dl_variant")

_orig_run_analysis = base.run_analysis
_orig_common_metadata = base._common_metadata

# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
_P_INTRO = "Take the seed input: FIFA Men’s World Cup edition <one by one each edition>.\n\n"

_P_QUESTION = (
    "Which footballers were named to a national team squad for the seed World Cup edition, AND have EVER won the "
    "best player award of a domestic league in which they played club football at any point in their career — "
    "before, during, or after that edition, in any season, and in any country in which they played?\n\n"
)
_P_AWARD_DEF = (
    "A league's best player award means the season-long award presented to the best player of that league, whether "
    "it is given by the league itself, by the players' union or by the press. Monthly or weekly awards, "
    "young-player, goalkeeper, top-scorer and team-of-the-season honours, cup awards, and continental or "
    "international awards do not count.\n\n"
)
_P_SCOPE_OPEN = (
    "The set of leagues is not given to you and it differs for every player. For each squad player you must work "
    "out for yourself which domestic league(s) he played in during his career, which best player award each of "
    "those leagues has (or had), and whether he won it. Decide for yourself what \"domestic league\" means. Do not "
    "restrict yourself to a short list of well-known leagues or awards: leagues in any country count, and so do "
    "awards under any name, including awards of leagues that have been renamed or no longer exist.\n\n"
)
_P_SCOPE_PINNED = (
    "Here \"domestic league\" means the top-tier national club league of a country, in any country in which the "
    "player played club football (not only his own country). Cups, lower divisions, and continental or "
    "international competitions are not domestic leagues. The set of leagues is not given to you and it differs "
    "for every player: for each squad player work out which domestic league(s) he played in during his career, "
    "which best player award each of those leagues has (or had), and whether he won it. Do not restrict yourself "
    "to a short list of well-known leagues or awards; awards under any name count, including awards of leagues "
    "that have been renamed or no longer exist.\n\n"
)
_P_DETAILS = (
    "For each such player, give: each qualifying award, their birthplace, and the population of that birthplace. "
    "For birthplace, give the actual place of birth (a city, town, or settlement) — not a country, a historical "
    "polity, or a sports club/national-team entity.\n\n"
    "Return the complete set of qualifying players for this edition, not just examples. If no player qualifies, "
    "state that explicitly rather than omitting the edition.\n\n"
    "Output format, one row per qualifying player:\n\n"
    "S.No\tWorldCup_Year\tTargetPlayer\tAward_Type\tBirthplace\tPopulation\n\n"
    "In the Award_Type cell write every qualifying award as 'Award name (League, Country, Season)', separated by "
    "'; ' when the player has several.\n\n"
)
_P_INTERPRETATION = (
    "After the table, add one line beginning 'INTERPRETATION:' that states in one or two sentences what you took "
    "'domestic league' to mean and how you decided which leagues were relevant for each player.\n\n"
)
_P_CITATIONS = (
    "RESPONSE MUST CONTAIN — citation of the source used for confirming the seed FIFA World Cup edition was "
    "actually held (and not cancelled or abandoned) in the stated calendar year, the source(s) used for the list of "
    "players named to national team squads for that edition, the source(s) used to establish which domestic "
    "league(s) each qualifying player played in, the source(s) used to establish that he won each listed league's "
    "best player award, the source for each qualifying player's birthplace, and the source for the population of "
    "that birthplace. \n\n"
    "RESPONSE MUST ALSO CONTAIN - citations in the same order in which it is used to answer the query clearly "
    "stating the steps and citations against it."
)


def build_dl_prompt(definition: str) -> str:
    scope = _P_SCOPE_PINNED if definition == "pinned" else _P_SCOPE_OPEN
    interp = _P_INTERPRETATION if definition == "open" else ""
    prompt = _P_INTRO + _P_QUESTION + _P_AWARD_DEF + scope + _P_DETAILS + interp + _P_CITATIONS
    if prompt.count(base.PLACEHOLDER) != 1:
        raise ValueError("the variant prompt must contain the placeholder exactly once")
    return prompt


# ---------------------------------------------------------------------------
# STRATEGY ADDENDA (S1 stays empty = prompt verbatim)
# ---------------------------------------------------------------------------
ADDENDUM_DL: Dict[str, str] = {
    base.STRATEGY_FLAT_CONJUNCTIVE: "",
    base.STRATEGY_SEQUENTIAL_CHAIN: (
        "Work through these steps in order for each edition and state each step together with its citation: Step 1 "
        "confirm the edition was actually held in that year; Step 2 identify the national team squads and the "
        "players named to them; Step 3 for each squad player identify the domestic league(s) he played in during "
        "his career; Step 4 for each of those leagues identify its season-long best player award(s) and keep only "
        "the players who ever won one of them; Step 5 list each qualifying award with its league, country and "
        "season; Step 6 identify the birthplace; Step 7 state the population of that birthplace."),
    base.STRATEGY_VARIABLE_BINDING: (
        "For each edition evaluate these bindings in order: held = was_held(year); squads = national_team_squads(year); "
        "squad_players = players_in(squads); leagues = {p: domestic_leagues_played_in(p) for p in squad_players}; "
        "league_awards = {l: best_player_awards(l) for every league l in leagues}; award_set = {p in squad_players | "
        "p won a season-long best player award of some league in leagues[p]}; awards = awards_of(award_set); "
        "birthplaces = birthplace_of(award_set); populations = population_of(birthplaces). Produce the answer from "
        "the final bindings."),
    base.STRATEGY_ENTITY_FIRST: (
        "Start from each World Cup edition and expand outward: edition -> was it actually held -> national team "
        "squads -> players in those squads -> each player's club career and the domestic leagues he played in -> "
        "each league's season-long best player award -> keep players who won one -> award(s) with league, country "
        "and season -> birthplace -> population of the birthplace."),
    base.STRATEGY_CONSTRAINT_FIRST: (
        "Start from the award constraint: for every domestic league you can identify (in any country, current or "
        "defunct) list the winners of its season-long best player award, determine for which World Cup editions each "
        "winner was named to a national team squad (after confirming each edition was held), then give their "
        "award(s), birthplace and the population of the birthplace grouped by edition."),
    base.STRATEGY_PAIRWISE_EDGES: (
        "Establish each relation independently before composing: player-squad (named to a squad of an edition), "
        "edition-held, player-league (domestic league he played in), league-award (the league's season-long best "
        "player award), player-award (won it), player-birthplace, place-population. Then compose the join per "
        "edition: squad players whose leagues have an award they won, valid only if the edition was held, then "
        "birthplace, then population."),
    base.STRATEGY_CONTRASTIVE: (
        "For each edition include a player only if BOTH hold: (A) named to a national team squad for that edition; "
        "(B) won the season-long best player award of a domestic league in which he played club football, in any "
        "season. Exclude a player who only won monthly or weekly awards, young-player, goalkeeper, top-scorer or "
        "team-of-the-season honours, cup awards, or continental or international awards (for example the Ballon "
        "d'Or or UEFA awards); who won a league award for a league he did not play in; who was not in a squad for "
        "that edition (for example only in another edition's squad or only in qualifying); or if the edition was "
        "not held."),
    base.STRATEGY_SELF_CHECK: (
        "After each reasoning step, verify the result before proceeding (edition held; squad players; leagues "
        "played in; the league's best player award; awards, birthplace and population). Write 'unknown' for any "
        "fact you cannot verify and do not invent facts or sources."),
    base.STRATEGY_MATERIALIZATION: (
        "Materialize intermediate results as checkpoints and complete each before the next: CHECKPOINT 1 which "
        "editions were held and in which year; CHECKPOINT 2 the squads and players per edition; CHECKPOINT 3 for "
        "each squad player the domestic leagues he played in; CHECKPOINT 4 the players who won a season-long best "
        "player award in one of those leagues; CHECKPOINT 5 birthplace and population for each retained player."),
    base.STRATEGY_TEMPORAL_SCOPE: (
        "Keep two temporal scopes separate: (A) edition-specific - squads of that edition only; (B) career-wide - "
        "the leagues he played in and the award season may be any time in his career, before, during or after the "
        "edition. Evaluate both conditions against their own scope, then combine them."),
    base.STRATEGY_SELF_ASK: (
        "Use Self-Ask decomposition INTERNALLY: ask and answer the follow-up sub-questions (which editions were held; "
        "who was in the squads; which domestic leagues each squad player played in; which season-long best player "
        "award each of those leagues has; did he win it; where was he born; what is the population there) and then "
        "write only the single, self-contained final response. Do not ask the user any question and do not wait for "
        "input."),
    base.STRATEGY_FACTORIZATION: (
        "Factorize into independent atomic factors and solve each on its own first: F1 which editions were held; F2 "
        "squad players per edition; F3 all footballers who ever won the season-long best player award of a domestic "
        "league (career-wide, any league worldwide); then combine by explicit conjunction F2 AND F3 per edition "
        "(valid only if F1 holds) and resolve birthplace and population for that final set only."),
    base.STRATEGY_PROB_CHAINING: (
        "Decompose into ordered units (edition held; squad players; leagues played in; league award membership; "
        "awards; birthplace; population). For each unit give a confidence in [0,1] and propagate a joint confidence "
        "along the chain by the probability chain rule; retain a player only if the joint confidence is at least 0.5. "
        "Do not inflate confidence. Do not add columns to the required table; confidences may be given in the step "
        "notes."),
    base.STRATEGY_BACKWARD_GOAL: (
        "Work backward from the required output: to give a population you need the birthplace; to give a birthplace "
        "you need the qualifying player set; to get that set you need, for each squad player, the domestic leagues "
        "he played in and the season-long best player awards of those leagues; the squad list depends on the "
        "edition having been held (base case). Resolve the base case first and then execute forward."),
    base.STRATEGY_DUAL_PATH: (
        "Solve via two independent paths and cross-check them: PATH A squad-first (edition held -> squad players -> "
        "their leagues -> those leagues' best player awards -> winners -> birthplace, population); PATH B "
        "award-first (every domestic league you can identify -> winners of its season-long best player award -> "
        "their World Cup squads -> group by edition -> birthplace, population). The two player sets must be "
        "identical for every edition; if they differ, state the discrepancy and resolve to the set you can best "
        "verify."),
}

# ---------------------------------------------------------------------------
# AWARD HANDLING (open text instead of two known awards)
# ---------------------------------------------------------------------------
_SEASON_RE = re.compile(r"\b(?:1[89]|20)\d{2}(?:\s*[-\u2013/]\s*(?:\d{2}|\d{4}))?\b")
_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")
_INTERP_RE = re.compile(r"^[\s#*>_\-]*INTERPRETATION\s*[:\u2013\u2014-]\s*(.+)$", re.I | re.M)


def _award_items(text: Any) -> List[str]:
    return [i.strip() for i in re.split(r"\s*;\s*", str(text or "")) if i.strip()]


def _league_of(item: str) -> str:
    m = _PAREN_RE.search(item)
    return m.group(1).split(",")[0].strip() if m else ""


def dl_award_set(text: Any):
    """Normalised award items (season stripped). Replaces base.award_set; the Ballon d'Or / UEFA counters become 0."""
    out = set()
    for item in _award_items(text):
        t = re.sub(r"[^a-z0-9 ]+", " ", base._ascii_fold(_SEASON_RE.sub(" ", item)).lower())
        t = " ".join(t.split())
        if t:
            out.add(t)
    return out


def audit_citations_dl(text: str, row_lines, n_players: int, urls: List[str]) -> Dict[str, Any]:
    """Same heuristic as the base audit, with an extra LEAGUES category (career / leagues played in)."""
    order = base._CAT_ORDER
    lines = [l for i, l in enumerate(text.splitlines()) if i not in row_lines and l.strip()]
    first: Dict[str, int] = {}
    for cat in order:
        for idx, l in enumerate(lines):
            if base._CAT_RE[cat].search(l) and base._SRC_RE.search(l):
                first[cat] = idx
                break
    required = order if n_players > 0 else [c for c in order if c in ("EDITION_HELD", "SQUAD", "AWARD")]
    idxs = [first[c] for c in order if c in first]
    order_ok = bool(idxs) and all(idxs[i] <= idxs[i + 1] for i in range(len(idxs) - 1))
    return {"categories_present": [c for c in order if c in first], "n_categories": len(first),
            "complete": all(c in first for c in required), "order_ok": bool(order_ok), "n_urls": len(urls),
            "n_unique_domains": len({re.sub(r"^https?://(?:www\.)?", "", u).split("/")[0].lower() for u in urls})}


def _common_metadata_dl(a: Any, run_mode: str) -> Dict[str, Any]:
    d = _orig_common_metadata(a, run_mode)
    d.update({"variant": "domestic_league_best_player", "variant_version": DL_VERSION,
              "dl_definition": _STATE.get("definition"),
              "variant_note": "awards are open text 'Award name (League, Country, Season)'; RAG3..RAG6 refused; "
                              "no ground truth for this task is bundled"})
    return d


_STATE: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# ANALYSIS ADDENDUM (appended to the base analysis_report.txt)
# ---------------------------------------------------------------------------
def _dl_addendum(root: Path, raw: bool) -> None:
    outdir = root / (base.ANALYSIS_DIR_NAME + ("_raw" if raw else ""))
    report = outdir / "analysis_report.txt"
    if not report.exists():
        return
    row_recs: List[Dict[str, Any]] = []
    item_recs: List[Dict[str, Any]] = []
    call_recs: List[Dict[str, Any]] = []
    interp_recs: List[Dict[str, Any]] = []
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(base.ANALYSIS_DIR_NAME) or jf.parent.name.startswith("diagnosis"):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        md = data.get("metadata", {}) or {}
        for rec in data.get("results", []):
            if bool(rec.get("system_directive_used", False)) == raw:
                continue
            text = str(rec.get("Generated_Response", ""))
            if text.startswith("ERROR"):
                continue
            model = rec.get("model_alias") or md.get("model_alias")
            year = rec.get("Year_Requested") or ""
            m = _INTERP_RE.search(text)
            call_recs.append({"model_alias": model, "has_interpretation": bool(m)})
            if m:
                interp_recs.append({"model_alias": model, "strategy": rec.get("strategy_code"), "year": year,
                                    "interpretation": m.group(1).strip()})
            for r in rec.get("Parsed_Rows") or []:
                if base.is_sentinel_player(r.get("Player", "")):
                    continue
                items = _award_items(r.get("Award_Type", ""))
                row_recs.append({"model_alias": model, "year": r.get("WorldCup_Year"), "player": r.get("Player"),
                                 "n_items": len(items)})
                for it in items:
                    item_recs.append({"model_alias": model, "year": r.get("WorldCup_Year"), "player": r.get("Player"),
                                      "award_item": it, "league": _league_of(it)})
    L: List[str] = ["", "=" * 100, "D. DOMESTIC-LEAGUE VARIANT ADDENDUM", "=" * 100,
                    f"  variant version {DL_VERSION}; definition mode: {_STATE.get('definition')}",
                    "  Awards are read from the Award_Type cell: items split on ';', league = first comma-separated part of the "
                    "final parenthesis. Items not written in that format count as 'league not parsed'.",
                    "  Sections 1-18 above are the base report: its award-type counts (section 11) do not apply here, and "
                    "ground-truth scoring is OFF unless you supplied a ground truth for this task."]
    cdf = pd.DataFrame(call_recs)
    rowdf = pd.DataFrame(row_recs)
    if rowdf.empty:
        L.append("\n  No parsed rows to analyse.")
    else:
        itemdf = pd.DataFrame(item_recs, columns=["model_alias", "year", "player", "award_item", "league"])
        itemdf["lkey"] = itemdf["league"].map(base.place_key)
        itemdf["akey"] = itemdf["award_item"].map(base.name_key)
        per_model = rowdf.groupby("model_alias").agg(
            rows=("player", "size"), mean_awards_per_row=("n_items", "mean"),
            rows_without_award_text=("n_items", lambda s: int((s == 0).sum()))).reset_index()
        extra = []
        for mname, d in itemdf.groupby("model_alias"):
            extra.append({"model_alias": mname, "award_items": len(d),
                          "distinct_leagues": int(d.loc[d["lkey"] != "", "lkey"].nunique()),
                          "distinct_award_types": int(len(d[["akey", "lkey"]].drop_duplicates())),
                          "league_not_parsed_share": float((d["lkey"] == "").mean())})
        if extra:
            per_model = per_model.merge(pd.DataFrame(extra), on="model_alias", how="left")
        if not cdf.empty:
            ic = cdf.groupby("model_alias")["has_interpretation"].mean().rename("interpretation_line_share").reset_index()
            per_model = per_model.merge(ic, on="model_alias", how="left")
        per_model.to_csv(outdir / "dl_summary_by_model.csv", index=False, encoding="utf-8-sig")
        L.append("\nD1. WHAT EACH MODEL SURFACED (rows, awards per row, distinct leagues / award types)")
        L.append(base._tbl(per_model))
        lk = itemdf[itemdf["lkey"] != ""]
        top = pd.DataFrame()
        if not lk.empty:
            cnt = lk.groupby("lkey").size().sort_values(ascending=False)
            disp = lk.groupby("lkey")["league"].agg(lambda s: s.value_counts().index[0])
            nmod = lk.groupby("lkey")["model_alias"].nunique()
            top = pd.DataFrame({"league": disp.reindex(cnt.index).values, "award_items": cnt.values,
                                "models_returning": nmod.reindex(cnt.index).values})
            top.to_csv(outdir / "dl_top_leagues.csv", index=False, encoding="utf-8-sig")
            L.append(f"\nD2. LEAGUES RETURNED ({len(top)} distinct across all records; top 25 by award items)")
            L.append(base._tbl(top.head(25)))
        if interp_recs:
            idf = pd.DataFrame(interp_recs)
            idf.to_csv(outdir / "dl_interpretations.csv", index=False, encoding="utf-8-sig")
            L.append(f"\nD3. INTERPRETATION LINES: {len(idf)} call(s) wrote one -> {outdir / 'dl_interpretations.csv'}")
            for _, r in idf.head(5).iterrows():
                L.append(f"  [{r['model_alias']} {r['year']}] {str(r['interpretation'])[:300]}")
        else:
            L.append("\nD3. No INTERPRETATION lines found (expected in 'pinned' mode, where none is requested).")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if not top.empty:
                t20 = top.head(20)[::-1]
                fig, ax = plt.subplots(figsize=(8, 0.34 * len(t20) + 1.5))
                ax.barh(t20["league"], t20["award_items"], color="#2a6fbb")
                ax.set_xlabel("award items returned"); ax.set_title("Leagues most often returned (domestic-league variant)")
                fig.savefig(outdir / "dl_top_leagues.png", dpi=150, bbox_inches="tight"); plt.close(fig)
            pm = per_model.sort_values("mean_awards_per_row")
            fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(pm) + 1.4)))
            ax.barh(pm["model_alias"], pm["mean_awards_per_row"], color="#c0392b")
            ax.set_xlabel("mean awards per returned row"); ax.set_title("Awards per returned player, by model")
            fig.savefig(outdir / "dl_awards_per_row_by_model.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        except Exception as e:
            logger.warning(f"variant plots skipped: {e}")
    txt = report.read_text(encoding="utf-8")
    txt = re.sub(r"(11\. AWARD TYPES RETURNED\n)  Ballon d'Or rows=[^\n]*\n",
                 r"\1  (not applicable to the domestic-league variant: awards are open text, see section D)\n", txt)
    report.write_text(txt + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def run_analysis_dl(root: Path = None, do_url_check: bool = False, raw: bool = False) -> None:
    root = root or base.OUTPUT_ROOT
    _orig_run_analysis(root, do_url_check=do_url_check, raw=raw)
    try:
        _dl_addendum(root, raw)
    except Exception as e:
        logger.error(f"domestic-league addendum failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dl_definition", choices=["open", "pinned"], default="open")
    pre.add_argument("--rag_tier", "--RAG", dest="rag_tier", nargs="+", default=["RAG0"])
    ns, _ = pre.parse_known_args()
    argv, skip = [sys.argv[0]], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--dl_definition":
            skip = True
            continue
        if a.startswith("--dl_definition="):
            continue
        argv.append(a)
    sys.argv = argv

    try:
        tiers = base.resolve_rag_tiers(ns.rag_tier)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    blocked = tuple(base.GT_TIERS) + (base.RAG_TIER_TITLES_AUTHORS_PRIZE,)
    bad = [base.RAG_CODE[t] for t in tiers if t in blocked]
    if bad:
        print(f"ERROR: {bad} inject Ballon d'Or / UEFA evidence from the original task and are refused in the "
              f"domestic-league variant. Use RAG0 (RAG1/RAG2 need a squad-only reference file).")
        return 2

    _STATE["definition"] = ns.dl_definition
    tag = f"dl_{ns.dl_definition}"
    base.BASE_PROMPT = build_dl_prompt(ns.dl_definition)
    base.STRATEGY_ADDENDUM = ADDENDUM_DL
    base.PRIZE_CRITERIA = ["Season-long best player award of a domestic league in which the player played "
                           f"(definition mode: {ns.dl_definition})"]
    base.OUTPUT_ROOT = Path(f"./claude_code_football_player_{tag}_strategies")
    base.OUTPUT_ROOT_PY = Path(f"./claude_code_football_player_{tag}_by_year")
    base.GROUND_TRUTH_FILE = "df_gt_domestic_league.csv"
    base._AWARD_RUN_RE = re.compile(r"(?!)")          # never matches: merged-row recovery off
    base.award_set = dl_award_set
    base._CAT_RE["AWARD"] = re.compile(r"player of the (?:year|season)|footballer of the year|best player|"
                                       r"most valuable player|\bmvp\b|golden player", re.I)
    base._CAT_RE["LEAGUES"] = re.compile(r"club career|career|played (?:for|in|at)|\bclubs?\b|league (?:history|"
                                         r"appearances)", re.I)
    base._CAT_ORDER = ["EDITION_HELD", "SQUAD", "LEAGUES", "AWARD", "BIRTHPLACE", "POPULATION"]
    base.audit_citations = audit_citations_dl
    base._common_metadata = _common_metadata_dl
    base.run_analysis = run_analysis_dl
    logger.info(f"Domestic-league variant {DL_VERSION}: definition={ns.dl_definition}; "
                f"roots={base.OUTPUT_ROOT} | {base.OUTPUT_ROOT_PY}")
    return base.main()


if __name__ == "__main__":
    sys.exit(main())