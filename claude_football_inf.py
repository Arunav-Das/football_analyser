#!/usr/bin/env python3
"""
Football Player Analyser - Claude Code (cloud session) Edition
==============================================================
SCRIPT_VERSION 2026-11-04.8-claude-code

SCOPE
  Claude models ONLY. Every call is a headless Claude Code invocation (`claude -p`) made from inside a Claude Code
  cloud session. No Bedrock endpoint, API key or region exists.
  Only the two repository CSVs are read, and only for RAG1..RAG6 and optional ground-truth scoring:
      world_cup_squads_reference.csv   df_gt_football.csv

TWO RUN MODES
  single-shot (default) : ONE call per (model, strategy, RAG tier). User turn = the prompt exactly as written,
                          including the literal text "<one by one each edition>".
                          Output root: ./claude_code_football_player_strategies
  --per_year            : ONE call per (model, strategy, RAG tier, YEAR); only the placeholder is replaced by the year.
                          Output root: ./claude_code_football_player_by_year

CALL CONFIGURATION (Claude Code headless, streamed)
  claude -p --model <id> --output-format stream-json --verbose --include-partial-messages --tools "" --max-turns 1
         [--system-prompt <directive>]
  prompt via stdin; cwd = empty temp dir; output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; optional MAX_THINKING_TOKENS.
  temperature / top_p / top_k are NOT controllable through the CLI.

EVIDENCE ON DISK (nothing is discarded)
  <output_dir>/raw_calls/<label>_L<level>_try<n>_prompt.txt   exact system + user text sent
                                                _events.jsonl  every raw line the CLI emitted
                                                _text.txt      visible answer as it was generated
                                                _stderr.txt    CLI stderr
  <output_root>/run.log                                        full log of every run (appended)
  A call that hits the output-token cap is NOT retried: it is stored as a truncated record with its partial text.
  The model's text is never altered; stalls, refusals, loops, contradictions are only flagged. --reparse is offline.
"""
import re
import os
import sys
import json
import math
import time
import atexit
import shutil
import logging
import tempfile
import argparse
import threading
import traceback
import subprocess
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests          # used only by --check_urls (URL reachability)

SCRIPT_VERSION = "2026-11-04.8-claude-code"

# ---------------------------------------------------------------------------
# CONSTANTS - Claude models only (alias -> model string passed to `claude --model`)
# ---------------------------------------------------------------------------
FAMILY = "claude"
MODEL_REGISTRY: Dict[str, str] = {
    "claude-opus-4-8":   "claude-opus-4-8",
    "claude-opus-4-7":   "claude-opus-4-7",
    "claude-opus-4-6":   "claude-opus-4-6",
    "claude-opus-4-5":   "claude-opus-4-5-20251101",
    "claude-opus-4-1":   "claude-opus-4-1-20250805",
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
}
MODEL_SIZE_LABEL: Dict[str, str] = {
    "claude-opus-4-8": "opus-4-8", "claude-opus-4-7": "opus-4-7", "claude-opus-4-6": "opus-4-6",
    "claude-opus-4-5": "opus-4-5", "claude-opus-4-1": "opus-4-1", "claude-haiku-4-5": "haiku-4-5",
    "claude-sonnet-4-6": "sonnet-4-6", "claude-sonnet-4-5": "sonnet-4-5",
}
DEFAULT_MODEL_ALIAS = "claude-sonnet-4-6"

REFERENCE_FILE    = "world_cup_squads_reference.csv"
GROUND_TRUTH_FILE = "df_gt_football.csv"

GENERATION_MAX_NEW_TOKENS = 20000     # single-shot: one answer covers every edition with citations
PER_YEAR_MAX_NEW_TOKENS   = 8000      # per-year: one edition per call (used when --max_new_tokens is not given)
MAX_RETRIES    = 3
RETRY_BASE_SEC = 5.0
REQUEST_TIMEOUT = 1800
MAX_CONSECUTIVE_YEAR_ERRORS = 3       # per-year: abort a cell after this many failed calls in a row
PROGRESS_LOG_EVERY_SEC = 30

PLACEHOLDER = "<one by one each edition>"
PER_YEAR_LIST: List[int] = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990,
                            1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]

# ---------------------------------------------------------------------------
# DIRECTIVES (passed with --system-prompt). The user turn is never modified.
# ---------------------------------------------------------------------------
ONE_SHOT_SYSTEM = (
    "You are a non-interactive batch research processor. Your single response is parsed by a program and no human "
    "will reply to it. Complete the ENTIRE task in this one response: process every FIFA Men's World Cup edition, in "
    "chronological order, one after another, until all editions are done. Do not ask questions. Do not offer to "
    "continue. Do not wait for confirmation or say 'let me know when you are ready'. Do not write a conversational "
    "introduction or closing. For an edition with no qualifying player, state that explicitly. Follow the output "
    "format and the citation requirements of the user's message exactly."
)
DIRECTIVE_L2_ADDENDUM = (
    " Never decline, never say the task is too large or cannot be completed in one response, and never explain "
    "limitations: a long answer is expected and your output budget is large enough (tens of thousands of tokens). "
    "Do not skip an edition because it is hard. If you are certain a player qualifies but unsure of the birthplace or "
    "population, write 'unknown' for that field instead of inventing it; if you are unsure whether a player qualifies, "
    "leave that player out. Start your response immediately with the table header row and then the first edition; "
    "write no preface."
)
DIRECTIVE_L3_ADDENDUM = (
    " Work through the editions strictly in chronological order and write each edition's rows and its citations as "
    "soon as you have them, then move on to the next edition until the last edition has been written."
)
DIRECTIVE_LEVELS: Dict[int, str] = {1: ONE_SHOT_SYSTEM,
                                    2: ONE_SHOT_SYSTEM + DIRECTIVE_L2_ADDENDUM,
                                    3: ONE_SHOT_SYSTEM + DIRECTIVE_L2_ADDENDUM + DIRECTIVE_L3_ADDENDUM}
PER_YEAR_SYSTEM = (
    "You are a non-interactive batch research processor. Your single response is parsed by a program and no human "
    "will reply to it. The user message names exactly ONE FIFA Men's World Cup edition. Complete the ENTIRE task for "
    "that one edition in this one response. Do not ask questions. Do not offer to continue or to process other "
    "editions. Do not wait for confirmation or say 'let me know when you are ready'. Do not write a conversational "
    "introduction or closing. If no player qualifies, state that explicitly. Follow the output format and the "
    "citation requirements of the user's message exactly."
)
PY_L2_ADDENDUM = (
    " Never decline, never say the task cannot be completed, and never explain limitations. If you are certain a "
    "player qualifies but unsure of the birthplace or population, write 'unknown' for that field instead of "
    "inventing it; if you are unsure whether a player qualifies, leave that player out. Start your response "
    "immediately with the table header row; write no preface."
)
PY_L3_ADDENDUM = (
    " Write each qualifying player's row as soon as you have identified him, then give the numbered citations in the "
    "order they were used, and stop."
)
DIRECTIVE_LEVELS_PY: Dict[int, str] = {1: PER_YEAR_SYSTEM,
                                       2: PER_YEAR_SYSTEM + PY_L2_ADDENDUM,
                                       3: PER_YEAR_SYSTEM + PY_L2_ADDENDUM + PY_L3_ADDENDUM}
# Opt-in (--no_tool_directive): appended to whichever directive level is active. Without the flag the directive
# text is unchanged, so earlier runs stay reproducible.
NO_TOOL_ADDENDUM = (
    " You have NO tools in this session: no web search, no web fetch, no shell, no code execution and no file access. "
    "Do not write tool calls, <function_calls> or <invoke> markup, shell commands or code of any kind - nothing you "
    "write will be executed and no result will come back. Answer entirely from your own knowledge in this single "
    "response. For each required citation, name the source you are relying on (title and URL) from memory; do not "
    "attempt to open or verify it."
)

_STALL_RE = re.compile(
    r"let me know|ready for me to proceed|shall i (?:proceed|continue)|would you like me to|do you want me to|"
    r"please confirm|should i (?:proceed|continue)|(?:proceed|continue|move on) (?:with|to) the next|"
    r"i(?:'ll| will) (?:now )?(?:proceed|continue)|when you(?:'re| are) ready", re.I)
_DECLINE_RE = re.compile(
    r"cannot be (?:produced|completed|provided|generated|fulfilled)|"
    r"(?:can(?:'|\u2019)?t|cannot|unable to|not (?:possible|feasible|able) to)\s+(?:\w+\s+){0,6}"
    r"(?:complete|produce|provide|generate|list|answer|fulfil)|"
    r"exceed(?:s)? the (?:practical )?limits|beyond the (?:scope|limits)|"
    r"i(?:'|\u2019)?m sorry|i apologi[sz]e", re.I)
_NEG_CLAIM_RE = re.compile(
    r"no\s+(?:player|footballer)s?\s+from\s+the\s+(?:19|20)\d{2}[^.\n]{0,100}(?:won|has\s+(?:ever\s+)?won)|"
    r"none\s+of\s+(?:these|the)\s+(?:players|footballers)\s+(?:won|has|have)", re.I)


def detect_stall(text: str) -> bool:
    return bool(text) and bool(_STALL_RE.search(text[-600:]))


def detect_decline(text: str) -> bool:
    return bool(text) and bool(_DECLINE_RE.search(text[:1500] + " " + text[-800:]))


def detect_contradiction(text: str, n_rows: int) -> bool:
    return n_rows > 0 and bool(_NEG_CLAIM_RE.search(text))


def _norm_line(l: str) -> str:
    return re.sub(r"\s+", " ", l.strip().lower())


def detect_degenerate(text: str) -> Tuple[bool, str]:
    """Repetition loop at the END of the response. FLAG ONLY - the text is never altered."""
    if not text:
        return False, ""
    lines = [l for l in text.splitlines() if l.strip()]
    run, last = 0, None
    for l in reversed(lines):
        n = _norm_line(l)
        if last is None or n == last:
            run += 1
            last = n
        else:
            break
    if run >= 6 and last and len(last) > 8:
        return True, f"tail loop: '{last[:80]}' x{run}"
    for tok, grp in groupby(text.split()):
        if len(tok) > 2 and sum(1 for _ in grp) > 30:
            return True, f"token run: '{tok[:40]}' > 30x"
    return False, ""


QUERY_MODE_NLQ_GENERAL = "nlq_general"
QUERY_MODE_PER_YEAR    = "per_year"
MODE_SUFFIX: Dict[str, str] = {QUERY_MODE_NLQ_GENERAL: "NLQ_GENERAL", QUERY_MODE_PER_YEAR: "NLQ_PERYEAR"}
PRIZE_CRITERIA = ["Ballon d'Or (men's)",
                  "UEFA Men's Player of the Year (incl. UEFA Best Player in Europe, UEFA Club Footballer of the Year)"]
QUERY_ID = 1

# ---------------------------------------------------------------------------
# THE PROMPT - sent exactly as given (per-year: ONLY the placeholder is replaced by the year)
# ---------------------------------------------------------------------------
BASE_PROMPT = (
    "Take the seed input: FIFA Men’s World Cup edition <one by one each edition>.\n"
    "\n"
    "Which footballers were named to a national team squad for the seed World Cup edition, AND have EVER won a men’s "
    "Ballon d’Or or a UEFA Men’s Player of the Year Award (including the award’s previous names: UEFA Best Player in "
    "Europe Award, UEFA Club Footballer of the Year) at any point in their career — before, during, or after that "
    "edition, independent of which specific year the award was won?\n"
    "\n"
    "For each such player, give: their award(s) won, their birthplace, and the population of that birthplace. For "
    "birthplace, give the actual place of birth (a city, town, or settlement) — not a country, a historical polity, or "
    "a sports club/national-team entity.\n"
    "\n"
    "Return the complete set of qualifying players for this edition, not just examples. If no player qualifies, state "
    "that explicitly rather than omitting the edition.\n"
    "\n"
    "Output format, one row per qualifying player:\n"
    "\n"
    "S.No\tWorldCup_Year\tTargetPlayer\tAward_Type\tBirthplace\tPopulation\n"
    "\n"
    "\n"
    "RESPONSE MUST CONTAIN — citation of the source used for confirming the seed FIFA World Cup edition was actually "
    "held (and not cancelled or abandoned) in the stated calendar year, the source(s) used for the list of players "
    "named to national team squads for that edition, the source(s) used for deciding which of those players have won "
    "the listed award(s) (Ballon d'Or and/or UEFA Men's Player of the Year / Best Player in Europe / earlier name), "
    "the source for each qualifying player's birthplace, and the source for the population of that birthplace. \n"
    "\n"
    "RESPONSE MUST ALSO CONTAIN - citations in the same order in which it is used to answer the query clearly "
    "stating the steps and citations against it."
)
assert BASE_PROMPT.count(PLACEHOLDER) == 1, "placeholder must appear exactly once in BASE_PROMPT"

STRATEGY_FLAT_CONJUNCTIVE = "flat_conjunctive"
STRATEGY_SEQUENTIAL_CHAIN = "sequential_chain"
STRATEGY_VARIABLE_BINDING = "variable_binding"
STRATEGY_ENTITY_FIRST     = "entity_first"
STRATEGY_CONSTRAINT_FIRST = "constraint_first"
STRATEGY_PAIRWISE_EDGES   = "pairwise_edges"
STRATEGY_CONTRASTIVE      = "contrastive_constraint"
STRATEGY_SELF_CHECK       = "self_check"
STRATEGY_MATERIALIZATION  = "materialization_checkpoint"
STRATEGY_TEMPORAL_SCOPE   = "temporal_scope_disambiguation"
STRATEGY_SELF_ASK         = "self_ask"
STRATEGY_FACTORIZATION    = "factorization"
STRATEGY_PROB_CHAINING    = "probabilistic_chaining"
STRATEGY_BACKWARD_GOAL    = "backward_goal_decomposition"
STRATEGY_DUAL_PATH        = "dual_path_consistency"
STRATEGY_ALL              = "all"

SUPPORTED_STRATEGIES: List[str] = [
    STRATEGY_FLAT_CONJUNCTIVE, STRATEGY_SEQUENTIAL_CHAIN, STRATEGY_VARIABLE_BINDING,
    STRATEGY_ENTITY_FIRST, STRATEGY_CONSTRAINT_FIRST, STRATEGY_PAIRWISE_EDGES,
    STRATEGY_CONTRASTIVE, STRATEGY_SELF_CHECK, STRATEGY_MATERIALIZATION,
    STRATEGY_TEMPORAL_SCOPE, STRATEGY_SELF_ASK, STRATEGY_FACTORIZATION,
    STRATEGY_PROB_CHAINING, STRATEGY_BACKWARD_GOAL, STRATEGY_DUAL_PATH, STRATEGY_ALL,
]
INDIVIDUAL_STRATEGIES: List[str] = [s for s in SUPPORTED_STRATEGIES if s != STRATEGY_ALL]

STRATEGY_SUFFIX: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "S1_FLAT", STRATEGY_SEQUENTIAL_CHAIN: "S2_SEQ",
    STRATEGY_VARIABLE_BINDING: "S3_VAR", STRATEGY_ENTITY_FIRST: "S4A_ENT",
    STRATEGY_CONSTRAINT_FIRST: "S4B_CON", STRATEGY_PAIRWISE_EDGES: "S5_PAIR",
    STRATEGY_CONTRASTIVE: "S6_CNEG", STRATEGY_SELF_CHECK: "S7_CHK",
    STRATEGY_MATERIALIZATION: "S8_CKPT", STRATEGY_TEMPORAL_SCOPE: "S9_TSCOPE",
    STRATEGY_SELF_ASK: "S11_SASK", STRATEGY_FACTORIZATION: "S12_FACT",
    STRATEGY_PROB_CHAINING: "S13_PCHAIN", STRATEGY_BACKWARD_GOAL: "S14_BGOAL",
    STRATEGY_DUAL_PATH: "S15_DPATH",
}
STRATEGY_CODE: Dict[str, str] = {s: STRATEGY_SUFFIX[s].split("_")[0] for s in INDIVIDUAL_STRATEGIES}
CODE_TO_STRATEGY: Dict[str, str] = {v: k for k, v in STRATEGY_CODE.items()}

STRATEGY_ADDENDUM: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "",
    STRATEGY_SEQUENTIAL_CHAIN: (
        "Work through these steps in order for each edition and state each step together with its citation: Step 1 "
        "confirm the edition was actually held in that year; Step 2 identify the national team squads and the players "
        "named to them; Step 3 keep only the players who ever won either award; Step 4 list the award(s) won; Step 5 "
        "identify the birthplace; Step 6 state the population of that birthplace."),
    STRATEGY_VARIABLE_BINDING: (
        "For each edition evaluate these bindings in order: held = was_held(year); squads = national_team_squads(year); "
        "squad_players = players_in(squads); award_set = {p in squad_players | p ever won a men's Ballon d'Or or a "
        "UEFA Men's Player of the Year Award}; awards = awards_of(award_set); birthplaces = birthplace_of(award_set); "
        "populations = population_of(birthplaces). Produce the answer from the final bindings."),
    STRATEGY_ENTITY_FIRST: (
        "Start from each World Cup edition and expand outward: edition -> was it actually held -> national team squads "
        "-> players in those squads -> keep players who ever won either award -> award(s) -> birthplace -> population "
        "of the birthplace."),
    STRATEGY_CONSTRAINT_FIRST: (
        "Start from the award constraint: take all footballers who ever won a men's Ballon d'Or or a UEFA Men's Player "
        "of the Year Award, determine for which World Cup editions each was named to a national team squad (after "
        "confirming each edition was held), then give their award(s), birthplace and the population of the birthplace "
        "grouped by edition."),
    STRATEGY_PAIRWISE_EDGES: (
        "Establish each relation independently before composing: player-squad (named to a squad of an edition), "
        "edition-held, player-award, player-birthplace, place-population. Then compose the join per edition: squad "
        "players intersected with award winners, valid only if the edition was held, then birthplace, then population."),
    STRATEGY_CONTRASTIVE: (
        "For each edition include a player only if BOTH hold: (A) named to a national team squad for that edition; (B) "
        "ever won a men's Ballon d'Or or a UEFA Men's Player of the Year Award in any year. Exclude a player who lacks "
        "either award, who was not in a squad for that edition (for example only in another edition's squad or only in "
        "qualifying), or if the edition was not held. Other individual awards and women's awards do not count."),
    STRATEGY_SELF_CHECK: (
        "After each reasoning step, verify the result before proceeding (edition held; squad players; award filter; "
        "awards, birthplace and population). Write 'unknown' for any fact you cannot verify and do not invent facts "
        "or sources."),
    STRATEGY_MATERIALIZATION: (
        "Materialize intermediate results as checkpoints and complete each before the next: CHECKPOINT 1 which editions "
        "were held and in which year; CHECKPOINT 2 the squads and players per edition; CHECKPOINT 3 the players who "
        "ever won either award; CHECKPOINT 4 birthplace and population for each retained player."),
    STRATEGY_TEMPORAL_SCOPE: (
        "Keep two temporal scopes separate: (A) edition-specific - squads of that edition only; (B) career-wide - the "
        "award may have been won in any year, before, during or after the edition. Evaluate both conditions against "
        "their own scope, then combine them."),
    STRATEGY_SELF_ASK: (
        "Use Self-Ask decomposition INTERNALLY: ask and answer the follow-up sub-questions (which editions were held; "
        "who was in the squads; who won either award; where was each born; what is the population there) and then "
        "write only the single, self-contained final response. Do not ask the user any question and do not wait for "
        "input."),
    STRATEGY_FACTORIZATION: (
        "Factorize into independent atomic factors and solve each on its own first: F1 which editions were held; F2 "
        "squad players per edition; F3 all footballers who ever won either award (career-wide); then combine by "
        "explicit conjunction F2 AND F3 per edition (valid only if F1 holds) and resolve birthplace and population for "
        "that final set only."),
    STRATEGY_PROB_CHAINING: (
        "Decompose into ordered units (edition held; squad players; award membership; awards; birthplace; "
        "population). For each unit give a confidence in [0,1] and propagate a joint confidence along the chain by the "
        "probability chain rule; retain a player only if the joint confidence is at least 0.5. Do not inflate "
        "confidence. Do not add columns to the required table; confidences may be given in the step notes."),
    STRATEGY_BACKWARD_GOAL: (
        "Work backward from the required output: to give a population you need the birthplace; to give a birthplace "
        "you need the qualifying player set; to get that set you need the award-winner list and the squad list of the "
        "edition; the squad list depends on the edition having been held (base case). Resolve the base case first "
        "and then execute forward."),
    STRATEGY_DUAL_PATH: (
        "Solve via two independent paths and cross-check them: PATH A squad-first (edition held -> squad players -> "
        "filter to award winners -> awards, birthplace, population); PATH B award-first (all award winners -> their "
        "World Cup squads -> group by edition -> awards, birthplace, population). The two player sets must be "
        "identical for every edition; if they differ, state the discrepancy and resolve to the set you can best verify."),
}

RAG_TIER_NO_EVIDENCE            = "no_evidence"
RAG_TIER_YEAR_LIST_ONLY         = "year_list_only"
RAG_TIER_TITLES_AUTHORS         = "titles_authors"
RAG_TIER_TITLES_AUTHORS_PRIZE   = "titles_authors_prize"
RAG_TIER_FULL_WITH_BIRTHPLACE   = "full_with_birthplace"
RAG_TIER_FULL_WITH_POPULATION   = "full_with_population"
RAG_TIER_FULL_WITH_DISTRACTORS  = "full_with_distractors"

RAG_TIERS: List[str] = [
    RAG_TIER_NO_EVIDENCE, RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS,
    RAG_TIER_TITLES_AUTHORS_PRIZE, RAG_TIER_FULL_WITH_BIRTHPLACE,
    RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS,
]
GT_TIERS = (RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS)
REF_TIERS = (RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE)
RAG_TIER_SUFFIX: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "", RAG_TIER_YEAR_LIST_ONLY: "_RAG1_TITLES",
    RAG_TIER_TITLES_AUTHORS: "_RAG2_AUTHORS", RAG_TIER_TITLES_AUTHORS_PRIZE: "_RAG3_PRIZE",
    RAG_TIER_FULL_WITH_BIRTHPLACE: "_RAG4_BIRTHPLACE", RAG_TIER_FULL_WITH_POPULATION: "_RAG5_POPULATION",
    RAG_TIER_FULL_WITH_DISTRACTORS: "_RAG6_DISTRACTORS",
}
RAG_CODE: Dict[str, str] = {t: f"RAG{i}" for i, t in enumerate(RAG_TIERS)}
CODE_TO_RAG: Dict[str, str] = {v: k for k, v in RAG_CODE.items()}
RAG_TIER_DESCRIPTION: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "RAG0 - No evidence: the model resolves everything from parametric memory (prompt unchanged).",
    RAG_TIER_YEAR_LIST_ONLY: "RAG1 - Squad players of every edition in REFERENCE_FILE appended after the prompt.",
    RAG_TIER_TITLES_AUTHORS: "RAG2 - Squad players + national team, every edition (REFERENCE_FILE).",
    RAG_TIER_TITLES_AUTHORS_PRIZE: "RAG3 - Squad players + team + award on record, every edition (REFERENCE_FILE).",
    RAG_TIER_FULL_WITH_BIRTHPLACE: "RAG4 - Qualifying players + awards + birthplace, no population, every edition (GROUND_TRUTH_FILE).",
    RAG_TIER_FULL_WITH_POPULATION: "RAG5 - All ground-truth records incl. population, every edition (GROUND_TRUTH_FILE).",
    RAG_TIER_FULL_WITH_DISTRACTORS: "RAG6 - RAG5 + distractors: squad players with no qualifying award on record (needs REFERENCE_FILE with an award column).",
}
RAG_MAX_ROWS            = 0
RAG_DISTRACTOR_MAX_ROWS = 10

OUTPUT_ROOT    = Path("./claude_code_football_player_strategies")
OUTPUT_ROOT_PY = Path("./claude_code_football_player_by_year")
ANALYSIS_DIR_NAME = "analysis"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def model_id_to_slug(model_id: str) -> str:
    return model_id.replace(":", "_").replace(".", "_")


def clean_year(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "unknown", ""):
        return ""
    m = re.match(r"^(\d{3,4})(?:\.0+)?$", s)
    if m:
        return m.group(1).zfill(4)
    try:
        f = float(s)
        if math.isnan(f):
            return ""
        return f"{int(f):04d}"
    except (ValueError, TypeError):
        return ""


def _norm_col(c: str) -> str:
    return re.sub(r"[\s._]", "", str(c).lower())


def _find_col(columns, aliases: List[str]) -> Optional[str]:
    m = {_norm_col(c): c for c in columns}
    for a in aliases:
        if a in m:
            return m[a]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as fh:
        head = fh.readline()
    sep = "\t" if ("\t" in head and head.count("\t") >= head.count(",")) else ","
    df = pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8-sig", keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean_src(s: Any) -> str:
    return re.sub(r"\s*\|\s*", " / ", re.sub(r"\s+", " ", str(s or "")).strip())


def _split_tokens(tokens: List[str]) -> List[str]:
    return [t.strip() for tok in tokens for t in str(tok).split(",") if t.strip()]


_SENTINELS = {"no result", "unknown", "na", "none", "error", "no qualifying player", "unparsed",
              "no target found", "stalled", "declined", "off year", "empty"}


def is_sentinel_player(s: str) -> bool:
    n = " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())
    return n in _SENTINELS or n.endswith(" error")


def is_error_row(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+_ERROR", str(s).strip()))


def _ascii_fold(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


_NAME_SUFFIX = {"jr", "sr", "ii", "iii"}


def name_key(s: Any) -> str:
    t = re.sub(r"\([^)]*\)", " ", str(s))
    t = re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(t).lower())
    return " ".join(w for w in t.split() if w not in _NAME_SUFFIX)


def names_match(a: Any, b: Any) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return small <= large and (len(large) - len(small)) <= 2


def place_key(s: Any) -> str:
    first = str(s).split(",")[0]
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(first).lower()).split())


def places_match(a: Any, b: Any) -> bool:
    ka, kb = place_key(a), place_key(b)
    return bool(ka and kb and (ka == kb or ka in kb or kb in ka))


def award_set(text: Any) -> Set[str]:
    t = _ascii_fold(text).lower()
    out: Set[str] = set()
    if "ballon" in t:
        out.add("BALLON_DOR")
    if "uefa" in t or "best player in europe" in t or "club footballer" in t:
        out.add("UEFA_POTY")
    return out


# ---- population parsing ------------------------------------------------------
_UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7, "hundred": 100.0}
_UNIT_RE = r"billion|bn|millions?|mn|mm|thousand|lakhs?|lac|crore|cr|hundred|[bmk]"
_NUM_RE = re.compile(
    r"(?<![\w.])("
    r"\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?!\d)"
    r"|\d+(?:[.,]\d+)?"
    r")\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)
_RANGE_RE = re.compile(r"(\d[\d.,]*)\s*(?:-|\u2013|\u2014|to)\s*(\d[\d.,]*)\s*(" + _UNIT_RE + r")(?![a-z])", re.I)
_SCI_RE = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+")
_WORD_NUM_RE = re.compile(r"\b(?:one|a)\s+(billion|million|thousand)\b", re.I)


def _to_float(tok: str, has_unit: bool) -> float:
    t = tok.replace("\u00a0", " ").replace("\u202f", " ").strip()
    if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?", t):
        return float(re.sub(r"[, ]", "", t))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", t):
        return float(t) if (has_unit and t.count(".") == 1) else float(t.replace(".", ""))
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return float(t.replace(",", "."))
    return float(t.replace(",", ""))


def parse_population_value(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if _SCI_RE.fullmatch(s):
        return float(s)
    if not re.search(r"\d", s):
        m = _WORD_NUM_RE.search(s)
        return _UNIT[m.group(1).lower()] if m else None
    rng = _RANGE_RE.search(s)
    if rng:
        try:
            a, b = _to_float(rng.group(1), True), _to_float(rng.group(2), True)
            return (a + b) / 2.0 * _UNIT[rng.group(3).lower()]
        except ValueError:
            pass
    for m in _NUM_RE.finditer(s):
        tok, unit = m.group(1), (m.group(2) or "").lower()
        if not unit and re.fullmatch(r"(?:1[5-9]|20)\d{2}", tok) and \
                re.search(r"(?:as of|in|census|est\.?|\()\s*$", s[:m.start()].lower()):
            continue
        try:
            return _to_float(tok, bool(unit)) * _UNIT.get(unit, 1.0)
        except ValueError:
            continue
    return None


def pop_to_str(raw: Any) -> str:
    v = parse_population_value(raw)
    return str(int(round(v))) if v is not None else ""


def gt_pop_value(raw: Any) -> Optional[int]:
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f):
            return int(round(f))
    except ValueError:
        pass
    v = parse_population_value(s)
    return int(round(v)) if v is not None else None


def gt_pop_to_str(raw: Any) -> str:
    v = gt_pop_value(raw)
    return str(v) if v is not None else ""


ST_OK, ST_TRUNC_LENGTH = "ok", "truncated_length"
ST_TRUNC_REASONING, ST_EMPTY = "truncated_in_reasoning", "empty"


# ---------------------------------------------------------------------------
# OPTIONAL RAG EVIDENCE STORES (the two repository CSVs; read only for RAG1..RAG6 / scoring)
# ---------------------------------------------------------------------------
@dataclass
class ReferenceRecord:
    year: str
    player: str
    team: str
    award: str
    source: str


@dataclass
class GroundTruthRecord:
    year: str
    player: str
    award_type: str
    birthplace: str
    popcount: str
    pop_raw: str
    src_award: str = ""
    src_birthplace: str = ""
    src_population: str = ""
    src_edition: str = ""
    src_squad: str = ""


def _resolve_path(filename: str) -> Path:
    for c in [Path(filename), Path(__file__).resolve().parent / filename, Path.cwd() / filename]:
        if c.exists():
            return c
    raise FileNotFoundError(f"'{filename}' not found in cwd or script directory.")


class ReferenceStore:
    def __init__(self, path: str = REFERENCE_FILE):
        self.path = path
        self.by_year: Dict[str, List[ReferenceRecord]] = {}
        self.has_award_col = False
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = _read_table(candidate)
        c_year = _find_col(df.columns, ["worldcupyear", "year", "editionyear"])
        c_player = _find_col(df.columns, ["player", "targetplayer", "footballer", "name"])
        c_team = _find_col(df.columns, ["team", "nationalteam", "country", "nation"])
        c_award = _find_col(df.columns, ["awardtype", "award", "prizetype", "prize"])
        c_src = _find_col(df.columns, ["source", "squadsource", "sourceurl"])
        if c_year is None or c_player is None:
            raise ValueError(f"'{candidate}' needs WorldCup_Year and Player columns. Found: {list(df.columns)}")
        self.has_award_col = c_award is not None
        count = 0
        for _, r in df.iterrows():
            y, p = clean_year(r[c_year]), str(r[c_player]).strip()
            if not y or not p or is_sentinel_player(p):
                continue
            self.by_year.setdefault(y, []).append(ReferenceRecord(
                y, p, str(r[c_team]).strip() if c_team else "", str(r[c_award]).strip() if c_award else "",
                _clean_src(r[c_src]) if c_src else ""))
            count += 1
        for y in self.by_year:
            seen, uniq = set(), []
            for r in self.by_year[y]:
                k = (name_key(r.player), r.team.lower())
                if k not in seen:
                    seen.add(k); uniq.append(r)
            self.by_year[y] = uniq
        logger.info(f"ReferenceStore loaded {count} rows from '{candidate}' spanning {len(self.by_year)} editions.")


class GroundTruthStore:
    def __init__(self, path: str = GROUND_TRUTH_FILE):
        self.path = path
        self.by_year: Dict[str, List[GroundTruthRecord]] = {}
        self.all_records: List[GroundTruthRecord] = []
        self.n_raw_rows = self.skipped = self.pop_conversion_mismatches = 0
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = _read_table(candidate)
        c_year = _find_col(df.columns, ["worldcupyear", "year", "editionyear"])
        c_player = _find_col(df.columns, ["targetplayer", "player", "footballer"])
        c_award = _find_col(df.columns, ["awardtype", "award", "prizetype", "prize"])
        c_bp = _find_col(df.columns, ["birthplace"])
        c_pop = _find_col(df.columns, ["popcount", "population"])
        c_s = {k: _find_col(df.columns, v) for k, v in {
            "award": ["sourceaward"], "bp": ["sourcebirthplace"], "pop": ["sourcepopulation"],
            "ed": ["sourceedition"], "sq": ["sourcesquad"]}.items()}
        missing = [n for n, c in [("WorldCup_Year", c_year), ("TargetPlayer/Player", c_player),
                                  ("Award_Type", c_award), ("Birthplace", c_bp), ("PopCount/Population", c_pop)] if c is None]
        if missing:
            raise ValueError(f"'{candidate}' missing {missing}. Found: {list(df.columns)}")
        self.n_raw_rows = len(df)
        get = lambda row, c: _clean_src(row[c]) if c else ""
        for _, row in df.iterrows():
            year, player = clean_year(row[c_year]), str(row[c_player]).strip()
            if not year or not player:
                self.skipped += 1
                continue
            raw_pop = str(row[c_pop]).strip()
            canon = gt_pop_to_str(raw_pop)
            try:
                direct = float(raw_pop.replace(",", ""))
                if canon and abs(int(canon) - direct) > 0.5:
                    self.pop_conversion_mismatches += 1
            except ValueError:
                pass
            rec = GroundTruthRecord(year=year, player=player, award_type=str(row[c_award]).strip(),
                                    birthplace=str(row[c_bp]).strip(), popcount=canon, pop_raw=raw_pop,
                                    src_award=get(row, c_s["award"]), src_birthplace=get(row, c_s["bp"]),
                                    src_population=get(row, c_s["pop"]), src_edition=get(row, c_s["ed"]),
                                    src_squad=get(row, c_s["sq"]))
            self.by_year.setdefault(year, []).append(rec)
            self.all_records.append(rec)
        logger.info(f"GroundTruthStore: {len(self.all_records)}/{self.n_raw_rows} rows loaded from '{candidate}' "
                    f"spanning {len(self.by_year)} editions.")
        if self.skipped:
            logger.warning(f"GroundTruthStore: skipped {self.skipped} row(s) with a blank year or player.")
        if self.pop_conversion_mismatches:
            logger.error(f"GroundTruthStore: {self.pop_conversion_mismatches} population value(s) differ from float(raw).")


_EVIDENCE_POP_RE = re.compile(r"(?<!\w)population=(\d+|unknown)")


def _src_suffix(r: GroundTruthRecord, with_pop: bool) -> str:
    bits = []
    if r.src_award:
        bits.append(f"award_src={r.src_award}")
    if r.src_birthplace:
        bits.append(f"birthplace_src={r.src_birthplace}")
    if with_pop and r.src_population:
        bits.append(f"population_src={r.src_population}")
    return (" | " + " | ".join(bits)) if bits else ""


def build_rag_evidence_block(tier: str, ref_store: Optional[ReferenceStore], gt_store: Optional[GroundTruthStore],
                             max_rows: int = 0, only_year: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    stats: Dict[str, Any] = {"editions": 0, "rows_available": 0, "rows_injected": 0, "rows_error": 0,
                             "truncated": False, "pop_check_failures": 0, "evidence_complete": False}
    if tier == RAG_TIER_NO_EVIDENCE:
        return None, stats
    head = ("RETRIEVED EVIDENCE (FIFA Men’s World Cup editions present in the evidence file)" if only_year is None
            else f"RETRIEVED EVIDENCE (FIFA Men’s World Cup edition {only_year})")
    lines = [head, "-" * 60]

    def years_of(d: Dict[str, Any]) -> List[str]:
        return [y for y in sorted(d, key=int) if only_year is None or y == str(only_year)]

    def cap(recs: list):
        stats["rows_available"] += len(recs)
        if max_rows and len(recs) > max_rows:
            stats["truncated"] = True
            recs = recs[:max_rows]
        stats["rows_injected"] += len(recs)
        return recs

    if tier in REF_TIERS:
        if ref_store is None or not ref_store.by_year:
            return None, stats
        for y in years_of(ref_store.by_year):
            recs = cap(list(ref_store.by_year[y]))
            stats["editions"] += 1
            srcs = sorted({r.source for r in recs if r.source})[:3]
            lines.append(f"Edition {y} - players named to national team squads ({len(recs)} records)"
                         + (f"; source(s): {'; '.join(srcs)}" if srcs else "") + ":")
            for r in recs:
                if tier == RAG_TIER_YEAR_LIST_ONLY:
                    lines.append(f"  - {r.player}")
                elif tier == RAG_TIER_TITLES_AUTHORS:
                    lines.append(f"  - {r.player} | team={r.team or 'unknown'}")
                else:
                    lines.append(f"  - {r.player} | team={r.team or 'unknown'} | award_on_record={r.award or 'none recorded'}")
    elif tier in GT_TIERS:
        if gt_store is None or not gt_store.by_year:
            return None, stats
        with_pop = tier != RAG_TIER_FULL_WITH_BIRTHPLACE
        for y in years_of(gt_store.by_year):
            recs = cap(list(gt_store.by_year[y]))
            stats["editions"] += 1
            lab = [("Edition-held source(s)", {r.src_edition for r in recs if r.src_edition}),
                   ("Squad-list source(s)", {r.src_squad for r in recs if r.src_squad})]
            extra = "; ".join(f"{k}: {'; '.join(sorted(v)[:3])}" for k, v in lab if v)
            lines.append(f"Edition {y} - qualifying players ({len(recs)} records)" + (f"; {extra}" if extra else "") + ":")
            for r in recs:
                if is_error_row(r.player):
                    stats["rows_error"] += 1
                    lines.append(f"  - [{r.award_type}] Evidence unavailable for this award (retrieval error).")
                elif is_sentinel_player(r.player):
                    lines.append("  - No qualifying player found in ground-truth evidence for this edition.")
                elif not with_pop:
                    lines.append(f"  - [{r.award_type}] {r.player} | birthplace={r.birthplace or 'unknown'}" + _src_suffix(r, False))
                else:
                    line = f"  - [{r.award_type}] {r.player} | birthplace={r.birthplace or 'unknown'} | population={r.popcount or 'unknown'}"
                    expected = gt_pop_value(r.pop_raw)
                    m = _EVIDENCE_POP_RE.search(line)
                    shown = None if (m is None or m.group(1) == "unknown") else int(m.group(1))
                    if shown != expected:
                        stats["pop_check_failures"] += 1
                        logger.error(f"Edition {y}: population mismatch for {r.player!r}: raw={r.pop_raw!r} expected={expected} shown={shown}")
                    lines.append(line + _src_suffix(r, True))
        if not with_pop:
            lines.append("NOTE: population is not provided at this tier; resolve it yourself.")
        if tier == RAG_TIER_FULL_WITH_DISTRACTORS:
            if ref_store is not None and ref_store.has_award_col:
                lines.append("")
                lines.append("DISTRACTORS - players named to a squad of the edition shown who have NO qualifying award on record "
                             "(they do not qualify; do not list them):")
                for y in years_of(ref_store.by_year):
                    d = [r for r in ref_store.by_year[y] if not r.award][:RAG_DISTRACTOR_MAX_ROWS]
                    for r in d:
                        lines.append(f"  - [Edition {y}] {r.player} | team={r.team or 'unknown'} | no qualifying award on record")
            else:
                logger.warning("RAG6 needs --reference_file with an award column to build distractors; none added.")
    else:
        return None, stats
    if only_year is not None and stats["editions"] == 0:
        return None, stats
    if stats["truncated"]:
        lines.append(f"  ... ({stats['rows_available'] - stats['rows_injected']} further evidence rows omitted by "
                     f"--rag_max_rows; the lists above are NOT exhaustive)")
    lines.append("-" * 60)
    lines.append("Use the evidence above where it is relevant. Where it is silent, rely on your own verified knowledge "
                 "rather than inventing a fact.")
    stats["evidence_complete"] = (not stats["truncated"] and stats["rows_injected"] == stats["rows_available"]
                                  and stats["pop_check_failures"] == 0)
    return "\n".join(lines), stats


def report_rag_coverage(tier: str, ref_store: Optional[ReferenceStore], gt_store: Optional[GroundTruthStore],
                        max_rows: int) -> None:
    if tier == RAG_TIER_NO_EVIDENCE:
        return
    src = gt_store.by_year if (tier in GT_TIERS and gt_store is not None) else (ref_store.by_year if ref_store else {})
    counts = [len(v) for v in src.values()] or [0]
    print(f"\n  RAG COVERAGE ({RAG_CODE[tier]} / {tier}): {len(src)} editions in the evidence file; records total={sum(counts)}, "
          f"per edition min/median/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}")
    print(f"    {'--rag_max_rows=' + str(max_rows) + ' (per edition)' if max_rows else 'no row cap: every record is injected'}"
          f"; approx. evidence size ~{sum(counts) * 25} tokens")


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------
def build_prompt(strategy: str, evidence_block: Optional[str] = None, year: Optional[int] = None) -> str:
    base = BASE_PROMPT if year is None else BASE_PROMPT.replace(PLACEHOLDER, str(year))
    parts = [base]
    if evidence_block:
        parts.append(evidence_block)
    add = STRATEGY_ADDENDUM.get(strategy, "")
    if add:
        parts.append(f"Reasoning approach ({STRATEGY_CODE[strategy]} - {strategy}): {add}\n"
                     f"The output format and the citation requirements above remain unchanged.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# RESPONSE PARSER (multi-edition, format-tolerant) + EDITION SECTIONS + HEURISTIC CITATION AUDIT
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s)\]>\"'|]+|www\.[^\s)\]>\"'|]+", re.I)
_NONE_RE = re.compile(r"no\s+(?:footballers?|players?)\s+(?:qualif|met\b|meet|satisf|found)|no\s+qualifying\s+"
                      r"(?:footballers?|players?)|none\s+of\s+the\s+(?:players|footballers)|no\s+matching", re.I)
_SRC_RE = re.compile(r"https?://|www\.|wikipedia|fifa|rsssf|uefa|france\s+football|transfermarkt|worldfootball|"
                     r"britannica|geonames|census|statistic|worldometer|encyclop|archive|\bsources?\b|\bcitation|\bref\b", re.I)
_CAT_RE = {
    "EDITION_HELD": re.compile(r"\bheld\b|took place|was hosted|cancel|abandon", re.I),
    "SQUAD": re.compile(r"squad|roster|called up|named to", re.I),
    "AWARD": re.compile(r"ballon|best player in europe|club footballer|player of the year", re.I),
    "BIRTHPLACE": re.compile(r"birthplace|\bborn\b|place of birth", re.I),
    "POPULATION": re.compile(r"population|inhabitants|census", re.I),
}
_CAT_ORDER = ["EDITION_HELD", "SQUAD", "AWARD", "BIRTHPLACE", "POPULATION"]

_AWARD_ONE = (r"(?:Ballon\s+d['’]Or|UEFA\s+(?:Men['’]?s\s+Player\s+of\s+the\s+Year|Best\s+Player\s+in\s+Europe"
              r"(?:\s+Award)?|Club\s+Footballer\s+of\s+the\s+Year))")
_AWARD_RUN_RE = re.compile(_AWARD_ONE + r"(?:\s*(?:,|;|&|/|\band\b)\s*" + _AWARD_ONE + r")*", re.I)
_TAIL_POP_RE = re.compile(r"^(?P<bp>.*?)[\s,]*(?P<pop>\d[\d,.\u00a0\u202f ]*(?:\s*(?:million|thousand|[mk]))?)"
                          r"\s*(?:\[\d+\])*\s*$", re.I)
_FREE_ROW_RE = re.compile(r"^\s*(?:\d{1,3}[.)]\s+)?((?:1[89]|20)\d{2})\s+(.+?)\s*$")
_CITE_MARK_RE = re.compile(r"\s*\[\d+\]")
_SELF_EXCL_RE = re.compile(r"does\s+not\s+qualify|not\s+a\s+winner", re.I)
_NAME_ANNOT_RE = re.compile(r"\s+[\u2013\u2014-]\s+.*$")


def clean_player_name(s: Any) -> str:
    s = _CITE_MARK_RE.sub("", str(s))
    return _NAME_ANNOT_RE.sub("", s).strip()


def _parse_merged(text: str, player_hint: str = "") -> Optional[Tuple[str, str, str, str]]:
    m = _AWARD_RUN_RE.search(text)
    if not m:
        return None
    player = player_hint or text[:m.start()].strip(" \t-\u2013:|")
    if not player:
        return None
    award = m.group(0).strip()
    tail = text[m.end():].strip(" \t,;|-\u2013")
    bp, pop_raw = tail, ""
    t = _TAIL_POP_RE.match(tail)
    if t and t.group("bp").strip():
        bp, pop_raw = t.group("bp").strip(" ,;"), t.group("pop").strip()
    return player, award, bp, pop_raw


def _split_cells(line: str) -> Optional[List[str]]:
    s = line.strip()
    if "\t" in s:
        return [c.strip() for c in s.split("\t")]
    if s.count("|") >= 3:
        return [c.strip() for c in s.strip("|").split("|")]
    return None


def parse_table(text: str) -> Tuple[List[Dict[str, str]], Set[int]]:
    rows: List[Dict[str, str]] = []
    row_lines: Set[int] = set()
    for i, line in enumerate(text.splitlines()):
        clean = re.sub(r"[*`]+", "", line)
        cells = _split_cells(clean)
        if _SELF_EXCL_RE.search(clean) and (cells or _FREE_ROW_RE.match(clean)):
            row_lines.add(i)
            continue
        year = player = award = bp = pop_raw = None
        if cells:
            k = 1 if re.fullmatch(r"\d{1,3}[.)]?", cells[0]) else 0
            if k < len(cells) and re.fullmatch(r"(?:1[89]|20)\d{2}", cells[k]):
                year, rest = cells[k], cells[k + 1:]
                if len(rest) >= 3 and rest[0]:
                    player, award, bp = rest[0], rest[1], rest[2]
                    pop_raw = rest[3] if len(rest) > 3 else ""
                elif rest and rest[0]:
                    mg = _parse_merged(" ".join(rest[1:]), rest[0]) if len(rest) >= 2 else _parse_merged(rest[0])
                    if mg:
                        player, award, bp, pop_raw = mg
        else:
            m = _FREE_ROW_RE.match(clean)
            if m:
                mg = _parse_merged(m.group(2))
                if mg:
                    year = m.group(1)
                    player, award, bp, pop_raw = mg
        if year is None or not player:
            continue
        player = clean_player_name(player)
        award, bp = (_CITE_MARK_RE.sub("", str(x)).strip() for x in (award, bp))
        if not player:
            continue
        pop_raw = pop_raw or ""
        pop = "unknown" if (pop_raw and not re.search(r"\d", pop_raw)) else pop_to_str(pop_raw)
        rows.append({"S.No": "", "WorldCup_Year": year, "Player": player, "Award_Type": award,
                     "Birthplace": bp, "PopCount": pop})
        row_lines.add(i)
    seen, uniq = set(), []
    for r in rows:
        key = (r["WorldCup_Year"], name_key(r["Player"]))
        if key not in seen:
            seen.add(key); uniq.append(r)
    for n, r in enumerate(uniq, 1):
        r["S.No"] = str(n)
    return uniq, row_lines


_SEC_HDR_RE = re.compile(
    r"^[\s#*>_\-]*(?:(?:1[89]|20)\d{2})\s+(?:FIFA\s+)?(?:Men[’']?s\s+)?World\s*Cup"
    r"|^[\s#*>_\-]*(?:FIFA\s+)?(?:Men[’']?s\s+)?World\s*Cup\s*(?:edition\s*)?[:\-]?\s*(?:1[89]|20)\d{2}"
    r"|^[\s#*>_\-]*Edition\s*[:\-]?\s*(?:1[89]|20)\d{2}"
    r"|^[\s#*>_]*(?:\d{1,2}[.)]\s*)?(?:1[89]|20)\d{2}\s*[\u2013\u2014\-:]\s*[A-Z][^|\t]{2,40}$", re.I)
_SEC_NONE_RE = re.compile(r"no\s+qualifying|no\s+(?:footballers?|players?)\s+(?:qualif|met\b|meet|satisf|found)|"
                          r"none\s+qualif|not\s+held|was\s+not\s+held|cancell?ed", re.I)


def parse_edition_sections(text: str, row_lines: Set[int], row_years: Set[str]) -> Tuple[List[str], List[str]]:
    declared: List[str] = []
    body: Dict[str, List[str]] = {}
    cur = None
    for i, l in enumerate(text.splitlines()):
        if i in row_lines:
            continue
        if _SEC_HDR_RE.search(l):
            m = re.search(r"(?<!\d)((?:1[89]|20)\d{2})(?!\d)", l)
            if m:
                cur = m.group(1)
                if cur not in body:
                    declared.append(cur)
                    body[cur] = []
                continue
        if cur is not None and re.match(r"^[\s#*>_\-]*(?:citations?|sources?\s*:?\s*$|\[\d+\])", l, re.I) and "Source:" not in l:
            cur = None
        if cur is not None:
            body[cur].append(l)
    stated_none = [y for y in declared if y not in row_years and _SEC_NONE_RE.search(" ".join(body[y]))]
    return declared, stated_none


def audit_citations(text: str, row_lines: Set[int], n_players: int, urls: List[str]) -> Dict[str, Any]:
    lines = [l for i, l in enumerate(text.splitlines()) if i not in row_lines and l.strip()]
    first: Dict[str, int] = {}
    for cat in _CAT_ORDER:
        for idx, l in enumerate(lines):
            if _CAT_RE[cat].search(l) and _SRC_RE.search(l):
                first[cat] = idx
                break
    required = _CAT_ORDER if n_players > 0 else _CAT_ORDER[:3]
    idxs = [first[c] for c in _CAT_ORDER if c in first]
    order_ok = bool(idxs) and all(idxs[i] <= idxs[i + 1] for i in range(len(idxs) - 1))
    return {"categories_present": [c for c in _CAT_ORDER if c in first], "n_categories": len(first),
            "complete": all(c in first for c in required), "order_ok": bool(order_ok), "n_urls": len(urls),
            "n_unique_domains": len({re.sub(r"^https?://(?:www\.)?", "", u).split("/")[0].lower() for u in urls})}


def parse_response(raw_text: str, stalled: bool = False) -> Dict[str, Any]:
    blank = {"S.No": "", "WorldCup_Year": "", "Player": "", "Award_Type": "", "Birthplace": "", "PopCount": ""}
    if not raw_text or raw_text.startswith("ERROR") or raw_text.startswith("DEGENERATE_OUTPUT"):
        return {"rows": [dict(blank, Player="ERROR")], "urls": [], "audit": {}, "declared": [], "stated_none": [],
                "self_excluded": 0}
    rows, row_lines = parse_table(raw_text)
    urls = list(dict.fromkeys(u.rstrip(".,;:") for u in _URL_RE.findall(raw_text)))
    row_years = {r["WorldCup_Year"] for r in rows}
    declared, stated_none = parse_edition_sections(raw_text, row_lines, row_years)
    self_excl = 0
    for l in raw_text.splitlines():
        c = re.sub(r"[*`]+", "", l)
        if _SELF_EXCL_RE.search(c) and (_split_cells(c) or _FREE_ROW_RE.match(c)):
            self_excl += 1
    if not rows:
        none_claim = bool(_NONE_RE.search(raw_text) or stated_none)
        label = ("STALLED" if stalled else
                 "NO QUALIFYING PLAYER" if none_claim else
                 "DECLINED" if detect_decline(raw_text) else "UNPARSED")
        rows = [dict(blank, Player=label)]
    n_players = len([r for r in rows if not is_sentinel_player(r["Player"])])
    return {"rows": rows, "urls": urls, "audit": audit_citations(raw_text, row_lines, n_players, urls),
            "declared": declared, "stated_none": stated_none, "self_excluded": self_excl}


def parse_year_response(raw_text: str, year: int) -> Dict[str, Any]:
    blank = {"S.No": "", "WorldCup_Year": "", "Player": "", "Award_Type": "", "Birthplace": "", "PopCount": ""}
    ys = str(year)
    if raw_text is None or raw_text.startswith("ERROR") or raw_text.startswith("DEGENERATE_OUTPUT"):
        return {"rows": [dict(blank, Player="ERROR")], "label": "ERROR", "answered": False, "stated_none": False,
                "off_year": 0, "self_excluded": 0, "urls": [], "audit": {}}
    if not raw_text.strip():
        return {"rows": [dict(blank, Player="EMPTY")], "label": "EMPTY", "answered": False, "stated_none": False,
                "off_year": 0, "self_excluded": 0, "urls": [], "audit": {}}
    parsed = parse_response(raw_text, stalled=False)
    all_real = [r for r in parsed["rows"] if not is_sentinel_player(r["Player"])]
    kept = [dict(r) for r in all_real if r["WorldCup_Year"] == ys]
    off_year = len(all_real) - len(kept)
    for n, r in enumerate(kept, 1):
        r["S.No"] = str(n)
    if kept:
        label, rows, answered = "", kept, True
    else:
        none_claim = bool(_NONE_RE.search(raw_text)) or bool(parsed["stated_none"]) or parsed.get("self_excluded", 0) > 0
        if none_claim:
            label = "NO QUALIFYING PLAYER"
        elif detect_decline(raw_text):
            label = "DECLINED"
        elif detect_stall(raw_text):
            label = "STALLED"
        elif off_year:
            label = "OFF_YEAR"
        else:
            label = "UNPARSED"
        rows, answered = [dict(blank, Player=label)], label == "NO QUALIFYING PLAYER"
    return {"rows": rows, "label": label, "answered": answered, "stated_none": label == "NO QUALIFYING PLAYER",
            "off_year": off_year, "self_excluded": parsed.get("self_excluded", 0), "urls": parsed["urls"],
            "audit": parsed["audit"]}


def year_record_fields(raw_text: str, year: int) -> Dict[str, Any]:
    p = parse_year_response(raw_text, year)
    is_err = raw_text is None or raw_text.startswith("ERROR")
    deg, why = (False, "") if is_err else detect_degenerate(raw_text)
    real = [r for r in p["rows"] if not is_sentinel_player(r["Player"])]
    return {"Parsed_Rows": p["rows"], "Editions_Parsed": [str(year)] if real else [],
            "Year_Label": p["label"], "Year_Answered": p["answered"], "Year_Stated_None": p["stated_none"],
            "Off_Year_Rows": p["off_year"], "Self_Excluded_Rows": p["self_excluded"],
            "URLs_Cited": p["urls"], "Citation_Audit": p["audit"],
            "Conversational_Stall": p["label"] == "STALLED",
            "Closing_Offer": False if is_err else detect_stall(raw_text),
            "Degenerate_Flag": deg, "Degenerate_Reason": why,
            "Self_Contradiction": False if is_err else detect_contradiction(raw_text, len(real))}


class LatentPathExtractor:
    """Kept for output-schema compatibility: Claude Code returns no per-token logprobs, so every field is 0."""
    _ZERO = {
        "attention_mean": 0.0, "attention_min": 0.0, "attention_max": 0.0, "attention_std": 0.0,
        "attention_entropy": 0.0, "hidden_state_norm_mean": 0.0, "hidden_state_norm_min": 0.0,
        "hidden_state_norm_max": 0.0, "hidden_state_norm_std": 0.0, "token_prob_mean": 0.0,
        "token_prob_min": 0.0, "token_prob_max": 0.0, "token_prob_p10": 0.0, "token_prob_p90": 0.0,
        "confidence_score": 0.0, "perplexity_estimate": 0.0,
    }

    def extract(self, logprobs: Optional[List[float]] = None) -> Dict[str, float]:
        return dict(self._ZERO)


# ---------------------------------------------------------------------------
# CLAUDE CODE CLIENT - streamed headless `claude -p`; every call leaves raw evidence on disk
# ---------------------------------------------------------------------------
_SYSTEM_FLAG_COMPLAINT_RE = re.compile(r"system[- ]prompt|unknown (?:option|argument)|unrecognized (?:option|argument)", re.I)
_BILLING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


def _completion_tokens(usage: Any) -> float:
    if not isinstance(usage, dict):
        return float("nan")
    for k in ("output_tokens", "completion_tokens", "outputTokens", "generation_token_count"):
        if usage.get(k) is not None:
            try:
                return float(usage[k])
            except (TypeError, ValueError):
                pass
    return float("nan")


class ClaudeCodeClient:
    def __init__(self, model_alias: str = DEFAULT_MODEL_ALIAS, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
                 one_shot: bool = True, directive_level: int = 1, per_year: bool = False,
                 claude_bin: str = "claude", timeout: int = REQUEST_TIMEOUT,
                 max_thinking_tokens: Optional[int] = None, no_tool_directive: bool = False):
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{model_alias}'")
        base_levels = DIRECTIVE_LEVELS_PY if per_year else DIRECTIVE_LEVELS
        self.no_tool_directive = bool(no_tool_directive)
        self.levels: Dict[int, str] = ({k: v + NO_TOOL_ADDENDUM for k, v in base_levels.items()}
                                       if self.no_tool_directive else base_levels)
        if directive_level not in self.levels:
            raise ValueError(f"directive_level must be one of {sorted(self.levels)}")
        found = shutil.which(claude_bin)
        if not found:
            raise RuntimeError(f"'{claude_bin}' (Claude Code CLI) not found on PATH")
        self.bin, self.timeout = found, timeout
        self.model_alias, self.model_id, self.family = model_alias, MODEL_REGISTRY[model_alias], FAMILY
        self.visible_tokens = self.max_tokens = int(max_new_tokens)
        self.max_thinking_tokens = max_thinking_tokens
        self.directive_level = directive_level if one_shot else 0
        self.system_text: Optional[str] = self.levels[directive_level] if one_shot else None
        self.system_native = True            # False -> directive is prepended to the user turn instead
        self.dropped: List[str] = []
        self.raw_dir: Optional[Path] = None  # set by main / the analysers: <output>/raw_calls
        self.call_label = "call"             # set by the analysers before each call
        self.workdir = tempfile.mkdtemp(prefix="cc_football_")   # empty cwd: no CLAUDE.md / repo context is loaded
        atexit.register(shutil.rmtree, self.workdir, True)
        logger.info(f"ClaudeCodeClient ready  alias={model_alias}  model={self.model_id}  bin={self.bin}  "
                    f"max_output_tokens={self.max_tokens}  max_thinking_tokens={max_thinking_tokens}  directive="
                    f"{'ON (level ' + str(self.directive_level) + ')' if one_shot else 'OFF (Claude Code default system prompt)'}  "
                    f"no_tool_directive={self.no_tool_directive and one_shot}  "
                    f"mode={'per_year' if per_year else 'single_shot'}")

    @property
    def system_mode(self) -> str:
        return "none" if not self.system_text else ("native" if self.system_native else "prepended")

    def set_directive_level(self, level: int) -> None:
        self.directive_level = level
        self.system_text = self.levels[level]

    def _stem(self, attempt: int) -> Path:
        d = self.raw_dir or Path(self.workdir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.call_label}_L{self.directive_level}_try{attempt}"

    def _run_streaming(self, cmd: List[str], user_text: str, system_text: Optional[str],
                       env: Dict[str, str], stem: Path) -> Dict[str, Any]:
        """Run one call, writing everything to disk AS IT ARRIVES (a failed / killed call still leaves evidence)."""
        p_prompt, p_in = Path(f"{stem}_prompt.txt"), Path(f"{stem}_stdin.tmp")
        p_jsonl, p_text, p_err = Path(f"{stem}_events.jsonl"), Path(f"{stem}_text.txt"), Path(f"{stem}_stderr.txt")
        p_prompt.write_text((f"[SYSTEM]\n{system_text}\n\n" if system_text else "") + "[USER]\n" + user_text, encoding="utf-8")
        p_in.write_text(user_text, encoding="utf-8")
        text_parts: List[str] = []
        assistant_text, think_chars = "", 0
        result_evt: Optional[Dict[str, Any]] = None
        timed_out = {"v": False}
        t0 = time.perf_counter()
        last_log = t0
        with open(p_in, "rb") as fin, open(p_err, "wb") as ferr, \
                open(p_jsonl, "w", encoding="utf-8") as fj, open(p_text, "w", encoding="utf-8") as ft:
            proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=ferr, cwd=self.workdir, env=env,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)

            def _kill():
                timed_out["v"] = True
                proc.kill()

            watchdog = threading.Timer(self.timeout, _kill)
            watchdog.start()
            try:
                for line in proc.stdout:
                    fj.write(line); fj.flush()
                    try:
                        evt = json.loads(line)
                    except ValueError:
                        continue
                    et = evt.get("type")
                    if et == "stream_event":
                        ev = evt.get("event") or {}
                        if ev.get("type") == "content_block_delta":
                            d = ev.get("delta") or {}
                            if d.get("type") == "text_delta":
                                s = d.get("text", "")
                                text_parts.append(s); ft.write(s); ft.flush()
                            elif d.get("type") == "thinking_delta":
                                think_chars += len(d.get("thinking", ""))
                    elif et == "assistant":
                        for b in (evt.get("message") or {}).get("content") or []:
                            if isinstance(b, dict) and b.get("type") == "text":
                                assistant_text += b.get("text", "")
                    elif et == "result":
                        result_evt = evt
                    now = time.perf_counter()
                    if now - last_log >= PROGRESS_LOG_EVERY_SEC:
                        last_log = now
                        logger.info(f"  [{self.call_label}] streaming {now - t0:.0f}s: "
                                    f"text={sum(len(x) for x in text_parts)} chars  thinking={think_chars} chars")
                proc.wait()
            finally:
                watchdog.cancel()
        text = "".join(text_parts) or assistant_text
        if not text and result_evt and not result_evt.get("is_error"):
            text = str(result_evt.get("result") or "")
        if text and not text_parts:
            p_text.write_text(text, encoding="utf-8")
        stderr_text = p_err.read_text(encoding="utf-8", errors="replace")
        blob = stderr_text + " " + (str(result_evt.get("result", "")) if result_evt else "")
        try:
            p_in.unlink()
        except OSError:
            pass
        return {"text": text, "thinking_chars": think_chars, "result_evt": result_evt, "returncode": proc.returncode,
                "stderr": stderr_text, "timed_out": timed_out["v"], "latency": time.perf_counter() - t0,
                "hit_cap": bool(re.search(r"output token maximum", blob, re.I)),
                "files": [str(x) for x in (p_prompt, p_jsonl, p_text, p_err)]}

    def generate(self, prompt: str, max_tokens: Optional[int] = None, use_system: bool = True) -> Dict[str, Any]:
        mt = int(max_tokens or self.max_tokens)
        sysmsg = self.system_text if (use_system and self.system_text) else None
        wait, last_exc = RETRY_BASE_SEC, None
        for attempt in range(1, MAX_RETRIES + 1):
            native = bool(sysmsg and self.system_native)
            user_text = (sysmsg + "\n\n" + prompt) if (sysmsg and not self.system_native) else prompt
            cmd = [self.bin, "-p", "--model", self.model_id, "--output-format", "stream-json", "--verbose",
                   "--include-partial-messages", "--tools", "", "--max-turns", "1"]
            if native:
                cmd += ["--system-prompt", sysmsg]
            env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(mt))
            if self.max_thinking_tokens is not None:
                env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
            r = self._run_streaming(cmd, user_text, sysmsg if native else None, env, self._stem(attempt))
            text = r["text"].strip()
            evt = r["result_evt"] or {}
            usage = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
            base = {"raw_response": text, "response": text, "latency_sec": round(r["latency"], 3), "usage": usage,
                    "cost_usd": evt.get("total_cost_usd"), "max_tokens_sent": mt,
                    "system_mode": self.system_mode if use_system else "none",
                    "thinking_chars": r["thinking_chars"], "text_chars": len(text), "raw_files": r["files"]}
            if r["hit_cap"]:       # deterministic: do NOT retry; keep the partial text and flag it
                logger.warning(f"  [{self.call_label}] hit the {mt}-token output cap after {r['latency']:.0f}s "
                               f"(text={len(text)} chars, thinking={r['thinking_chars']} chars). Not retried. Raw: {r['files'][1]}")
                return dict(base, status=ST_TRUNC_LENGTH, finish_reason="output_token_cap")
            ok = r["returncode"] == 0 and bool(evt) and not evt.get("is_error") and not r["timed_out"]
            if ok:
                finish = evt.get("stop_reason") or evt.get("subtype")
                out_tok = _completion_tokens(usage)
                status = ST_EMPTY if not text else ST_OK
                if text and ((out_tok == out_tok and out_tok >= 0.98 * mt) or str(finish or "").lower() in ("max_tokens", "length")):
                    status = ST_TRUNC_LENGTH
                return dict(base, status=status, finish_reason=None if finish is None else str(finish))
            msg = ("TIMEOUT " if r["timed_out"] else "") + (str(evt.get("result", "")) if evt else "") + " " + r["stderr"][:400]
            if native and _SYSTEM_FLAG_COMPLAINT_RE.search(msg):
                self.system_native = False
                self.dropped.append("system_prompt_flag->prepended_to_user_turn")
                logger.warning(f"--system-prompt rejected ({msg[:120]!r}) - prepending the directive to the user turn")
                continue
            logger.warning(f"claude exit={r['returncode']} attempt {attempt}/{MAX_RETRIES}: {msg[:200]!r} "
                           f"(raw: {r['files'][1]}) - retry in {wait:.0f}s")
            time.sleep(wait); wait *= 2.0; last_exc = RuntimeError(msg[:300])
        raise RuntimeError(f"claude -p failed after {MAX_RETRIES} attempts. Last error: {last_exc}. "
                           f"Raw logs: {self.raw_dir or self.workdir}") from last_exc

    def validate_access(self) -> bool:
        logger.info(f"Validating Claude Code access  alias={self.model_alias}  model={self.model_id} ...")
        self.call_label = "validate"
        res = self.generate("Respond with only the word: OK", 256, use_system=False)
        logger.info(f"Validation OK - response: '{res['response'][:60]}'")
        return True


# ---------------------------------------------------------------------------
# RESTART MANAGER
#   A finished record counts as done only if it was produced with the SAME directive setting AND level, is not an
#   error / legacy placeholder, and (if truncated) was produced with a cap at least as large as the current one.
# ---------------------------------------------------------------------------
class RestartManager:
    def __init__(self, output_file: Path, want_system: bool = True, retry_stalled: bool = False,
                 want_level: int = 1, per_year: bool = False, max_tokens: int = 0):
        self.output_file = output_file
        self.want_system = want_system
        self.retry_stalled = retry_stalled
        self.want_level = want_level
        self.per_year = per_year
        self.max_tokens = max_tokens
        self.processed_ids: Set[int] = set()
        self.existing_results: List[Dict[str, Any]] = []

    def load_existing_results(self) -> bool:
        if not self.output_file.exists():
            logger.info("No existing results file found. Starting fresh.")
            return False
        try:
            with open(self.output_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.existing_results = data.get("results", [])
            for r in self.existing_results:
                qid = r.get("Year_Requested") if self.per_year else r.get("Query_ID")
                answer = str(r.get("Generated_Response", "")).strip()
                if qid is None or not answer or answer.startswith("ERROR"):
                    continue
                if answer.startswith("DEGENERATE_OUTPUT"):
                    continue
                if bool(r.get("system_directive_used", False)) != self.want_system:
                    logger.info("Existing record was produced with a different directive setting - it will be re-run.")
                    continue
                if self.want_system and int(r.get("directive_level_requested", 1)) != self.want_level:
                    logger.info("Existing record used a different directive level - it will be re-run.")
                    continue
                if (r.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                        and int(r.get("max_tokens_sent") or 0) < self.max_tokens):
                    logger.info("Existing record was truncated under a smaller output cap - it will be re-run.")
                    continue
                if self.retry_stalled and r.get("Conversational_Stall"):
                    logger.info("Existing record is a conversational stall - it will be re-run (--retry_stalled).")
                    continue
                self.processed_ids.add(int(qid))
            logger.info(f"Loaded existing results: {len(self.existing_results)} entries, "
                        f"{len(self.processed_ids)} reusable.")
            return len(self.processed_ids) > 0
        except Exception as e:
            logger.warning(f"Could not load existing results ({e}). Starting fresh.")
            return False


def _common_metadata(a: Any, run_mode: str) -> Dict[str, Any]:
    c = a.client
    return {
        "script_version": SCRIPT_VERSION, "run_mode": run_mode, "domain": "football_world_cup",
        "model_alias": a.model_alias, "model_name": a.model_id, "model_family": a.family,
        "model_size": MODEL_SIZE_LABEL.get(a.model_alias, "?"), "inference_backend": "claude_code_cli_cloud_session",
        "query_mode": a.mode, "strategy": a.strategy, "strategy_code": STRATEGY_CODE.get(a.strategy, "?"),
        "strategy_suffix": STRATEGY_SUFFIX.get(a.strategy, "?"),
        "rag_tier": a.rag_tier, "rag_code": RAG_CODE.get(a.rag_tier, "?"),
        "rag_tier_description": RAG_TIER_DESCRIPTION.get(a.rag_tier, ""),
        "rag_max_rows_per_edition": a.rag_max_rows or "all records",
        "output_suffix": a.suffix, "timestamp": datetime.now().isoformat(),
        "max_output_tokens_requested": c.max_tokens, "max_thinking_tokens_env": c.max_thinking_tokens,
        "params_adapted": c.dropped, "request_timeout_s": c.timeout,
        "sampling_note": "temperature / top_p / top_k are not controllable through `claude -p`; the CLI defaults apply",
        "claude_code_call": "claude -p --model <id> --output-format stream-json --verbose --include-partial-messages "
                            "--tools '' --max-turns 1 [--system-prompt <directive>]; cwd = empty temp dir; "
                            "output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; raw events in <output_dir>/raw_calls",
        "one_shot_directive": not a.raw_prompt, "system_mode": c.system_mode,
        "one_shot_system_text": c.system_text, "directive_level": c.directive_level,
        "no_tool_directive": bool(c.no_tool_directive and c.system_text),
        "escalate_on_decline": a.escalate,
        "raw_prompt_note": "RAW = no --system-prompt flag, so Claude Code's default system prompt applies" if a.raw_prompt else "",
        "response_text_policy": "Generated_Response is the model text as returned; never altered",
        "prize_criteria": PRIZE_CRITERIA, "rag_enabled": a.rag_tier != RAG_TIER_NO_EVIDENCE,
        "reference_file": REFERENCE_FILE, "ground_truth_file": GROUND_TRUTH_FILE,
        "citation_note": "Citations are model-asserted; Citation_Audit is a keyword/order heuristic, not source verification",
        "latent_note": "Claude Code returns no per-token logprobs; latent fields are zeros",
        "rag_tiers_available": RAG_TIER_DESCRIPTION,
    }


def _record_common(a: Any, res: Dict[str, Any], raw: str, history: list, start_level: int, prompt: str,
                   block: Optional[str], rag: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": a.mode, "strategy": a.strategy, "strategy_code": STRATEGY_CODE[a.strategy],
        "rag_tier": a.rag_tier, "rag_code": RAG_CODE[a.rag_tier],
        "system_directive_used": res["system_mode"] != "none", "system_mode": res["system_mode"],
        "directive_level": a.client.directive_level, "directive_level_requested": start_level,
        "system_text_used": None if a.raw_prompt else a.client.system_text,
        "Escalation_History": history if len(history) > 1 else None,
        "prompt_sent": prompt,
        "rag_evidence_injected": block is not None, "rag_editions_in_evidence": rag["editions"],
        "rag_rows_available": rag["rows_available"], "rag_rows_injected": rag["rows_injected"],
        "rag_rows_error": rag["rows_error"], "rag_truncated": rag["truncated"],
        "rag_evidence_complete": rag["evidence_complete"], "rag_pop_check_failures": rag["pop_check_failures"],
        "est_prompt_tokens": len(prompt) // 4,
        "model_alias": a.model_alias, "model_id": a.model_id, "model_family": a.family,
        "Generated_Response": raw, "generation_status": res["status"], "finish_reason": res["finish_reason"],
        "max_tokens_sent": res["max_tokens_sent"], "usage": res["usage"], "cost_usd_reported": res.get("cost_usd"),
        "thinking_chars": res.get("thinking_chars"), "text_chars": res.get("text_chars"),
        "raw_files": res.get("raw_files"),
        "generation_time_seconds": res["latency_sec"], "processing_timestamp": datetime.now().isoformat(),
    }


_EMPTY_RAG = {"editions": 0, "rows_available": 0, "rows_injected": 0, "rows_error": 0,
              "truncated": False, "pop_check_failures": 0, "evidence_complete": False}


# ---------------------------------------------------------------------------
# ANALYSER (single-shot) - one call per (model, strategy, RAG tier)
# ---------------------------------------------------------------------------
class WorldCupPlayerAnalyser:
    def __init__(self, strategy: str, client: ClaudeCodeClient, no_resume: bool = False,
                 rag_tier: str = RAG_TIER_NO_EVIDENCE, ref_store: Optional[ReferenceStore] = None,
                 gt_store: Optional[GroundTruthStore] = None, rag_max_rows: int = 0,
                 raw_prompt: bool = False, retry_stalled: bool = False, escalate_on_decline: bool = False):
        self.mode = QUERY_MODE_NLQ_GENERAL
        if strategy not in INDIVIDUAL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Use one of: {INDIVIDUAL_STRATEGIES}")
        self.strategy, self.no_resume, self.raw_prompt = strategy, no_resume, raw_prompt
        self.escalate = escalate_on_decline and not raw_prompt
        self.client = client
        self.model_alias, self.model_id, self.family = client.model_alias, client.model_id, client.family
        self.model_slug = model_id_to_slug(self.model_id)
        self.rag_tier, self.ref_store, self.gt_store = rag_tier, ref_store, gt_store
        self.rag_max_rows = rag_max_rows
        dsuf = "" if raw_prompt else ((f"_D{client.directive_level}" if client.directive_level > 1 else "") +
                                      ("_ESC" if self.escalate else ""))
        self.suffix = (f"{MODE_SUFFIX[self.mode]}_{STRATEGY_SUFFIX[strategy]}{RAG_TIER_SUFFIX.get(rag_tier, '')}"
                       f"{dsuf}{'_RAW' if raw_prompt else ''}")
        self.output_dir = OUTPUT_ROOT / f"{self.model_slug}_{self.suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.model_slug}_{self.suffix}.json"
        self.response_txt = self.output_dir / f"{self.model_slug}_{self.suffix}_response.txt"
        if no_resume and self.output_file.exists():
            logger.info("--no-resume: deleting existing output and reprocessing from scratch.")
            self.output_file.unlink()
        self.restart_manager = RestartManager(self.output_file, want_system=not raw_prompt,
                                              retry_stalled=retry_stalled, want_level=max(1, client.directive_level),
                                              max_tokens=client.max_tokens)
        if self.restart_manager.load_existing_results() and not no_resume:
            logger.info(f"RESUME MODE - query already done: {sorted(self.restart_manager.processed_ids)}")
        self._latent_extractor = LatentPathExtractor()

    def _save(self, results: List[Dict[str, Any]]) -> None:
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        md = _common_metadata(self, "single_shot")
        md.update({"total_queries": len(results), "successful_records": len(results) - errors, "error_records": errors,
                   "prompt_mode": "user_turn_verbatim_with_literal_placeholder"
                                  + ("" if self.raw_prompt else " + system directive")
                                  + ("" if self.strategy == STRATEGY_FLAT_CONJUNCTIVE else " + strategy addendum")
                                  + ("" if self.rag_tier == RAG_TIER_NO_EVIDENCE else " + evidence block")})
        output = {"metadata": md, "results": sorted(results, key=lambda x: x["Query_ID"])}
        with open(self.output_file, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results -> {self.output_file}")

    def process(self) -> List[Dict[str, Any]]:
        if QUERY_ID in self.restart_manager.processed_ids:
            logger.info("Query already processed. Nothing to do.")
            return self.restart_manager.existing_results
        results = [r for r in self.restart_manager.existing_results if int(r.get("Query_ID", -1)) != QUERY_ID
                   and not str(r.get("Generated_Response", "")).startswith("ERROR")]
        start_level = self.client.directive_level
        self.client.raw_dir = self.output_dir / "raw_calls"
        self.client.call_label = f"{self.model_slug}_{self.suffix}_all"
        logger.info(f"Sending the prompt  strategy={STRATEGY_CODE[self.strategy]}  rag={RAG_CODE[self.rag_tier]}  "
                    f"model={self.model_id}  directive={'OFF' if self.raw_prompt else 'level ' + str(start_level)}  "
                    f"escalate_on_decline={self.escalate}")
        try:
            block, rag = None, dict(_EMPTY_RAG)
            if self.rag_tier != RAG_TIER_NO_EVIDENCE:
                block, rag = build_rag_evidence_block(self.rag_tier, self.ref_store, self.gt_store, self.rag_max_rows)
                if block is None:
                    logger.warning(f"No evidence available for tier '{self.rag_tier}' - prompt sent without it")
            prompt = build_prompt(self.strategy, block)

            def _attempt():
                r = self.client.generate(prompt, use_system=not self.raw_prompt)
                t = r["response"]
                st = detect_stall(t)
                return r, t, st, parse_response(t, stalled=st)

            def _needs_escalation(p: Dict[str, Any]) -> bool:
                return p["rows"][0]["Player"] in ("DECLINED", "STALLED")

            res, raw, stall, parsed = _attempt()
            history = [{"level": self.client.directive_level, "label": parsed["rows"][0]["Player"],
                        "declined": _needs_escalation(parsed), "status": res["status"],
                        "completion_tokens": _completion_tokens(res["usage"]), "text": raw}]
            while (self.escalate and not self.raw_prompt and _needs_escalation(parsed)
                   and self.client.directive_level < max(self.client.levels)):
                new_level = self.client.directive_level + 1
                logger.warning(f"  {parsed['rows'][0]['Player']} at directive level {self.client.directive_level} "
                               f"- re-sending at level {new_level}")
                self.client.set_directive_level(new_level)
                res, raw, stall, parsed = _attempt()
                history.append({"level": new_level, "label": parsed["rows"][0]["Player"],
                                "declined": _needs_escalation(parsed), "status": res["status"],
                                "completion_tokens": _completion_tokens(res["usage"]), "text": raw})
            degenerate, deg_reason = detect_degenerate(raw)
            rows, audit = parsed["rows"], parsed["audit"]
            real = [r for r in rows if not is_sentinel_player(r["Player"])]
            contradiction = detect_contradiction(raw, len(real))
            declined = (not real) and rows[0]["Player"] == "DECLINED"
            try:
                self.response_txt.write_text(raw, encoding="utf-8")
            except OSError as e:
                logger.warning(f"Could not write {self.response_txt}: {e}")
            record: Dict[str, Any] = {"Query_ID": QUERY_ID, "run_mode": "single_shot"}
            record.update(_record_common(self, res, raw, history, start_level, prompt, block, rag))
            record.update({
                "prompt_is_verbatim": (self.strategy == STRATEGY_FLAT_CONJUNCTIVE and block is None
                                       and res["system_mode"] != "prepended"),
                "Conversational_Stall": stall, "Degenerate_Flag": degenerate, "Degenerate_Reason": deg_reason,
                "Self_Contradiction": contradiction, "Self_Excluded_Rows": parsed.get("self_excluded", 0),
                "Parsed_Rows": rows,
                "Editions_Parsed": sorted({r["WorldCup_Year"] for r in real}),
                "Editions_Declared": parsed["declared"], "Editions_Stated_None": parsed["stated_none"],
                "URLs_Cited": parsed["urls"], "Citation_Audit": audit,
            })
            record.update(self._latent_extractor.extract(None))
            results.append(record)
            logger.info(f"  OK {res['latency_sec']:.2f}s  status={res['status']}  rows={len(real)}  "
                        f"editions_with_rows={len(record['Editions_Parsed'])}  declared={len(parsed['declared'])}  "
                        f"stated_none={len(parsed['stated_none'])}  self_excluded={parsed.get('self_excluded', 0)}  "
                        f"stall={stall}  declined={declined}  contradiction={contradiction}  degenerate_tail={degenerate}  "
                        f"system_mode={res['system_mode']}  level={self.client.directive_level}  attempts={len(history)}  "
                        f"cit_complete={audit.get('complete')}  urls={len(parsed['urls'])}  usage={res['usage']}  "
                        f"text_chars={res.get('text_chars')}  thinking_chars={res.get('thinking_chars')}  "
                        f"cost_usd={res.get('cost_usd')}")
            if len(history) > 1:
                logger.info("  escalation history: " + "; ".join(
                    f"L{h['level']}:{h['label'] if h['declined'] else 'answered'}({h['completion_tokens']} tok)" for h in history))
            if stall:
                logger.warning("  CONVERSATIONAL STALL: the response ends by waiting for a further turn")
            if declined:
                logger.warning("  DECLINED: the model said the task cannot be completed in one response (no rows)")
            if contradiction:
                logger.warning("  SELF-CONTRADICTION: rows exist but the text says nobody from that edition qualified")
            if degenerate:
                logger.warning(f"  Possible repetition loop at the end ({deg_reason}) - flagged only, text kept")
            if res["status"] in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING):
                logger.warning("  Output was TRUNCATED - later editions are probably missing (see --max_new_tokens)")
            if not real and not stall and not declined:
                logger.warning("  No table rows were parsed. Inspect the raw_calls files; --reparse re-parses offline.")
            if self.client.dropped:
                logger.warning(f"  Call parameters were adapted: {self.client.dropped}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            err: Dict[str, Any] = {
                "Query_ID": QUERY_ID, "mode": self.mode, "run_mode": "single_shot", "strategy": self.strategy,
                "strategy_code": STRATEGY_CODE[self.strategy], "rag_tier": self.rag_tier, "rag_code": RAG_CODE[self.rag_tier],
                "system_directive_used": not self.raw_prompt, "system_mode": "n/a",
                "directive_level": start_level, "directive_level_requested": start_level,
                "Conversational_Stall": False, "Degenerate_Flag": False, "Degenerate_Reason": "",
                "Self_Contradiction": False, "Self_Excluded_Rows": 0, "rag_evidence_injected": False,
                "model_alias": self.model_alias, "model_id": self.model_id, "model_family": self.family,
                "Generated_Response": f"ERROR: {e}", "Parsed_Rows": [], "Editions_Parsed": [],
                "Editions_Declared": [], "Editions_Stated_None": [], "URLs_Cited": [], "Citation_Audit": {},
                "generation_status": "error", "generation_time_seconds": 0.0,
                "processing_timestamp": datetime.now().isoformat(),
            }
            err.update(self._latent_extractor.extract(None))
            results.append(err)
        finally:
            if not self.raw_prompt and start_level in self.client.levels:
                self.client.set_directive_level(start_level)
        self._save(results)
        logger.info(f"SUMMARY  model={self.model_alias}  strategy={STRATEGY_CODE[self.strategy]}  "
                    f"rag={RAG_CODE[self.rag_tier]}  output -> {self.output_file}")
        return results


# ---------------------------------------------------------------------------
# ANALYSER (per year) - one call per (model, strategy, RAG tier, YEAR)
# ---------------------------------------------------------------------------
class WorldCupYearLoopAnalyser:
    def __init__(self, strategy: str, client: ClaudeCodeClient, years: List[int], no_resume: bool = False,
                 rag_tier: str = RAG_TIER_NO_EVIDENCE, ref_store: Optional[ReferenceStore] = None,
                 gt_store: Optional[GroundTruthStore] = None, rag_max_rows: int = 0,
                 raw_prompt: bool = False, retry_stalled: bool = False, escalate_on_decline: bool = False):
        self.mode = QUERY_MODE_PER_YEAR
        if strategy not in INDIVIDUAL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Use one of: {INDIVIDUAL_STRATEGIES}")
        self.strategy, self.no_resume, self.raw_prompt = strategy, no_resume, raw_prompt
        self.years = [int(y) for y in years]
        self.escalate = escalate_on_decline and not raw_prompt
        self.client = client
        self.model_alias, self.model_id, self.family = client.model_alias, client.model_id, client.family
        self.model_slug = model_id_to_slug(self.model_id)
        self.rag_tier, self.ref_store, self.gt_store = rag_tier, ref_store, gt_store
        self.rag_max_rows = rag_max_rows
        dsuf = "" if raw_prompt else ((f"_D{client.directive_level}" if client.directive_level > 1 else "") +
                                      ("_ESC" if self.escalate else ""))
        self.suffix = (f"{MODE_SUFFIX[self.mode]}_{STRATEGY_SUFFIX[strategy]}{RAG_TIER_SUFFIX.get(rag_tier, '')}"
                       f"{dsuf}{'_RAW' if raw_prompt else ''}")
        self.output_dir = OUTPUT_ROOT_PY / f"{self.model_slug}_{self.suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.model_slug}_{self.suffix}.json"
        self.response_txt = self.output_dir / f"{self.model_slug}_{self.suffix}_response.txt"
        if no_resume and self.output_file.exists():
            logger.info("--no-resume: deleting existing per-year output and reprocessing from scratch.")
            self.output_file.unlink()
        self.restart_manager = RestartManager(self.output_file, want_system=not raw_prompt, retry_stalled=retry_stalled,
                                              want_level=max(1, client.directive_level), per_year=True,
                                              max_tokens=client.max_tokens)
        if self.restart_manager.load_existing_results() and not no_resume:
            logger.info(f"RESUME MODE - years already done: {sorted(self.restart_manager.processed_ids)}")
        self._latent_extractor = LatentPathExtractor()

    def _metadata(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        errors = sum(1 for r in results.values() if str(r.get("Generated_Response", "")).startswith("ERROR"))
        md = _common_metadata(self, "per_year")
        md.update({"years_requested": self.years, "total_queries": len(results),
                   "successful_records": len(results) - errors, "error_records": errors,
                   "prompt_mode": "user_turn_with_placeholder_replaced_by_year"
                                  + ("" if self.raw_prompt else " + per-year system directive")
                                  + ("" if self.strategy == STRATEGY_FLAT_CONJUNCTIVE else " + strategy addendum")
                                  + ("" if self.rag_tier == RAG_TIER_NO_EVIDENCE else " + evidence block (that year only)")})
        return md

    def _save(self, results: Dict[int, Dict[str, Any]]) -> None:
        output = {"metadata": self._metadata(results), "results": [results[y] for y in sorted(results)]}
        with open(self.output_file, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        parts = [f"===== YEAR {y} =====\n{results[y].get('Generated_Response', '')}" for y in sorted(results)]
        try:
            self.response_txt.write_text("\n\n".join(parts), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not write {self.response_txt}: {e}")

    def _run_year(self, year: int) -> Dict[str, Any]:
        start_level = self.client.directive_level
        self.client.raw_dir = self.output_dir / "raw_calls"
        self.client.call_label = f"{self.model_slug}_{self.suffix}_{year}"
        try:
            block, rag = None, dict(_EMPTY_RAG)
            if self.rag_tier != RAG_TIER_NO_EVIDENCE:
                block, rag = build_rag_evidence_block(self.rag_tier, self.ref_store, self.gt_store,
                                                      self.rag_max_rows, only_year=str(year))
                if block is None:
                    logger.warning(f"  [{year}] no evidence for tier '{self.rag_tier}' in the evidence file - prompt sent without it")
            prompt = build_prompt(self.strategy, block, year)

            def _attempt():
                r = self.client.generate(prompt, use_system=not self.raw_prompt)
                t = r["response"]
                return r, t, year_record_fields(t, year)

            def _needs_escalation(f: Dict[str, Any]) -> bool:
                return f["Year_Label"] in ("DECLINED", "STALLED")

            res, raw, fields = _attempt()
            history = [{"level": self.client.directive_level, "label": fields["Year_Label"] or "ROWS",
                        "status": res["status"], "completion_tokens": _completion_tokens(res["usage"]), "text": raw}]
            while (self.escalate and not self.raw_prompt and _needs_escalation(fields)
                   and self.client.directive_level < max(self.client.levels)):
                new_level = self.client.directive_level + 1
                logger.warning(f"  [{year}] {fields['Year_Label']} at directive level {self.client.directive_level} "
                               f"- re-sending at level {new_level}")
                self.client.set_directive_level(new_level)
                res, raw, fields = _attempt()
                history.append({"level": new_level, "label": fields["Year_Label"] or "ROWS", "status": res["status"],
                                "completion_tokens": _completion_tokens(res["usage"]), "text": raw})
            record: Dict[str, Any] = {"Query_ID": year, "Year_Requested": year, "run_mode": "per_year",
                                      "prompt_is_verbatim": False, "prompt_year_substituted": True}
            record.update(_record_common(self, res, raw, history, start_level, prompt, block, rag))
            record.update(fields)
            record.update(self._latent_extractor.extract(None))
            n_rows = len([r for r in fields["Parsed_Rows"] if not is_sentinel_player(r["Player"])])
            logger.info(f"  [{year}] OK {res['latency_sec']:.1f}s status={res['status']} label={fields['Year_Label'] or 'ROWS'} "
                        f"rows={n_rows} off_year={fields['Off_Year_Rows']} self_excluded={fields['Self_Excluded_Rows']} "
                        f"contradiction={fields['Self_Contradiction']} level={self.client.directive_level} "
                        f"attempts={len(history)} tokens={_completion_tokens(res['usage'])} "
                        f"text_chars={res.get('text_chars')} thinking_chars={res.get('thinking_chars')} "
                        f"cost_usd={res.get('cost_usd')}")
            if res["status"] in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING):
                logger.warning(f"  [{year}] output TRUNCATED (raw files: {res.get('raw_files')})")
            return record
        except Exception as e:
            logger.error(f"  [{year}] query failed: {e}")
            err: Dict[str, Any] = {
                "Query_ID": year, "Year_Requested": year, "mode": self.mode, "run_mode": "per_year",
                "strategy": self.strategy, "strategy_code": STRATEGY_CODE[self.strategy],
                "rag_tier": self.rag_tier, "rag_code": RAG_CODE[self.rag_tier],
                "system_directive_used": not self.raw_prompt, "system_mode": "n/a",
                "directive_level": start_level, "directive_level_requested": start_level,
                "rag_evidence_injected": False, "model_alias": self.model_alias, "model_id": self.model_id,
                "model_family": self.family, "Generated_Response": f"ERROR: {e}",
                "generation_status": "error", "generation_time_seconds": 0.0,
                "processing_timestamp": datetime.now().isoformat(),
            }
            err.update(year_record_fields(err["Generated_Response"], year))
            err.update(self._latent_extractor.extract(None))
            return err
        finally:
            if not self.raw_prompt and start_level in self.client.levels:
                self.client.set_directive_level(start_level)

    def process(self) -> List[Dict[str, Any]]:
        reusable = self.restart_manager.processed_ids
        by_year: Dict[int, Dict[str, Any]] = {int(r["Year_Requested"]): r for r in self.restart_manager.existing_results
                                              if r.get("Year_Requested") is not None and int(r["Year_Requested"]) in reusable}
        todo = [y for y in self.years if y not in by_year]
        logger.info(f"Per-year run  strategy={STRATEGY_CODE[self.strategy]}  rag={RAG_CODE[self.rag_tier]}  "
                    f"model={self.model_id}  directive={'OFF' if self.raw_prompt else 'level ' + str(self.client.directive_level)}  "
                    f"escalate={self.escalate}  years to run: {len(todo)} of {len(self.years)} "
                    f"({len(self.years) - len(todo)} already done)")
        consecutive_errors = 0
        for y in todo:
            rec = self._run_year(y)
            by_year[y] = rec
            self._save(by_year)
            if rec.get("generation_status") == "error":
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_YEAR_ERRORS:
                    logger.error(f"{consecutive_errors} consecutive failed calls - aborting this cell "
                                 f"(finished years are saved; re-run to retry the rest)")
                    break
            else:
                consecutive_errors = 0
        if not todo:
            self._save(by_year)
        n_ok = sum(1 for r in by_year.values() if not str(r.get("Generated_Response", "")).startswith("ERROR"))
        logger.info(f"SUMMARY  model={self.model_alias}  strategy={STRATEGY_CODE[self.strategy]}  rag={RAG_CODE[self.rag_tier]}  "
                    f"years ok={n_ok}/{len(self.years)}  output -> {self.output_file}")
        return [by_year[y] for y in sorted(by_year)]


# ---------------------------------------------------------------------------
# REPARSE - re-run the parser over STORED response text (no Claude calls); both run modes
# ---------------------------------------------------------------------------
def reparse_outputs(root: Path = OUTPUT_ROOT) -> int:
    n = 0
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME) or jf.parent.name.startswith("diagnosis"):
            continue
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning(f"Reparse: cannot read {jf} ({e})")
            continue
        per_year = (data.get("metadata", {}) or {}).get("run_mode") == "per_year"
        changed = False
        for r in data.get("results", []):
            txt = str(r.get("Generated_Response", ""))
            if not txt or txt.startswith("ERROR") or txt.startswith("DEGENERATE_OUTPUT"):
                continue
            if per_year:
                if r.get("Year_Requested") is None:
                    continue
                r.update(year_record_fields(txt, int(r["Year_Requested"])))
            else:
                stall = detect_stall(txt)
                deg, why = detect_degenerate(txt)
                parsed = parse_response(txt, stalled=stall)
                real = [x for x in parsed["rows"] if not is_sentinel_player(x["Player"])]
                r.update({"Parsed_Rows": parsed["rows"], "Editions_Parsed": sorted({x["WorldCup_Year"] for x in real}),
                          "Editions_Declared": parsed["declared"], "Editions_Stated_None": parsed["stated_none"],
                          "URLs_Cited": parsed["urls"], "Citation_Audit": parsed["audit"],
                          "Conversational_Stall": stall, "Degenerate_Flag": deg, "Degenerate_Reason": why,
                          "Self_Contradiction": detect_contradiction(txt, len(real)),
                          "Self_Excluded_Rows": parsed.get("self_excluded", 0)})
            n += 1
            changed = True
        if changed:
            with open(jf, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info(f"Reparse: {n} stored response(s) re-parsed under {root} (no model calls).")
    return n


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING - computed only from the saved outputs (ground truth optional)
# ---------------------------------------------------------------------------
def aggregate_year_records(meta: Dict[str, Any], year_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    recs = sorted([dict(r) for r in year_recs if r.get("Year_Requested") is not None],
                  key=lambda r: int(r["Year_Requested"]))
    n = max(1, len(recs))

    def is_err(r: Dict[str, Any]) -> bool:
        return str(r.get("Generated_Response", "")).startswith("ERROR") or r.get("Year_Label") == "ERROR"

    good = [r for r in recs if not is_err(r)]
    all_rows: List[Dict[str, Any]] = []
    declared: List[str] = []
    none_years: List[str] = []
    urls: List[str] = []
    texts: List[str] = []
    for r in recs:
        y = str(r["Year_Requested"])
        texts.append(f"===== YEAR {y} =====\n{r.get('Generated_Response', '')}")
        if is_err(r):
            continue
        real = [dict(x) for x in (r.get("Parsed_Rows") or []) if not is_sentinel_player(x.get("Player", ""))]
        all_rows += real
        if r.get("Year_Answered"):
            declared.append(y)
        if r.get("Year_Stated_None"):
            none_years.append(y)
        urls += list(r.get("URLs_Cited") or [])
    for i, x in enumerate(all_rows, 1):
        x["S.No"] = str(i)
    if not all_rows:
        labels = Counter(r.get("Year_Label") for r in good if r.get("Year_Label"))
        lab = labels.most_common(1)[0][0] if labels else ("ERROR" if not good else "UNPARSED")
        all_rows = [{"S.No": "", "WorldCup_Year": "", "Player": lab, "Award_Type": "", "Birthplace": "", "PopCount": ""}]

    def share(pred) -> float:
        return sum(1 for r in recs if pred(r)) / n

    trunc = lambda r: r.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
    toks = [t for t in (_completion_tokens(r.get("usage")) for r in good) if t == t]
    audits = [r.get("Citation_Audit") or {} for r in good if r.get("Year_Answered") and r.get("Citation_Audit")]
    inj = [r for r in good if r.get("rag_evidence_injected")]
    status = ("error" if not good else (ST_TRUNC_LENGTH if any(trunc(r) for r in good) else ST_OK))
    esc = [{"year": r["Year_Requested"], "history": r["Escalation_History"]} for r in recs if r.get("Escalation_History")]
    first = good[0] if good else {}
    return {
        "model_alias": meta.get("model_alias"), "strategy": meta.get("strategy"), "rag_tier": meta.get("rag_tier"),
        "Query_ID": 1, "run_mode": "per_year", "prompt_is_verbatim": False,
        "system_mode": first.get("system_mode", "none"),
        "system_directive_used": any(bool(r.get("system_directive_used")) for r in recs),
        "directive_level": first.get("directive_level", meta.get("directive_level", 1)),
        "Escalation_History": esc or None,
        "Generated_Response": ("\n\n".join(texts) if good else "ERROR: every year failed"),
        "Parsed_Rows": all_rows,
        "Editions_Declared": declared, "Editions_Stated_None": none_years,
        "URLs_Cited": list(dict.fromkeys(urls)),
        "Citation_Audit": {"complete_share": (float(np.mean([bool(a.get("complete")) for a in audits])) if audits else 0.0),
                           "order_ok_share": (float(np.mean([bool(a.get("order_ok")) for a in audits])) if audits else 0.0),
                           "n_urls": sum(int(a.get("n_urls", 0) or 0) for a in audits)},
        "generation_status": status,
        "usage": {"output_tokens": float(sum(toks))} if toks else None,
        "generation_time_seconds": float(sum(float(r.get("generation_time_seconds") or 0.0) for r in recs)),
        "Rate_Stalled": share(lambda r: r.get("Year_Label") == "STALLED"),
        "Rate_Declined": share(lambda r: r.get("Year_Label") == "DECLINED"),
        "Rate_Unparsed": share(lambda r: r.get("Year_Label") in ("UNPARSED", "OFF_YEAR", "EMPTY")),
        "Rate_Truncated": share(lambda r: (not is_err(r)) and trunc(r)),
        "Rate_Error": share(is_err),
        "Rate_Degenerate": share(lambda r: bool(r.get("Degenerate_Flag"))),
        "Rate_Contradiction": share(lambda r: bool(r.get("Self_Contradiction"))),
        "Self_Excluded_Rows": sum(int(r.get("Self_Excluded_Rows", 0) or 0) for r in good),
        "Off_Year_Rows": sum(int(r.get("Off_Year_Rows", 0) or 0) for r in good),
        "Years_Requested": len(recs), "Years_Answered": len(declared),
        "Years_Declined": sum(1 for r in recs if r.get("Year_Label") == "DECLINED"),
        "Years_Stalled": sum(1 for r in recs if r.get("Year_Label") == "STALLED"),
        "Years_Error": sum(1 for r in recs if is_err(r)),
        "rag_evidence_injected": bool(inj),
        "rag_evidence_complete": bool(inj) and all(bool(r.get("rag_evidence_complete")) for r in inj),
        "rag_pop_check_failures": sum(int(r.get("rag_pop_check_failures", 0) or 0) for r in good),
        "_year_records": recs,
    }


def collect_records(root: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME) or jf.parent.name.startswith("diagnosis"):
            continue
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning(f"Analysis: cannot read {jf} ({e})")
            continue
        if "results" not in data:
            continue
        md = data.get("metadata", {})
        if md.get("run_mode") == "per_year":
            if data["results"]:
                recs.append(aggregate_year_records(md, data["results"]))
            continue
        for r in data["results"]:
            r = dict(r)
            r.setdefault("model_alias", md.get("model_alias"))
            r.setdefault("strategy", md.get("strategy"))
            r.setdefault("rag_tier", md.get("rag_tier"))
            recs.append(r)
    return recs


def _gt_by_year(gt_store: GroundTruthStore) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for y, recs in gt_store.by_year.items():
        if any(is_error_row(r.player) for r in recs):
            continue
        merged: Dict[str, Dict[str, Any]] = {}
        for r in recs:
            if is_sentinel_player(r.player):
                continue
            d = merged.setdefault(name_key(r.player), {"player": r.player, "awards": set(),
                                                        "birthplace": r.birthplace, "pop": gt_pop_value(r.pop_raw)})
            d["awards"] |= award_set(r.award_type)
            if not d["birthplace"]:
                d["birthplace"] = r.birthplace
            if d["pop"] is None:
                d["pop"] = gt_pop_value(r.pop_raw)
        out[y] = list(merged.values())
    return out


def match_pairs(g_list: List[Dict[str, Any]], p_list: List[Dict[str, Any]]):
    rem_gt, unm, pairs = list(range(len(g_list))), list(range(len(p_list))), []
    for exact in (True, False):
        for gi in list(rem_gt):
            for pi in list(unm):
                a, b = g_list[gi]["player"], p_list[pi]["Player"]
                if (name_key(a) == name_key(b)) if exact else names_match(a, b):
                    pairs.append((gi, pi)); rem_gt.remove(gi); unm.remove(pi)
                    break
    return pairs, rem_gt, unm


def _rate(rec: Dict[str, Any], key: str, default: Any) -> float:
    v = rec.get(key)
    return float(v) if v is not None else float(bool(default))


def evaluate_record(rec: Dict[str, Any], gt_by_year: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    resp = str(rec.get("Generated_Response", ""))
    is_error = resp.startswith("ERROR") or resp.startswith("DEGENERATE_OUTPUT")
    status = rec.get("generation_status", "")
    audit = rec.get("Citation_Audit") or {}
    rows = [] if is_error else (rec.get("Parsed_Rows") or [])
    pred = [r for r in rows if not is_sentinel_player(r.get("Player", ""))]
    sentinel = (rows[0].get("Player", "") if (rows and not pred) else "")
    pops = [str(r.get("PopCount", "")) for r in pred]
    aw = [award_set(r.get("Award_Type", "")) for r in pred]
    ycount = Counter(r["WorldCup_Year"] for r in pred)
    years_pred = [int(r["WorldCup_Year"]) for r in pred if str(r["WorldCup_Year"]).isdigit()]
    declared = [] if is_error else [int(y) for y in (rec.get("Editions_Declared") or []) if str(y).isdigit()]
    stated_none = set() if is_error else set(str(y) for y in (rec.get("Editions_Stated_None") or []))
    all_years = years_pred + declared
    last_year = max(all_years) if all_years else None
    cc, oo = audit.get("complete_share"), audit.get("order_ok_share")
    out: Dict[str, Any] = {
        "model_alias": rec.get("model_alias"), "strategy": rec.get("strategy"), "rag_tier": rec.get("rag_tier"),
        "Query_ID": rec.get("Query_ID"), "run_mode": rec.get("run_mode", "single_shot"),
        "prompt_is_verbatim": bool(rec.get("prompt_is_verbatim", False)),
        "system_mode": rec.get("system_mode", "none"),
        "directive_level": rec.get("directive_level", 1 if rec.get("system_directive_used") else 0),
        "escalated": bool(rec.get("Escalation_History")),
        "stalled": _rate(rec, "Rate_Stalled", rec.get("Conversational_Stall", False)),
        "degenerate": _rate(rec, "Rate_Degenerate", rec.get("Degenerate_Flag", False)),
        "contradiction": _rate(rec, "Rate_Contradiction", rec.get("Self_Contradiction", False)),
        "declined": _rate(rec, "Rate_Declined", sentinel == "DECLINED"),
        "unparsed": _rate(rec, "Rate_Unparsed", sentinel == "UNPARSED"),
        "truncated": _rate(rec, "Rate_Truncated", status in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)),
        "error_share": _rate(rec, "Rate_Error", is_error),
        "self_excluded_n": int(rec.get("Self_Excluded_Rows", 0) or 0),
        "off_year_n": int(rec.get("Off_Year_Rows", 0) or 0),
        "years_requested": int(rec.get("Years_Requested", 0) or 0), "years_answered": int(rec.get("Years_Answered", 0) or 0),
        "years_declined": int(rec.get("Years_Declined", 0) or 0), "years_error": int(rec.get("Years_Error", 0) or 0),
        "status": status, "is_error": is_error,
        "latency": rec.get("generation_time_seconds"), "completion_tokens": _completion_tokens(rec.get("usage")),
        "sentinel": sentinel, "n_pred": len(pred), "n_editions": len(ycount), "year_counts": dict(ycount),
        "n_declared": len(set(declared)), "false_none_n": 0, "true_none_n": 0,
        "last_year": float(last_year) if last_year is not None else np.nan,
        "pairs": [(r["WorldCup_Year"], r["Player"]) for r in pred],
        "pop_numeric_n": sum(p.isdigit() for p in pops), "pop_total_n": len(pops),
        "pop_values": [int(p) for p in pops if p.isdigit()],
        "birth_blank_n": sum(1 for r in pred if not str(r.get("Birthplace", "")).strip()
                             or str(r.get("Birthplace", "")).strip().lower() == "unknown"),
        "award_ballon_n": sum("BALLON_DOR" in a for a in aw), "award_uefa_n": sum("UEFA_POTY" in a for a in aw),
        "award_unrecognised_n": sum(1 for a in aw if not a),
        "citation_complete": float(cc) if cc is not None else float(bool(audit.get("complete", False))),
        "citation_order_ok": float(oo) if oo is not None else float(bool(audit.get("order_ok", False))),
        "n_urls": audit.get("n_urls", 0) or 0, "urls": list(rec.get("URLs_Cited") or []),
        "rag_evidence_injected": bool(rec.get("rag_evidence_injected", False)),
        "rag_evidence_complete": bool(rec.get("rag_evidence_complete", False)),
        "rag_pop_check_failures": rec.get("rag_pop_check_failures", 0) or 0,
        "tp": 0, "n_pred_scored": 0, "n_gt": 0, "tp_r": 0, "n_gt_r": 0, "n_ps_r": 0,
        "precision": np.nan, "recall": np.nan, "f1": np.nan, "scored": False,
        "award_ok": 0, "award_n": 0, "birth_ok": 0, "birth_n": 0, "pop_5_n": 0, "pop_cmp_n": 0,
        "fn_players": [], "fp_players": [], "pop_log_ratios": [],
    }
    if not gt_by_year or is_error:
        return out
    pred_by_year: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in pred:
        pred_by_year[r["WorldCup_Year"]].append(r)
    tp = n_gt = n_ps = tp_r = n_gt_r = n_ps_r = 0
    for y, g_list in gt_by_year.items():
        p_list = pred_by_year.get(y, [])
        if y in stated_none:
            if g_list:
                out["false_none_n"] += 1
            else:
                out["true_none_n"] += 1
        pairs, rem_gt, unm = match_pairs(g_list, p_list)
        tp += len(pairs); n_gt += len(g_list); n_ps += len(p_list)
        if last_year is not None and str(y).isdigit() and int(y) <= last_year:
            tp_r += len(pairs); n_gt_r += len(g_list); n_ps_r += len(p_list)
        for gi, pi in pairs:
            g, p = g_list[gi], p_list[pi]
            out["award_n"] += 1
            out["award_ok"] += int(award_set(p.get("Award_Type", "")) == g["awards"])
            out["birth_n"] += 1
            out["birth_ok"] += int(places_match(p.get("Birthplace", ""), g["birthplace"]))
            pp = str(p.get("PopCount", "")).strip()
            if pp.isdigit() and g["pop"]:
                pv = int(pp)
                out["pop_cmp_n"] += 1
                out["pop_5_n"] += int(abs(pv - g["pop"]) / g["pop"] <= 0.05)
                if pv > 0:
                    out["pop_log_ratios"].append(math.log10(pv / g["pop"]))
        out["fn_players"] += [f"{y} {g_list[gi]['player']}" for gi in rem_gt]
        out["fp_players"] += [f"{y} {p_list[pi]['Player']}" for pi in unm]
    precision = tp / n_ps if n_ps else np.nan
    recall = tp / n_gt if n_gt else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else (0.0 if n_gt else np.nan))
    out.update({"tp": tp, "n_pred_scored": n_ps, "n_gt": n_gt, "tp_r": tp_r, "n_gt_r": n_gt_r, "n_ps_r": n_ps_r,
                "precision": precision, "recall": recall, "f1": f1, "scored": n_gt > 0})
    return out


def build_year_table(recs: List[Dict[str, Any]], gt_by_year: Optional[Dict[str, List[Dict[str, Any]]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in recs:
        for yr in rec.get("_year_records") or []:
            y = str(yr.get("Year_Requested"))
            text = str(yr.get("Generated_Response", ""))
            is_err = text.startswith("ERROR") or yr.get("Year_Label") == "ERROR"
            pred = [] if is_err else [r for r in (yr.get("Parsed_Rows") or []) if not is_sentinel_player(r.get("Player", ""))]
            label = "ERROR" if is_err else (yr.get("Year_Label") or "ROWS")
            g_list = gt_by_year.get(y) if gt_by_year else None
            tp = n_gt = 0
            gt_av = g_list is not None
            fn_names: List[str] = []
            fp_names: List[str] = []
            if gt_av:
                pairs, rem_gt, unm = match_pairs(g_list, pred)
                tp, n_gt = len(pairs), len(g_list)
                fn_names = [g_list[i]["player"] for i in rem_gt]
                fp_names = [pred[i]["Player"] for i in unm]
            n_pred = len(pred)
            precision = tp / n_pred if (gt_av and n_pred) else np.nan
            recall = tp / n_gt if (gt_av and n_gt) else np.nan
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision == precision and recall == recall and (precision + recall) > 0)
                  else (0.0 if (gt_av and n_gt and n_pred) else np.nan))
            stated_none = bool(yr.get("Year_Stated_None"))
            ca = yr.get("Citation_Audit") or {}
            rows.append({
                "model_alias": rec.get("model_alias"), "strategy": rec.get("strategy"), "rag_tier": rec.get("rag_tier"),
                "directive_level": rec.get("directive_level"), "year": int(y), "label": label,
                "answered": bool(yr.get("Year_Answered")), "gt_available": gt_av,
                "n_pred": n_pred, "n_gt": n_gt if gt_av else np.nan, "tp": tp,
                "precision": precision, "recall": recall, "f1": f1,
                "stated_none": stated_none, "false_none": bool(stated_none and gt_av and n_gt > 0),
                "true_none": bool(stated_none and gt_av and n_gt == 0),
                "off_year_rows": int(yr.get("Off_Year_Rows", 0) or 0),
                "self_excluded_rows": int(yr.get("Self_Excluded_Rows", 0) or 0),
                "contradiction": bool(yr.get("Self_Contradiction")), "closing_offer": bool(yr.get("Closing_Offer")),
                "status": yr.get("generation_status"), "completion_tokens": _completion_tokens(yr.get("usage")),
                "text_chars": yr.get("text_chars"), "thinking_chars": yr.get("thinking_chars"),
                "cost_usd": yr.get("cost_usd_reported"),
                "latency_s": yr.get("generation_time_seconds"), "citation_complete": bool(ca.get("complete", False)),
                "n_urls": int(ca.get("n_urls", 0) or 0),
                "missed_players": "; ".join(fn_names), "false_positive_players": "; ".join(fp_names),
            })
    return pd.DataFrame(rows)


def summarise_years(ydf: pd.DataFrame) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for y, g in ydf.groupby("year"):
        gs = g[g["gt_available"]]
        tp, npred, ngt = gs["tp"].sum(), gs["n_pred"].sum(), gs["n_gt"].sum()
        p = tp / npred if npred else np.nan
        r = tp / ngt if ngt else np.nan
        f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan
        n = len(g)
        out.append({"year": y, "calls": n, "gt_available": bool(g["gt_available"].any()),
                    "gt_players": int(gs["n_gt"].max()) if len(gs) else np.nan,
                    "answered_share": g["answered"].mean(),
                    "rows_share": (g["label"] == "ROWS").mean(),
                    "none_claim_share": (g["label"] == "NO QUALIFYING PLAYER").mean(),
                    "declined_share": (g["label"] == "DECLINED").mean(),
                    "stalled_share": (g["label"] == "STALLED").mean(),
                    "unparsed_share": g["label"].isin(["UNPARSED", "OFF_YEAR", "EMPTY"]).mean(),
                    "error_share": (g["label"] == "ERROR").mean(),
                    "mean_rows": g["n_pred"].mean(), "micro_precision": p, "micro_recall": r, "micro_f1": f,
                    "false_none_calls": int(g["false_none"].sum()), "true_none_calls": int(g["true_none"].sum()),
                    "mean_off_year_rows": g["off_year_rows"].mean(),
                    "mean_completion_tokens": pd.to_numeric(g["completion_tokens"], errors="coerce").mean(),
                    "mean_text_chars": pd.to_numeric(g["text_chars"], errors="coerce").mean(),
                    "mean_thinking_chars": pd.to_numeric(g["thinking_chars"], errors="coerce").mean(),
                    "mean_latency_s": pd.to_numeric(g["latency_s"], errors="coerce").mean()})
    return pd.DataFrame(out)


def add_consensus(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Counter]:
    df = df.copy()
    df["jaccard_to_consensus"] = np.nan
    usable = df[(~df["is_error"]) & (df["n_pred"] > 0)]
    freq: Counter = Counter()
    cons_rows: List[Dict[str, Any]] = []
    if len(usable) < 3:
        return df, pd.DataFrame(cons_rows), freq
    reps: Dict[str, List[str]] = defaultdict(list)
    disp: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    def canon(y: str, name: str) -> Tuple[str, str]:
        k = name_key(name)
        for r in reps[y]:
            if names_match(k, r):
                return (y, r)
        reps[y].append(k)
        return (y, k)

    sets: Dict[Any, Set[Tuple[str, str]]] = {}
    for idx, pairs in usable["pairs"].items():
        ks = set()
        for y, n in pairs:
            c = canon(y, n)
            ks.add(c); disp[c][n] += 1
        sets[idx] = ks
    cnt = Counter(k for ks in sets.values() for k in ks)
    cons = {k for k, v in cnt.items() if v / len(sets) >= 0.5}
    for idx, ks in sets.items():
        u = ks | cons
        df.loc[idx, "jaccard_to_consensus"] = (len(ks & cons) / len(u)) if u else 1.0
    for k, v in cnt.items():
        freq[(k[0], disp[k].most_common(1)[0][0])] = v
    for y in sorted({k[0] for k in cnt}, key=int):
        ck = [k for k in cnt if k[0] == y]
        cs = [k for k in ck if k in cons]
        cons_rows.append({"WorldCup_Year": y, "records_returning_edition": sum(1 for ks in sets.values() if any(k[0] == y for k in ks)),
                          "usable_records": len(sets), "consensus_size": len(cs),
                          "consensus_players": "; ".join(sorted(disp[k].most_common(1)[0][0] for k in cs)),
                          "players_seen_in_any_record": len(ck), "players_not_in_consensus": len(ck) - len(cs)})
    return df, pd.DataFrame(cons_rows), freq


def _agg(g: pd.DataFrame) -> Dict[str, Any]:
    def rate(a: str, b: str) -> float:
        d = g[b].sum()
        return g[a].sum() / d if d else np.nan

    sc = g[g["scored"]]
    n_ps, n_gt, tp = sc["n_pred_scored"].sum(), sc["n_gt"].sum(), sc["tp"].sum()
    p = tp / n_ps if n_ps else np.nan
    r = tp / n_gt if n_gt else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan
    n_ps_r, n_gt_r, tp_r = sc["n_ps_r"].sum(), sc["n_gt_r"].sum(), sc["tp_r"].sum()
    p_r = tp_r / n_ps_r if n_ps_r else np.nan
    r_r = tp_r / n_gt_r if n_gt_r else np.nan
    return {"records": len(g), "mean_rows": g["n_pred"].mean(), "mean_editions_covered": g["n_editions"].mean(),
            "mean_editions_declared": g["n_declared"].mean(),
            "median_last_edition": pd.to_numeric(g["last_year"], errors="coerce").median(),
            "false_none_per_record": g["false_none_n"].mean(), "true_none_per_record": g["true_none_n"].mean(),
            "mean_jaccard_to_consensus": g["jaccard_to_consensus"].mean(),
            "pop_numeric_rate": rate("pop_numeric_n", "pop_total_n"),
            "birthplace_blank_rate": rate("birth_blank_n", "pop_total_n"),
            "citation_complete": g["citation_complete"].mean(), "citation_order_ok": g["citation_order_ok"].mean(),
            "mean_urls": g["n_urls"].mean(), "truncated_rate": g["truncated"].mean(),
            "stall_rate": g["stalled"].mean(), "degenerate_rate": g["degenerate"].mean(),
            "declined_rate": g["declined"].mean(), "contradiction_rate": g["contradiction"].mean(),
            "escalated_rate": g["escalated"].mean(),
            "self_excluded_per_record": g["self_excluded_n"].mean(), "off_year_rows_per_record": g["off_year_n"].mean(),
            "mean_years_answered": g["years_answered"].mean(), "mean_years_declined": g["years_declined"].mean(),
            "mean_years_error": g["years_error"].mean(),
            "unparsed_rate": g["unparsed"].mean(), "error_rate": g["error_share"].mean(),
            "mean_latency_s": pd.to_numeric(g["latency"], errors="coerce").mean(),
            "mean_completion_tokens": g["completion_tokens"].mean(),
            "gt_scored_records": int(g["scored"].sum()), "gt_micro_precision": p, "gt_micro_recall": r, "gt_micro_f1": f,
            "gt_precision_reached": p_r, "gt_recall_reached": r_r,
            "gt_award_acc": rate("award_ok", "award_n"), "gt_birthplace_acc": rate("birth_ok", "birth_n"),
            "gt_pop_within_5pct": rate("pop_5_n", "pop_cmp_n")}


def summarise(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(by, key))
        d.update(_agg(g))
        rows.append(d)
    return pd.DataFrame(rows)


def edition_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[~df["is_error"]]
    years = sorted({y for yc in ok["year_counts"] for y in yc}, key=int)
    rows = []
    for y in years:
        cnts = [yc.get(y, 0) for yc in ok["year_counts"]]
        have = [c for c in cnts if c > 0]
        rows.append({"WorldCup_Year": y, "records_returning_edition": len(have), "share_of_records": len(have) / max(1, len(cnts)),
                     "mean_players_when_returned": float(np.mean(have)) if have else np.nan,
                     "max_players": max(cnts) if cnts else 0})
    return pd.DataFrame(rows)


def paired_delta(df: pd.DataFrame, factor: str, baseline: str, keys: List[str], metric: str) -> pd.DataFrame:
    d0 = df[df[metric].notna()].drop_duplicates(subset=keys + [factor])
    base = d0[d0[factor] == baseline].set_index(keys)[metric].rename("base")
    rows = []
    for lvl, g in d0[d0[factor] != baseline].groupby(factor):
        j = g.set_index(keys)[metric].to_frame("m").join(base, how="inner")
        if j.empty:
            continue
        d = j["m"] - j["base"]
        rows.append({factor: lvl, "paired_n": len(j), f"mean_delta_{metric}": d.mean(),
                     "wins": int((d > 1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()), "losses": int((d < -1e-9).sum())})
    return pd.DataFrame(rows)


def check_urls(urls: List[str], outdir: Path, timeout: float = 8.0, workers: int = 16) -> Dict[str, bool]:
    hdr = {"User-Agent": "Mozilla/5.0 (citation-reachability-check)"}

    def probe(u: str) -> Tuple[str, str]:
        url = u if u.lower().startswith("http") else "https://" + u
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout, headers=hdr)
            if r.status_code >= 400:
                r = requests.get(url, allow_redirects=True, timeout=timeout, stream=True, headers=hdr)
            return u, str(r.status_code)
        except Exception as e:
            return u, f"ERR:{type(e).__name__}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(probe, urls))
    pd.DataFrame({"url": list(res), "status": list(res.values())}).to_csv(outdir / "url_check.csv", index=False)
    return {u: (s.isdigit() and int(s) < 400) for u, s in res.items()}


def _tbl(df: pd.DataFrame) -> str:
    return "(no data)" if df is None or df.empty else df.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def _make_plots(df: pd.DataFrame, ed: pd.DataFrame, freq: Counter, outdir: Path, has_gt: bool) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}) - plots skipped")
        return []
    saved: List[str] = []

    def save(fig, name: str) -> None:
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    def label(v: Any) -> str:
        return RAG_CODE.get(v, STRATEGY_CODE.get(v, str(v)))

    def barh(fac: str, metric: str, title: str, name: str, xlim: Optional[Tuple[float, float]] = (0, 1)) -> None:
        s = summarise(df, [fac]).dropna(subset=[metric])
        if s.empty:
            return
        s = (s.assign(_o=s[fac].map({t: i for i, t in enumerate(RAG_TIERS)})).sort_values("_o") if fac == "rag_tier"
             else s.sort_values(metric))
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.32 * len(s) + 1.4)))
        ax.barh([label(v) for v in s[fac]], s[metric], color="#2a6fbb")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(metric); ax.set_title(title)
        save(fig, name)

    def heat(pivot: pd.DataFrame, title: str, name: str, vmax: Optional[float] = None) -> None:
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(0.6 * pivot.shape[1] + 4, 0.34 * pivot.shape[0] + 2))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax); ax.set_title(title)
        save(fig, name)

    if not ed.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(ed["WorldCup_Year"], ed["share_of_records"], color="#2a6fbb")
        ax.set_ylim(0, 1.02); ax.set_ylabel("share of records that returned rows for the edition")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_title("Edition coverage across records (drop-off = truncation, stall, omission or no qualifier)")
        save(fig, "edition_coverage.png")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(ed["WorldCup_Year"], ed["mean_players_when_returned"], color="#c0392b")
        ax.set_ylabel("mean qualifying players per returning record")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right"); ax.set_title("Players returned per edition")
        save(fig, "players_per_edition.png")
    barh("model_alias", "mean_editions_covered", "Mean editions with rows per response, by model", "editions_covered_by_model.png", None)
    barh("model_alias", "mean_editions_declared", "Mean editions declared / answered per response, by model", "editions_declared_by_model.png", None)
    barh("strategy", "mean_editions_covered", "Mean editions with rows per response, by strategy", "editions_covered_by_strategy.png", None)
    barh("model_alias", "stall_rate", "Conversational-stall rate by model", "stall_rate_by_model.png")
    barh("model_alias", "declined_rate", "Declined-to-answer rate by model", "declined_rate_by_model.png")
    barh("model_alias", "contradiction_rate", "Self-contradiction rate by model (rows vs the model's own note)", "contradiction_rate_by_model.png")
    barh("model_alias", "mean_jaccard_to_consensus", "Agreement with consensus by model", "agreement_by_model.png")
    barh("strategy", "mean_jaccard_to_consensus", "Agreement with consensus by strategy", "agreement_by_strategy.png")
    barh("rag_tier", "mean_jaccard_to_consensus", "Agreement with consensus by RAG tier", "agreement_by_rag.png")
    metric_h = "jaccard_to_consensus" if df["jaccard_to_consensus"].notna().any() else "n_pred"
    pv = df.pivot_table(values=metric_h, index="strategy", columns="rag_tier", aggfunc="mean")
    pv = pv.reindex(index=[s for s in INDIVIDUAL_STRATEGIES if s in pv.index], columns=[t for t in RAG_TIERS if t in pv.columns])
    pv.index = [STRATEGY_CODE[s] for s in pv.index]; pv.columns = [RAG_CODE[t] for t in pv.columns]
    heat(pv, f"Mean {metric_h}: strategy x RAG tier", "heatmap_strategy_x_rag.png", 1.0 if metric_h != "n_pred" else None)
    ex = [(r["model_alias"], y, c) for _, r in df[~df["is_error"]].iterrows() for y, c in r["year_counts"].items()]
    if ex:
        pm = pd.DataFrame(ex, columns=["m", "y", "c"]).pivot_table(values="c", index="m", columns="y", aggfunc="mean")
        pm = pm[sorted(pm.columns, key=int)]
        heat(pm, "Mean qualifying players returned: model x edition", "heatmap_model_x_edition.png")
    for fac, fname in (("model_alias", "citation_quality_by_model.png"), ("strategy", "citation_quality_by_strategy.png")):
        cs = summarise(df, [fac]).sort_values("citation_complete")
        if cs.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(cs) + 1.4)))
        y = np.arange(len(cs))
        ax.barh(y - 0.2, cs["citation_complete"], 0.4, label="all required citation categories present (heuristic)")
        ax.barh(y + 0.2, cs["citation_order_ok"], 0.4, label="in required order (heuristic)")
        ax.set_yticks(y); ax.set_yticklabels([label(v) for v in cs[fac]]); ax.set_xlim(0, 1)
        ax.set_title("Citation completeness"); ax.legend(fontsize=7, loc="lower right")
        save(fig, fname)
    h = summarise(df, ["model_alias"]).sort_values("truncated_rate")
    if not h.empty:
        cats = [("truncated_rate", "truncated"), ("stall_rate", "conversational stall"), ("declined_rate", "declined"),
                ("degenerate_rate", "repetition loop at end (flag)"), ("unparsed_rate", "unparsed (no table row found)"),
                ("error_rate", "error")]
        fig, ax = plt.subplots(figsize=(8, max(3.4, 0.5 * len(h) + 1.4)))
        y = np.arange(len(h))
        w = 0.8 / len(cats)
        for i, (col, lab) in enumerate(cats):
            ax.barh(y - 0.4 + w * (i + 0.5), h[col], w, label=lab)
        ax.set_yticks(y); ax.set_yticklabels(h["model_alias"]); ax.set_xlim(0, 1)
        ax.set_title("Generation health by model"); ax.legend(fontsize=7)
        save(fig, "generation_health_by_model.png")
    ct = df[df["completion_tokens"].notna()]
    if not ct.empty:
        s = ct.groupby("strategy")["completion_tokens"].mean().reindex([x for x in INDIVIDUAL_STRATEGIES if x in set(ct["strategy"])])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([STRATEGY_CODE[x] for x in s.index], s.values, color="#2a6fbb")
        ax.set_ylabel("mean completion tokens per cell (reported usage)"); ax.set_title("Output length by strategy")
        save(fig, "completion_tokens_by_strategy.png")
        s2 = ct.groupby("model_alias")["completion_tokens"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.32 * len(s2) + 1.4)))
        ax.barh(s2.index, s2.values, color="#2a6fbb")
        ax.set_xlabel("mean completion tokens per cell (reported usage)"); ax.set_title("Output length by model")
        save(fig, "completion_tokens_by_model.png")
    barh("model_alias", "pop_numeric_rate", "Share of returned rows with a numeric population", "population_numeric_rate_by_model.png")
    pops = [v for lst in df["pop_values"] for v in lst if v > 0]
    if pops:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.log10(pops), bins=40, color="#2a6fbb")
        ax.set_xlabel("log10(population stated by the model)"); ax.set_ylabel("rows")
        ax.set_title("Distribution of stated birthplace populations")
        save(fig, "population_distribution.png")
    if freq:
        top = freq.most_common(20)
        fig, ax = plt.subplots(figsize=(8, 0.34 * len(top) + 1.5))
        ax.barh([f"{n} ({y})" for (y, n), _ in top][::-1], [c for _, c in top][::-1], color="#2a6fbb")
        ax.set_xlabel("records returning the player"); ax.set_title("Players most often returned (edition in brackets)")
        save(fig, "top_returned_players.png")
    am = df.groupby("model_alias")[["award_ballon_n", "award_uefa_n", "award_unrecognised_n"]].sum()
    if am.values.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(am) + 1.4)))
        left = np.zeros(len(am))
        for col, lab in (("award_ballon_n", "Ballon d'Or"), ("award_uefa_n", "UEFA Men's POTY family"),
                         ("award_unrecognised_n", "unrecognised award text")):
            ax.barh(am.index, am[col], left=left, label=lab); left += am[col].values
        ax.set_xlabel("returned rows"); ax.set_title("Award types returned"); ax.legend(fontsize=7)
        save(fig, "award_composition_by_model.png")
    la = df.groupby("model_alias").agg(j=("jaccard_to_consensus", "mean"),
                                       lat=("latency", lambda s: pd.to_numeric(s, errors="coerce").mean())).dropna()
    if not la.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(la["lat"], la["j"], s=40)
        for n, r in la.iterrows():
            ax.annotate(n, (r["lat"], r["j"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("latency per cell (s)"); ax.set_ylabel("mean agreement with consensus")
        ax.set_title("Latency vs agreement")
        save(fig, "latency_vs_agreement.png")
    if has_gt and df["scored"].any():
        sc = df[df["scored"]]
        for fac, fname in (("model_alias", "gt_f1_by_model.png"), ("strategy", "gt_f1_by_strategy.png"), ("rag_tier", "gt_f1_by_rag.png")):
            s = summarise(sc, [fac])
            s = (s.assign(_o=s[fac].map({t: i for i, t in enumerate(RAG_TIERS)})).sort_values("_o") if fac == "rag_tier"
                 else s.sort_values("gt_micro_f1"))
            y, hh = np.arange(len(s)), 0.27
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(s) + 1.5)))
            ax.barh(y - hh, s["gt_micro_precision"], hh, label="precision")
            ax.barh(y, s["gt_micro_recall"], hh, label="recall")
            ax.barh(y + hh, s["gt_micro_f1"], hh, label="F1 (micro)")
            ax.set_yticks(y); ax.set_yticklabels([label(v) for v in s[fac]]); ax.set_xlim(0, 1)
            ax.set_title(f"Ground-truth scoring by {fac}"); ax.legend(loc="lower right")
            save(fig, fname)
        pe = summarise(sc, ["model_alias"]).sort_values("gt_micro_recall")
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.36 * len(pe) + 1.4)))
        y = np.arange(len(pe))
        ax.barh(y - 0.2, pe["gt_micro_recall"], 0.4, label="recall (all editions)", color="#c0392b")
        ax.barh(y + 0.2, pe["gt_recall_reached"], 0.4, label="recall (editions reached only)", color="#2a6fbb")
        ax.set_yticks(y); ax.set_yticklabels(pe["model_alias"]); ax.set_xlim(0, 1)
        ax.set_title("Ground-truth recall by model: all vs reached editions"); ax.legend(fontsize=7)
        save(fig, "gt_recall_by_model.png")
        fn = pe.set_index("model_alias")[["false_none_per_record", "true_none_per_record"]]
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(fn) + 1.4)))
        yy = np.arange(len(fn))
        ax.barh(yy - 0.2, fn["false_none_per_record"], 0.4, label="false 'no qualifier' claims per record", color="#c0392b")
        ax.barh(yy + 0.2, fn["true_none_per_record"], 0.4, label="correct 'no qualifier' claims per record", color="#2a6fbb")
        ax.set_yticks(yy); ax.set_yticklabels(fn.index); ax.legend(fontsize=7)
        ax.set_title("Explicit 'no qualifying player' claims vs ground truth")
        save(fig, "none_claims_by_model.png")
    return saved


def _make_year_plots(ydf: pd.DataFrame, ysum: pd.DataFrame, outdir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}) - per-year plots skipped")
        return []
    saved: List[str] = []

    def save(fig, name: str) -> None:
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    def heat(value: str, title: str, name: str, vmax: Optional[float] = None) -> None:
        pv = ydf.pivot_table(values=value, index="model_alias", columns="year", aggfunc="mean")
        if pv.empty or pv.isna().all().all():
            return
        pv = pv[sorted(pv.columns)]
        fig, ax = plt.subplots(figsize=(0.55 * pv.shape[1] + 4, 0.34 * pv.shape[0] + 2))
        im = ax.imshow(np.ma.masked_invalid(pv.values.astype(float)), aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([str(c) for c in pv.columns], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=7)
        fig.colorbar(im, ax=ax); ax.set_title(title + "  (blank = no ground truth / no data)")
        save(fig, name)

    heat("recall", "Recall by model x edition (per-year calls)", "year_recall_heatmap.png", 1.0)
    heat("precision", "Precision by model x edition (per-year calls)", "year_precision_heatmap.png", 1.0)
    heat("n_pred", "Rows returned by model x edition", "year_rows_heatmap.png")
    heat("completion_tokens", "Completion tokens by model x edition", "year_tokens_heatmap.png")
    heat("text_chars", "Visible answer length (chars) by model x edition", "year_text_chars_heatmap.png")
    heat("thinking_chars", "Thinking length (chars) by model x edition", "year_thinking_chars_heatmap.png")
    if not ysum.empty:
        ys = ysum.sort_values("year")
        x = np.arange(len(ys))
        fig, ax = plt.subplots(figsize=(11, 4.6))
        w = 0.27
        ax.bar(x - w, ys["micro_precision"].fillna(0), w, label="precision (micro)")
        ax.bar(x, ys["micro_recall"].fillna(0), w, label="recall (micro)")
        ax.bar(x + w, ys["micro_f1"].fillna(0), w, label="F1 (micro)")
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in ys["year"]], rotation=60, ha="right")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        ax.set_title("Ground-truth scoring per edition, pooled over all models / cells (years without GT show 0)")
        save(fig, "year_pr_f1.png")
        cats = [("rows_share", "rows returned"), ("none_claim_share", "explicit 'no qualifier'"),
                ("declined_share", "declined"), ("stalled_share", "stalled"), ("unparsed_share", "unparsed / off-year / empty"),
                ("error_share", "call error")]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        bottom = np.zeros(len(ys))
        for col, lab in cats:
            ax.bar(x, ys[col].fillna(0).values, bottom=bottom, label=lab)
            bottom += ys[col].fillna(0).values
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in ys["year"]], rotation=60, ha="right")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=7, ncol=3)
        ax.set_title("What each per-year call produced (share of calls)")
        save(fig, "year_outcome_shares.png")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.bar(x - 0.2, ys["mean_rows"], 0.4, label="mean rows returned per call", color="#2a6fbb")
        gtp = ys["gt_players"].fillna(0)
        ax.bar(x + 0.2, gtp, 0.4, label="ground-truth qualifying players", color="#c0392b")
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in ys["year"]], rotation=60, ha="right")
        ax.legend(fontsize=8); ax.set_title("Rows returned vs ground-truth size per edition")
        save(fig, "year_rows_vs_gt.png")
    return saved


def run_analysis(root: Path = OUTPUT_ROOT, do_url_check: bool = False, raw: bool = False) -> None:
    all_recs = collect_records(root)
    recs = [r for r in all_recs if bool(r.get("system_directive_used", False)) != raw]
    if not recs:
        other = len(all_recs)
        logger.warning(f"Analysis: no {'raw-prompt' if raw else 'directive'} result files under "
                       f"{root.resolve()} ({other} record(s) of the other kind found"
                       f"{'; use --raw_prompt to analyse them' if (other and not raw) else ''}).")
        return
    gt_by_year: Optional[Dict[str, List[Dict[str, Any]]]] = None
    try:
        gt_by_year = _gt_by_year(GroundTruthStore(GROUND_TRUTH_FILE))
    except (FileNotFoundError, ValueError):
        logger.info("Analysis: no ground-truth file found - structural and consensus analysis only (no accuracy scoring).")
    df = pd.DataFrame([evaluate_record(r, gt_by_year) for r in recs])
    df = df.drop_duplicates(subset=["model_alias", "strategy", "rag_tier", "directive_level", "run_mode", "Query_ID"],
                            keep="last").reset_index(drop=True)
    df, cons, freq = add_consensus(df)
    ed = edition_table(df)
    outdir = root / (ANALYSIS_DIR_NAME + ("_raw" if raw else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    per_year_mode = any(r.get("_year_records") for r in recs)

    url_note = "not run (use --check_urls)"
    df["url_ok_rate"] = np.nan
    if do_url_check:
        all_urls = sorted({u for lst in df["urls"] for u in lst})
        if all_urls:
            ok = check_urls(all_urls, outdir)
            df["url_ok_rate"] = [np.mean([ok[u] for u in lst]) if lst else np.nan for lst in df["urls"]]
            url_note = f"{len(all_urls)} unique URLs probed; {sum(ok.values())} reachable (HTTP < 400) -> {outdir / 'url_check.csv'}"
        else:
            url_note = "no URLs were cited"

    flat = df.drop(columns=["fn_players", "fp_players", "pop_log_ratios", "pairs", "year_counts", "pop_values", "urls"])
    flat.to_csv(outdir / "eval_per_record.csv", index=False, encoding="utf-8-sig")
    by_model = summarise(df, ["model_alias"]).sort_values("mean_jaccard_to_consensus", ascending=False)
    by_strat = summarise(df, ["strategy"]).assign(code=lambda d: d["strategy"].map(STRATEGY_CODE))
    by_rag = summarise(df, ["rag_tier"]).assign(code=lambda d: d["rag_tier"].map(RAG_CODE)).sort_values("code")
    by_cfg = summarise(df, ["model_alias", "strategy", "rag_tier", "directive_level"])
    for name, t in [("summary_by_model", by_model), ("summary_by_strategy", by_strat), ("summary_by_rag_tier", by_rag),
                    ("summary_by_model_strategy_rag_level", by_cfg), ("summary_by_edition", ed), ("consensus_by_edition", cons)]:
        t.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    gt_loaded = bool(gt_by_year)
    has_gt = gt_loaded and bool(df["scored"].any())
    plots = _make_plots(df, ed, freq, outdir, has_gt)

    ydf, ysum = pd.DataFrame(), pd.DataFrame()
    if per_year_mode:
        ydf = build_year_table(recs, gt_by_year)
        if not ydf.empty:
            ysum = summarise_years(ydf)
            ydf.to_csv(outdir / "per_year_eval.csv", index=False, encoding="utf-8-sig")
            ysum.to_csv(outdir / "per_year_summary.csv", index=False, encoding="utf-8-sig")
            plots += _make_year_plots(ydf, ysum, outdir)

    levels = sorted(df["directive_level"].dropna().unique().tolist())
    L: List[str] = []
    add = L.append
    add("=" * 100); add("FOOTBALL PLAYER ANALYSER (CLAUDE CODE) - ANALYSIS REPORT")
    add(f"generated {datetime.now().isoformat(timespec='seconds')}  (script version {SCRIPT_VERSION})"); add("=" * 100)
    add("\n1. SCOPE")
    add(f"  run mode: {'PER-YEAR (placeholder replaced by the year; one call per year, aggregated per model x strategy x RAG tier)' if per_year_mode else 'single-shot (placeholder kept; the model resolves all editions)'}")
    add(f"  prompt mode: {'RAW (user turn only; Claude Code default system prompt, no directive)' if raw else 'user turn + --system-prompt directive'}")
    add(f"  directive levels present: {levels}   escalated records: {int(df['escalated'].sum())}  "
        f"(levels are pooled in the by-model tables below; filter eval_per_record.csv on directive_level to compare them)")
    add(f"  records={len(df)} (one per model x strategy x RAG tier x level)  models={df['model_alias'].nunique()}  strategies={df['strategy'].nunique()}  "
        f"rag_tiers={df['rag_tier'].nunique()}  editions seen in answers={len(ed)}  verbatim-prompt records={int(df['prompt_is_verbatim'].sum())}")
    add("  Backend: `claude -p` inside a Claude Code cloud session; temperature / top_p are not controllable; tools disabled; "
        "raw events per call are in <cell dir>/raw_calls.")
    add("  Editions are read from the WorldCup_Year column of the returned rows and (single-shot) from the edition headers the model wrote. "
        "'Agreement with consensus' = Jaccard overlap between a record's (edition, player) set and the set returned by >=50% of the usable "
        "records (needs >=3). It measures agreement, NOT correctness.")
    if per_year_mode:
        add("  Per-year: tokens, latency and URL counts are TOTALS per cell (sum over its years); stall / decline / error / truncation "
            "figures are shares of the cell's year calls. Rows labelled with a different year than requested are dropped and counted (off_year_rows).")
    if has_gt:
        gt_line = (f"ON ({int(df['scored'].sum())} scored records; only editions present in the GT file are scored; editions with no "
                   f"qualifier count every returned row as a false positive; '..._reached' metrics only count editions up to the last one the model "
                   f"declared or returned; false/true 'none' = explicit 'no qualifying player' claims checked against the GT)")
    elif gt_loaded:
        gt_line = "OFF (ground-truth file loaded, but no usable record to score)"
    else:
        gt_line = "OFF (no ground-truth file found)"
    add(f"  Ground-truth scoring: {gt_line}")
    add("\n2. HEADLINE (all records pooled)")
    h = _agg(df)
    head_keys = ["mean_rows", "mean_editions_covered", "mean_editions_declared", "median_last_edition", "false_none_per_record",
                 "true_none_per_record", "mean_jaccard_to_consensus", "pop_numeric_rate", "birthplace_blank_rate", "citation_complete",
                 "citation_order_ok", "mean_urls", "truncated_rate", "stall_rate", "declined_rate", "contradiction_rate",
                 "escalated_rate", "self_excluded_per_record", "degenerate_rate", "unparsed_rate", "error_rate",
                 "mean_latency_s", "mean_completion_tokens"]
    if per_year_mode:
        head_keys += ["mean_years_answered", "mean_years_declined", "mean_years_error", "off_year_rows_per_record"]
    for k in head_keys:
        add(f"  {k:<28}{h[k]:.4f}" if h[k] == h[k] else f"  {k:<28}n/a")
    cols = ["records", "mean_rows", "mean_editions_covered", "mean_editions_declared", "median_last_edition",
            "false_none_per_record", "true_none_per_record", "mean_jaccard_to_consensus", "pop_numeric_rate",
            "citation_complete", "citation_order_ok", "mean_urls", "truncated_rate", "stall_rate", "declined_rate",
            "contradiction_rate", "escalated_rate", "self_excluded_per_record", "degenerate_rate", "unparsed_rate",
            "error_rate", "mean_latency_s", "mean_completion_tokens"]
    if per_year_mode:
        cols += ["mean_years_answered", "mean_years_declined", "mean_years_error", "off_year_rows_per_record"]
    add("\n3. BY MODEL (sorted by agreement with consensus)"); add(_tbl(by_model[["model_alias"] + cols]))
    add("\n4. BY STRATEGY"); add(_tbl(by_strat[["code"] + cols]))
    add("\n5. BY RAG TIER"); add(_tbl(by_rag[["code"] + cols]))
    metric = "f1" if has_gt else "jaccard_to_consensus"
    pr = paired_delta(df, "rag_tier", RAG_TIER_NO_EVIDENCE, ["model_alias", "strategy", "directive_level"], metric)
    ps = paired_delta(df, "strategy", STRATEGY_FLAT_CONJUNCTIVE, ["model_alias", "rag_tier", "directive_level"], metric)
    if not pr.empty:
        pr["rag_tier"] = pr["rag_tier"].map(RAG_CODE)
    if not ps.empty:
        ps["strategy"] = ps["strategy"].map(STRATEGY_CODE)
    add(f"\n6. PAIRED RAG EFFECT vs RAG0 on {metric} (same model, strategy, directive level)"); add(_tbl(pr))
    add(f"\n7. PAIRED STRATEGY EFFECT vs S1 on {metric} (same model, RAG tier, directive level)"); add(_tbl(ps))
    add("\n8. EDITION COVERAGE (from parsed rows)"); add(_tbl(ed))
    add("\n9. CONSENSUS SETS PER EDITION (players returned by >=50% of usable records)"); add(_tbl(cons))
    add("\n10. PLAYERS MOST OFTEN RETURNED (edition, player, number of records)")
    add("  " + ("; ".join(f"{y} {n}: {c}" for (y, n), c in freq.most_common(25)) or "n/a"))
    add("\n11. AWARD TYPES RETURNED")
    add(f"  Ballon d'Or rows={int(df['award_ballon_n'].sum())}  UEFA-family rows={int(df['award_uefa_n'].sum())}  "
        f"rows whose award text matched neither={int(df['award_unrecognised_n'].sum())}")
    add("\n12. POPULATION AND BIRTHPLACE FIELDS")
    add(f"  numeric population rate={h['pop_numeric_rate']:.3f}  blank/unknown birthplace rate={h['birthplace_blank_rate']:.3f}")
    add("\n13. CITATION QUALITY (keyword/order heuristic over the whole response; sources are NOT verified)")
    add(f"  all required categories present={df['citation_complete'].mean():.3f}  in required order={df['citation_order_ok'].mean():.3f}  "
        f"mean URLs per cell={df['n_urls'].mean():.2f}")
    add(f"  URL reachability: {url_note}")
    if df["url_ok_rate"].notna().any():
        add(f"  mean share of cited URLs reachable per cell={df['url_ok_rate'].mean():.3f} (reachable != supports the claim)")
    add("\n14. GENERATION HEALTH"); add(_tbl(df.groupby(["model_alias", "directive_level", "status"]).size().reset_index(name="count")))
    add("\n15. RAG EVIDENCE INTEGRITY")
    inj = df[df["rag_evidence_injected"]]
    add(f"  records with evidence injected={len(inj)}  incomplete evidence={int((~inj['rag_evidence_complete']).sum())}  "
        f"population-check failures={int(df['rag_pop_check_failures'].sum())}")
    if has_gt:
        add("\n16. GROUND-TRUTH SCORING (scored records; editions present in the GT file only)")
        add(_tbl(by_model[["model_alias", "gt_scored_records", "median_last_edition", "gt_micro_precision", "gt_micro_recall",
                           "gt_micro_f1", "gt_precision_reached", "gt_recall_reached", "false_none_per_record",
                           "true_none_per_record", "gt_award_acc", "gt_birthplace_acc", "gt_pop_within_5pct"]]))
        sc = df[df["scored"]]
        miss = Counter(p for lst in sc["fn_players"] for p in lst).most_common(15)
        fps = Counter(p for lst in sc["fp_players"] for p in lst).most_common(15)
        add("  Most missed qualifying players (edition player: records): " + ("; ".join(f"{a}: {b}" for a, b in miss) or "none"))
        add("  Most frequent false positives:                              " + ("; ".join(f"{a}: {b}" for a, b in fps) or "none"))
    if per_year_mode and not ysum.empty:
        add("\n16A. PER-YEAR ANALYSIS (every call = one edition; pooled over models / strategies / RAG tiers / levels)")
        add(_tbl(ysum))
        for val, ttl in (("recall", "RECALL"), ("precision", "PRECISION"), ("n_pred", "ROWS RETURNED"),
                         ("text_chars", "VISIBLE ANSWER LENGTH (chars)"), ("thinking_chars", "THINKING LENGTH (chars)")):
            pv = ydf.pivot_table(values=val, index="model_alias", columns="year", aggfunc="mean")
            if not pv.empty:
                add(f"\n  {ttl} by model x edition (mean over the model's cells; blank = not defined / no GT)")
                add(_tbl(pv.reset_index()))
        fnc = ydf[ydf["false_none"]].groupby("year").size()
        if len(fnc):
            add("\n  False 'no qualifying player' claims per edition (calls): " + ", ".join(f"{y}: {c}" for y, c in fnc.items()))
        mp = Counter(p for s in ydf["missed_players"] for p in str(s).split("; ") if p)
        fp = Counter(p for s in ydf["false_positive_players"] for p in str(s).split("; ") if p)
        add("  Most missed players across all per-year calls: " + ("; ".join(f"{a}: {b}" for a, b in mp.most_common(15)) or "none"))
        add("  Most frequent false-positive names:             " + ("; ".join(f"{a}: {b}" for a, b in fp.most_common(15)) or "none"))
        cost = pd.to_numeric(ydf["cost_usd"], errors="coerce").sum() if "cost_usd" in ydf else float("nan")
        if cost == cost and cost > 0:
            add(f"  Reported cost across per-year calls (Claude Code 'total_cost_usd', summed): {cost:.2f} USD")

    ins: List[str] = []
    bm = by_model.dropna(subset=["mean_jaccard_to_consensus"])
    if len(bm) > 1:
        ins.append(f"Highest agreement with the consensus: {bm.iloc[0]['model_alias']} ({bm.iloc[0]['mean_jaccard_to_consensus']:.3f}); "
                   f"lowest: {bm.iloc[-1]['model_alias']} ({bm.iloc[-1]['mean_jaccard_to_consensus']:.3f}).")
    cv = by_model.dropna(subset=["mean_editions_declared"]).sort_values("mean_editions_declared")
    if len(cv) > 1:
        ins.append(f"Editions declared / answered per cell: most {cv.iloc[-1]['model_alias']} ({cv.iloc[-1]['mean_editions_declared']:.1f}), "
                   f"fewest {cv.iloc[0]['model_alias']} ({cv.iloc[0]['mean_editions_declared']:.1f}).")
    if has_gt and h["false_none_per_record"] == h["false_none_per_record"] and h["false_none_per_record"] > 0:
        fnm = by_model[by_model["false_none_per_record"] > 0].sort_values("false_none_per_record", ascending=False)
        ins.append(f"On average {h['false_none_per_record']:.2f} editions per record are wrongly declared 'no qualifying player' "
                   f"(ground truth has qualifiers): " + ", ".join(f"{r['model_alias']} {r['false_none_per_record']:.1f}" for _, r in fnm.head(8).iterrows()) + ".")
    if h["self_excluded_per_record"] == h["self_excluded_per_record"] and h["self_excluded_per_record"] > 0:
        sem = by_model[by_model["self_excluded_per_record"] > 0].sort_values("self_excluded_per_record", ascending=False)
        ins.append(f"Rows the model itself marked 'does NOT qualify' were ignored (not scored as predictions); mean per record "
                   f"{h['self_excluded_per_record']:.1f}: " + ", ".join(f"{r['model_alias']} {r['self_excluded_per_record']:.1f}" for _, r in sem.head(8).iterrows())
                   + ". Padding the table with non-qualifying rows is itself a quality issue.")
    if per_year_mode and h["off_year_rows_per_record"] == h["off_year_rows_per_record"] and h["off_year_rows_per_record"] > 0:
        oy = by_model[by_model["off_year_rows_per_record"] > 0].sort_values("off_year_rows_per_record", ascending=False)
        ins.append(f"Rows labelled with a different year than the one requested were dropped; mean per cell {h['off_year_rows_per_record']:.1f}: "
                   + ", ".join(f"{r['model_alias']} {r['off_year_rows_per_record']:.1f}" for _, r in oy.head(8).iterrows()) + ".")
    if h["declined_rate"] == h["declined_rate"] and h["declined_rate"] > 0:
        dcm = by_model[by_model["declined_rate"] > 0].sort_values("declined_rate", ascending=False)
        ins.append(f"{h['declined_rate']:.1%} of {'per-year calls' if per_year_mode else 'responses'} declined to answer: "
                   + ", ".join(f"{r['model_alias']} {r['declined_rate']:.0%}" for _, r in dcm.head(8).iterrows())
                   + ". Compare with --raw_prompt; the directive itself can change decline behaviour.")
    if h["escalated_rate"] == h["escalated_rate"] and h["escalated_rate"] > 0:
        ins.append(f"{h['escalated_rate']:.1%} of records contain escalated calls (stronger directive after a decline / stall; "
                   f"attempt texts are in Escalation_History; report these separately from fixed-level runs).")
    if h["contradiction_rate"] == h["contradiction_rate"] and h["contradiction_rate"] > 0:
        ctm = by_model[by_model["contradiction_rate"] > 0].sort_values("contradiction_rate", ascending=False)
        ins.append(f"{h['contradiction_rate']:.1%} of responses contradict themselves (table rows vs the model's own 'no player qualified' note): "
                   + ", ".join(f"{r['model_alias']} {r['contradiction_rate']:.0%}" for _, r in ctm.head(8).iterrows()) + ".")
    if h["stall_rate"] == h["stall_rate"] and h["stall_rate"] > 0:
        stm = by_model[by_model["stall_rate"] > 0].sort_values("stall_rate", ascending=False)
        ins.append(f"{h['stall_rate']:.1%} of responses were conversational stalls (no answer, only an offer to continue): "
                   + ", ".join(f"{r['model_alias']} {r['stall_rate']:.0%}" for _, r in stm.head(8).iterrows())
                   + ". Inspect Generated_Response of these records; --retry_stalled re-runs them.")
    if h["degenerate_rate"] == h["degenerate_rate"] and h["degenerate_rate"] > 0:
        dgm = by_model[by_model["degenerate_rate"] > 0].sort_values("degenerate_rate", ascending=False)
        ins.append(f"{h['degenerate_rate']:.1%} of responses end in a repetition loop (flag only, text kept): "
                   + ", ".join(f"{r['model_alias']} {r['degenerate_rate']:.0%}" for _, r in dgm.head(8).iterrows()) + ".")
    sv = by_strat.dropna(subset=["mean_jaccard_to_consensus"]).sort_values("mean_jaccard_to_consensus")
    if len(sv) > 1:
        ins.append(f"Strategy spread in agreement: best {sv.iloc[-1]['code']} ({sv.iloc[-1]['mean_jaccard_to_consensus']:.3f}), worst "
                   f"{sv.iloc[0]['code']} ({sv.iloc[0]['mean_jaccard_to_consensus']:.3f}).")
    rv = by_rag.dropna(subset=["mean_jaccard_to_consensus"])
    if len(rv) > 1:
        ins.append("Agreement by RAG tier: " + ", ".join(f"{r['code']}={r['mean_jaccard_to_consensus']:.3f}" for _, r in rv.iterrows()) + ".")
    if not pr.empty:
        col = f"mean_delta_{metric}"
        b = pr.sort_values(col).iloc[-1]
        ins.append(f"Largest paired effect over RAG0: {b['rag_tier']} (mean delta {metric} {b[col]:+.3f}; {b['wins']} wins / {b['ties']} ties / "
                   f"{b['losses']} losses over {b['paired_n']} pairs).")
    if len(ed) >= 6:
        e2 = ed.sort_values("share_of_records")
        ins.append("Editions returned by the fewest records: " + ", ".join(f"{r['WorldCup_Year']} ({r['share_of_records']:.0%})" for _, r in e2.head(3).iterrows())
                   + "; by the most: " + ", ".join(f"{r['WorldCup_Year']} ({r['share_of_records']:.0%})" for _, r in e2.tail(3).iterrows()) + ".")
    if per_year_mode and not ysum.empty:
        sg = ysum[ysum["gt_available"] & ysum["micro_recall"].notna()].sort_values("micro_recall")
        if len(sg) >= 2:
            ins.append(f"Per-year recall (pooled): hardest edition {int(sg.iloc[0]['year'])} ({sg.iloc[0]['micro_recall']:.2f}), "
                       f"easiest {int(sg.iloc[-1]['year'])} ({sg.iloc[-1]['micro_recall']:.2f}).")
        sa = ysum.sort_values("answered_share")
        ins.append("Lowest answered share (rows or explicit 'none'): " + ", ".join(f"{int(r['year'])} ({r['answered_share']:.0%})" for _, r in sa.head(3).iterrows()) + ".")
        ng = ysum[~ysum["gt_available"]]
        for _, r in ng.iterrows():
            ins.append(f"{int(r['year'])} has NO ground truth in the GT file: its rows are collected (mean {r['mean_rows']:.1f} rows per call, "
                       f"answered share {r['answered_share']:.0%}) but not scored.")
    if h["truncated_rate"] == h["truncated_rate"] and h["truncated_rate"] > 0:
        trm = by_model[by_model["truncated_rate"] > 0].sort_values("truncated_rate", ascending=False)
        ins.append(f"{h['truncated_rate']:.1%} of responses were truncated (models: "
                   + ", ".join(f"{r['model_alias']} {r['truncated_rate']:.0%}" for _, r in trm.head(8).iterrows())
                   + "); truncated answers keep their partial text and omit later rows. Use the '..._reached' columns to judge them on the editions they did reach.")
    if h["unparsed_rate"] == h["unparsed_rate"] and h["unparsed_rate"] > 0:
        ins.append(f"{h['unparsed_rate']:.1%} of records had no recognisable table row, no explicit 'no player qualifies' statement and no refusal "
                   f"(open the raw_calls/*_text.txt files; --reparse re-parses stored text after a parser change).")
    if h["mean_completion_tokens"] == h["mean_completion_tokens"]:
        ins.append(f"Mean reported completion tokens per cell: {h['mean_completion_tokens']:.0f}.")
    ins.append(f"Citation chains are complete (heuristic) in {df['citation_complete'].mean():.1%} of records.")
    add("\n17. KEY INSIGHTS (computed from the data above)")
    L.extend(f"  - {s}" for s in ins)
    add("\n18. LIMITATIONS")
    add("  - Consensus = agreement among records, not truth; a systematic error shared by most records becomes the consensus.")
    add("  - Citation audit is a keyword heuristic over the whole response; it cannot confirm a source exists or supports a claim.")
    add("  - Stall, decline, contradiction and repetition-loop detection are phrase/line heuristics; they only flag, never alter text.")
    add("  - Edition headers are detected from lines such as '**1954 FIFA World Cup**' / 'World Cup 1954' / 'Edition: 1954' / '### 6. 1958 - Sweden'; other layouts are missed.")
    add("  - 'Reached' = up to the latest edition year that appears in a parsed row or an edition header; an edition skipped inside that range counts as missed.")
    add("  - Rows are recovered from merged columns only when a known award name is present in the row text.")
    add("  - Rows containing 'does NOT qualify' / 'not a winner' are treated as the model's own exclusions and are not scored as predictions.")
    add("  - Name merging uses normalised names and token subsets; a bare surname shared by two players in one edition can be merged.")
    add("  - Per-year mode changes the user turn (the placeholder becomes the year) and the system message; do not compare its numbers directly with single-shot runs.")
    add("  - Editions absent from the ground-truth file (e.g. 2026) are never scored.")
    add("  - Sampling parameters cannot be set through `claude -p`; a call that hits the output-token cap is stored truncated, not retried.")
    add("  - Stronger directive levels / escalation change what the model is told; do not mix them with level-1 runs when comparing models.")
    add("  - One call per cell and year: small differences between cells may be sampling noise (no replicates).")
    add("\nFILES: " + (", ".join(Path(p).name for p in plots) if plots else "(no plots written)"))
    (outdir / "analysis_report.txt").write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Analysis written to {outdir} ({len(plots)} plots, report + CSVs)")
    print("\n".join(L[:21]))
    print(f"\n  Full report: {outdir / 'analysis_report.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_strategies(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in _split_tokens(tokens):
        u = t.upper().replace("-", "_")
        if u == "ALL":
            out += INDIVIDUAL_STRATEGIES
        elif u == "S4":
            out += [STRATEGY_ENTITY_FIRST, STRATEGY_CONSTRAINT_FIRST]
        elif u in CODE_TO_STRATEGY:
            out.append(CODE_TO_STRATEGY[u])
        elif t.lower() in INDIVIDUAL_STRATEGIES:
            out.append(t.lower())
        else:
            raise ValueError(f"Unknown strategy '{t}'. Use S1..S9, S11..S15 (S4 = S4A+S4B), a full name, or 'all'.")
    return list(dict.fromkeys(out))


def resolve_rag_tiers(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in _split_tokens(tokens):
        u = t.upper()
        if u == "ALL":
            out += RAG_TIERS
        elif u in CODE_TO_RAG:
            out.append(CODE_TO_RAG[u])
        elif t.lower() in RAG_TIERS:
            out.append(t.lower())
        else:
            raise ValueError(f"Unknown RAG tier '{t}'. Use RAG0..RAG6, a full tier name, or 'all'.")
    return list(dict.fromkeys(out))


def resolve_years(tokens: Optional[List[str]]) -> List[int]:
    if not tokens:
        return list(PER_YEAR_LIST)
    out: List[int] = []
    for t in _split_tokens(tokens):
        if not re.fullmatch(r"\d{4}", t):
            raise ValueError(f"Invalid year '{t}' (expected a 4-digit year).")
        out.append(int(t))
    return list(dict.fromkeys(out))


def main() -> int:
    global REFERENCE_FILE, GROUND_TRUTH_FILE
    p = argparse.ArgumentParser(
        description=(f"Football Player Analyser - Claude Code edition (script version {SCRIPT_VERSION})\n"
                     "Claude models only; every call is a streamed headless `claude -p` run inside a Claude Code cloud session.\n"
                     "Single-shot (default): user turn = the prompt as-is; --per_year: one call per year (placeholder replaced by the year).\n"
                     "Raw events of every call are saved under <cell dir>/raw_calls; the full log is <output root>/run.log.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(["", "Models (--model; several aliases or 'all')", "-" * 42]
                         + [f"  {a:<20} -> {MODEL_REGISTRY[a]}{'  [DEFAULT]' if a == DEFAULT_MODEL_ALIAS else ''}" for a in sorted(MODEL_REGISTRY)]
                         + ["", "Strategy selectors (--strategy / --S): "] + [f"  {STRATEGY_CODE[s]:<4} {s}" for s in INDIVIDUAL_STRATEGIES]
                         + ["  S4   = S4A + S4B", "", "RAG selectors (--rag_tier / --RAG):"] + [f"  {RAG_CODE[t]}  {t}" for t in RAG_TIERS]
                         + ["", f"Default --max_new_tokens = {GENERATION_MAX_NEW_TOKENS} single-shot / {PER_YEAR_MAX_NEW_TOKENS} per-year",
                            "Output roots: " + str(OUTPUT_ROOT) + "  |  " + str(OUTPUT_ROOT_PY)]))
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_ALIAS], choices=sorted(MODEL_REGISTRY) + ["all"])
    p.add_argument("--strategy", "--S", "-S", dest="strategy", nargs="+", default=["S1"], metavar="S<n>")
    p.add_argument("--rag_tier", "--RAG", dest="rag_tier", nargs="+", default=["RAG0"], metavar="RAG<n>")
    p.add_argument("--per_year", action="store_true", default=False)
    p.add_argument("--years", nargs="+", default=None, metavar="YYYY")
    p.add_argument("--claude_bin", default="claude", help="Claude Code CLI executable (default: claude)")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="seconds allowed per `claude -p` call")
    p.add_argument("--max_thinking_tokens", type=int, default=None,
                   help="sets MAX_THINKING_TOKENS for the call (0 = try to disable thinking); default: untouched")
    p.add_argument("--no_tool_directive", action="store_true", default=False,
                   help="append an explicit 'you have no tools - answer from your own knowledge' instruction to the "
                        "system directive (ignored with --raw_prompt); default: directive unchanged")
    p.add_argument("--no-resume", dest="no_resume", action="store_true", default=False)
    p.add_argument("--reference_file", default=REFERENCE_FILE, help="only for RAG1..RAG3 (and RAG6 distractors)")
    p.add_argument("--ground_truth_file", default=GROUND_TRUTH_FILE, help="only for RAG4..RAG6 and optional scoring")
    p.add_argument("--rag_max_rows", type=int, default=RAG_MAX_ROWS)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--raw_prompt", action="store_true", default=False,
                   help="no --system-prompt flag (Claude Code default system prompt); outputs go to *_RAW folders")
    p.add_argument("--directive_level", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--escalate_on_decline", action="store_true", default=False)
    p.add_argument("--retry_stalled", action="store_true", default=False)
    p.add_argument("--reparse", action="store_true", default=False)
    p.add_argument("--no_analysis", action="store_true", default=False)
    p.add_argument("--analyze_only", action="store_true", default=False)
    p.add_argument("--check_urls", action="store_true", default=False)
    args = p.parse_args()
    REFERENCE_FILE, GROUND_TRUTH_FILE = args.reference_file, args.ground_truth_file
    root = OUTPUT_ROOT_PY if args.per_year else OUTPUT_ROOT

    root.mkdir(parents=True, exist_ok=True)                       # full log of every run, kept next to the results
    fh = logging.FileHandler(root / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"=== run started: {' '.join(sys.argv)} (script version {SCRIPT_VERSION}) ===")

    if args.reparse:
        reparse_outputs(root)
        run_analysis(root, do_url_check=args.check_urls, raw=args.raw_prompt)
        return 0
    if args.analyze_only:
        run_analysis(root, do_url_check=args.check_urls, raw=args.raw_prompt)
        return 0
    try:
        strategies_to_run = resolve_strategies(args.strategy)
        tiers = resolve_rag_tiers(args.rag_tier)
        years = resolve_years(args.years) if args.per_year else []
    except ValueError as e:
        p.error(str(e))
    if args.years and not args.per_year:
        logger.warning("--years is only used together with --per_year; ignored.")
    if args.max_new_tokens is None:
        args.max_new_tokens = PER_YEAR_MAX_NEW_TOKENS if args.per_year else GENERATION_MAX_NEW_TOKENS
    if args.raw_prompt and (args.directive_level != 1 or args.escalate_on_decline):
        logger.warning("--raw_prompt disables the system directive: --directive_level / --escalate_on_decline are ignored.")
    if not shutil.which(args.claude_bin):
        print(f"ERROR: '{args.claude_bin}' (Claude Code CLI) not found on PATH. This script only runs through Claude Code.")
        return 1
    set_vars = [v for v in _BILLING_ENV_VARS if os.environ.get(v)]
    if set_vars:
        logger.warning(f"Environment variable(s) {set_vars} are set: the Claude Code CLI may bill through them instead of "
                       f"the cloud-session credit.")

    aliases = sorted(MODEL_REGISTRY) if "all" in args.model else list(dict.fromkeys(args.model))
    uses_rag = any(t != RAG_TIER_NO_EVIDENCE for t in tiers)
    n_cells = len(aliases) * len(strategies_to_run) * len(tiers)

    print("=" * 72)
    print(f"FOOTBALL PLAYER ANALYSER - CLAUDE CODE   (script version {SCRIPT_VERSION})")
    print("=" * 72)
    print(f"  Mode             : {'PER-YEAR loop (placeholder replaced by the year)' if args.per_year else 'single-shot (placeholder kept)'}")
    if args.per_year:
        print(f"  Years ({len(years)})     : {' '.join(str(y) for y in years)}")
    print(f"  Models ({len(aliases)}) : {', '.join(aliases)}")
    print(f"  Strategy(ies)    : {', '.join(STRATEGY_CODE[s] for s in strategies_to_run)}")
    print(f"  RAG tier(s)      : {', '.join(RAG_CODE[t] for t in tiers)}")
    print(f"  System directive : {'OFF (--raw_prompt: Claude Code default system prompt)' if args.raw_prompt else 'ON via --system-prompt (fallback: prepended to the user turn)'}")
    print(f"  Directive level  : {'-' if args.raw_prompt else args.directive_level}   escalate_on_decline: {args.escalate_on_decline and not args.raw_prompt}"
          f"   no_tool_directive: {args.no_tool_directive and not args.raw_prompt}")
    print(f"  Calls            : {n_cells * (len(years) if args.per_year else 1)} "
          f"({n_cells} cell(s){' x ' + str(len(years)) + ' years' if args.per_year else ''}; escalation can add more per declined call)")
    print(f"  Resume           : {'OFF (--no-resume)' if args.no_resume else 'ON (default)'}{'  + retry stalled' if args.retry_stalled else ''}")
    print(f"  Max new tokens   : {args.max_new_tokens} per call; max thinking tokens: {args.max_thinking_tokens}; timeout {args.timeout}s")
    print(f"  Output root      : {root}   (log: {root / 'run.log'}; raw events: <cell dir>/raw_calls)")
    print("=" * 72)

    ref_store: Optional[ReferenceStore] = None
    gt_store: Optional[GroundTruthStore] = None
    if uses_rag:
        try:
            if any(t in REF_TIERS for t in tiers) or RAG_TIER_FULL_WITH_DISTRACTORS in tiers:
                try:
                    ref_store = ReferenceStore(REFERENCE_FILE)
                except (FileNotFoundError, ValueError) as e:
                    if any(t in REF_TIERS for t in tiers):
                        raise
                    logger.warning(f"Reference file not loaded ({e}); RAG6 will have no distractors.")
            if any(t in GT_TIERS for t in tiers):
                gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}\n       RAG1..RAG6 need their evidence file; RAG0 needs none.")
            return 1
        for t in tiers:
            report_rag_coverage(t, ref_store, gt_store, args.rag_max_rows)

    overall_start = datetime.now()
    failed: List[str] = []
    for m_idx, alias in enumerate(aliases, 1):
        print(f"\n{'#' * 72}\n  MODEL [{m_idx}/{len(aliases)}] {alias}\n{'#' * 72}")
        try:
            client = ClaudeCodeClient(model_alias=alias, max_new_tokens=args.max_new_tokens, one_shot=not args.raw_prompt,
                                      directive_level=args.directive_level, per_year=args.per_year,
                                      claude_bin=args.claude_bin, timeout=args.timeout,
                                      max_thinking_tokens=args.max_thinking_tokens,
                                      no_tool_directive=args.no_tool_directive)
            client.raw_dir = root / "raw_calls"
            client.validate_access()
        except Exception as e:
            logger.error(f"{alias}: startup check failed - skipping this model ({e})")
            failed.append(f"{alias}[startup]")
            continue
        for tier in tiers:
            for strategy in strategies_to_run:
                print(f"\n{'-' * 72}\n  {alias} | {STRATEGY_CODE[strategy]} ({strategy}) | {RAG_CODE[tier]} ({tier})\n{'-' * 72}")
                try:
                    common = dict(strategy=strategy, client=client, no_resume=args.no_resume, rag_tier=tier,
                                  ref_store=ref_store, gt_store=gt_store, rag_max_rows=args.rag_max_rows,
                                  raw_prompt=args.raw_prompt, retry_stalled=args.retry_stalled,
                                  escalate_on_decline=args.escalate_on_decline)
                    analyser = (WorldCupYearLoopAnalyser(years=years, **common) if args.per_year
                                else WorldCupPlayerAnalyser(**common))
                    t0 = datetime.now()
                    results = analyser.process()
                    ok = sum(1 for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR"))
                    expected = len(years) if args.per_year else 1
                    print(f"  done: success={ok}/{expected} errors={len(results) - ok} time={datetime.now() - t0}  -> {analyser.output_file}")
                    if len(results) - ok or (args.per_year and len(results) < len(years)):
                        failed.append(f"{alias}[{STRATEGY_CODE[strategy]},{RAG_CODE[tier]}]")
                except KeyboardInterrupt:
                    print("\n  Interrupted - progress saved.")
                    return 130
                except Exception as e:
                    print(f"  Fatal error in {alias}/{STRATEGY_CODE[strategy]}/{RAG_CODE[tier]}: {e}")
                    traceback.print_exc()
                    failed.append(f"{alias}[{STRATEGY_CODE[strategy]},{RAG_CODE[tier]}]")

    print(f"\n{'=' * 72}\nALL RUNS COMPLETE\n  Total wall-clock time : {datetime.now() - overall_start}\n  Output root           : {root}")
    if failed:
        print(f"  Failed / incomplete: {failed}")
    print("=" * 72)
    if not args.no_analysis:
        try:
            run_analysis(root, do_url_check=args.check_urls, raw=args.raw_prompt)
        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())