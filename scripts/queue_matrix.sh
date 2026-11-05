#!/usr/bin/env bash
# Experiment matrix, run one after another. Each run -> runs/<id>_<label>/ (scripts/run_one.py), then commit + push.
# Common to every run: claude-sonnet-4-6, per-year, all 23 editions, strategy S1, directive level 3,
# 64,000-token output cap, 1,800 s timeout, --no-resume.
#   thinking off    : CLAUDE_CODE_DISABLE_THINKING=1 CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 --max_thinking_tokens 0
#   thinking 1024   : CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 --max_thinking_tokens 1024   (fixed low budget)
#   tools off       : --no_tool_directive  (CLI already gets --tools "" and --max-turns 1)
#   tools on        : --enable_tools WebSearch,WebFetch --max_turns 20 --max_budget_usd 2.0
set -u
cd "$(dirname "$0")/.."
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMON="--model claude-sonnet-4-6 --per_year --strategy S1 --directive_level 3 --max_new_tokens 64000 --timeout 1800 --no-resume"
OFF_ENV="--env CLAUDE_CODE_DISABLE_THINKING=1 --env CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1"
LOW_ENV="--env CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1"
TOOLS="--enable_tools WebSearch,WebFetch --max_turns 20 --max_budget_usd 2.0"
MOD="python3 football_analyzer_inf_mod.py"
DOM="python3 claude_football_inf_dom.py --dl_definition open"
MOD_ROOT=claude_code_football_player_by_year
DOM_ROOT=claude_code_football_player_dl_open_by_year

commit() {   # $1 = run folder
  rm -rf __pycache__ scripts/__pycache__
  git add "$1" >/dev/null 2>&1
  git commit -q -m "Add run $(basename "$1")" -m "$(python3 -c "import json,sys;c=json.load(open('$1/run_config.json'));print('Command: '+' '.join(c['command'])+'\nEnv: '+json.dumps(c['env_overrides'])+'\nExit code: '+str(c['exit_code'])+'\nStarted '+c['started_utc']+', finished '+str(c['finished_utc']))")" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012yeSnfoQP9BeUVEf3CuEVM" && \
  for i in 1 2 3 4; do git push -q origin "$BRANCH" && break; sleep $((2**i)); done
  echo "QUEUE committed $(basename "$1") $(git log --oneline -1 | cut -c1-7)"
}

run() {      # $1 id, $2 label, $3 root, $4 env flags, rest = command
  local id=$1 label=$2 root=$3 envs=$4; shift 4
  echo "QUEUE start ${id}_${label} $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  python3 scripts/run_one.py --id "$id" --label "$label" --root "$root" $envs -- "$@"
  echo "QUEUE done ${id}_${label} exit=$? $(date -u +%H:%M:%S)"
  commit "runs/${id}_${label}"
}

# shellcheck disable=SC2086
{
run 04 mod_bdo_RAG4_L3_64k_nothink_notools_all   $MOD_ROOT "$OFF_ENV" $MOD $COMMON --rag_tier RAG4 --max_thinking_tokens 0 --no_tool_directive
run 05 mod_bdo_RAG5_L3_64k_nothink_notools_all   $MOD_ROOT "$OFF_ENV" $MOD $COMMON --rag_tier RAG5 --max_thinking_tokens 0 --no_tool_directive
run 06 mod_bdo_RAG0_L3_64k_think1024_notools_all $MOD_ROOT "$LOW_ENV" $MOD $COMMON --rag_tier RAG0 --max_thinking_tokens 1024 --no_tool_directive
run 07 mod_bdo_RAG0_L3_64k_nothink_tools_all     $MOD_ROOT "$OFF_ENV" $MOD $COMMON --rag_tier RAG0 --max_thinking_tokens 0 $TOOLS
run 08 mod_bdo_RAG0_L3_64k_think1024_tools_all   $MOD_ROOT "$LOW_ENV" $MOD $COMMON --rag_tier RAG0 --max_thinking_tokens 1024 $TOOLS
run 09 dom_open_RAG0_L3_64k_think1024_notools_all $DOM_ROOT "$LOW_ENV" $DOM $COMMON --rag_tier RAG0 --max_thinking_tokens 1024 --no_tool_directive
run 10 dom_open_RAG0_L3_64k_nothink_tools_all     $DOM_ROOT "$OFF_ENV" $DOM $COMMON --rag_tier RAG0 --max_thinking_tokens 0 $TOOLS
run 11 dom_open_RAG0_L3_64k_think1024_tools_all   $DOM_ROOT "$LOW_ENV" $DOM $COMMON --rag_tier RAG0 --max_thinking_tokens 1024 $TOOLS
}
echo "QUEUE all done $(date -u +%H:%M:%S)"
