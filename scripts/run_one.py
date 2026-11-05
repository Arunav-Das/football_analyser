#!/usr/bin/env python3
"""Run ONE experiment into its own folder, never touching earlier runs.

  python3 scripts/run_one.py --id 04 --label mod_bdo_RAG4_L3_64k_nothink_notools_all \
      --root claude_code_football_player_by_year \
      --env CLAUDE_CODE_DISABLE_THINKING=1 --env CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 \
      -- python3 football_analyzer_inf_mod.py --per_year ...

* refuses to start if runs/<id>_<label>/ already exists or if the script's output root (--root) is not empty,
  so a run can never mix with or overwrite another run;
* writes runs/<id>_<label>/run_config.json (command, env overrides, git commit, start/end time, exit code);
* console output -> runs/<id>_<label>/run_stdout.log;
* when the command exits (success or failure) the output root is moved to runs/<id>_<label>/<root>/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--root", required=True, help="output root the script writes to")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--note", default="")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        p.error("no command given after --")

    run_dir = Path("runs") / f"{a.id}_{a.label}"
    root = Path(a.root)
    if run_dir.exists():
        sys.exit(f"refusing: {run_dir} already exists")
    if root.exists() and (root.is_file() or any(root.iterdir())):
        sys.exit(f"refusing: output root {root} is not empty (move the previous run out first)")
    run_dir.mkdir(parents=True)

    env_over = dict(kv.split("=", 1) for kv in a.env)
    cfg = {"run_id": a.id, "label": a.label, "output_root": str(root), "command": cmd, "env_overrides": env_over,
           "note": a.note, "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_commit": git("rev-parse", "HEAD"),
           "git_dirty_files": [l for l in git("status", "--porcelain").splitlines() if l],
           "started_utc": now(), "finished_utc": None, "exit_code": None}
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with open(run_dir / "run_stdout.log", "w", encoding="utf-8") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, **env_over)).returncode

    cfg.update(finished_utc=now(), exit_code=rc)
    if root.exists():
        shutil.move(str(root), str(run_dir / root.name))
        cfg["output_moved_to"] = str(run_dir / root.name)
    else:
        cfg["output_moved_to"] = None
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"{run_dir}: exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
