#!/usr/bin/env python3
"""PreToolUse guard: block Write/Edit/NotebookEdit to the primary (`main`)
checkout of THIS repository.

The failure mode this defends against: a session told to work in a
`.claude/worktrees/<name>` worktree strays back into the shared main checkout
and edits it directly. It denies file-mutating tools whose target resolves
under the primary worktree, with two carve-outs:

  * the nested `<main>/.claude/worktrees/` subtree is ALLOWED (that IS where
    the lightweight worktrees live), and
  * anything outside the primary worktree is ALLOWED -- other repos, the
    sibling `...-root/<name>` container worktrees, /tmp, etc.

The primary worktree is found dynamically as the parent of the shared git
common dir (`git rev-parse --git-common-dir`), so there is no hardcoded path
and every checkout of this repo shares the same target.

Escape hatch: export CLAUDE_ALLOW_MAIN_EDITS=1 to edit main deliberately.

Wired from the committed `.claude/settings.json` as a PreToolUse hook on
Write|Edit|NotebookEdit. Fails OPEN on any error: a guard bug must never brick
editing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _allow() -> None:
    # No output + exit 0 == no opinion; the tool proceeds.
    sys.exit(0)


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def _primary_worktree(start: str) -> str:
    """Parent of the shared git common dir == the primary (main) worktree."""
    common = subprocess.check_output(
        ["git", "-C", start, "rev-parse",
         "--path-format=absolute", "--git-common-dir"],
        stderr=subprocess.DEVNULL, text=True,
    ).strip()
    return os.path.dirname(os.path.realpath(common))


def _existing_dir(path: str, fallback: str) -> str:
    """Nearest existing ancestor of a (possibly not-yet-created) path."""
    d = os.path.dirname(path)
    while d and not os.path.isdir(d):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return d if os.path.isdir(d) else fallback


def main() -> None:
    if os.environ.get("CLAUDE_ALLOW_MAIN_EDITS"):
        _allow()

    data = json.load(sys.stdin)
    ti = data.get("tool_input") or {}
    target = ti.get("file_path") or ti.get("notebook_path")
    if not target:
        _allow()

    path = os.path.abspath(os.path.normpath(os.path.expanduser(target)))
    start = _existing_dir(path, data.get("cwd") or os.getcwd())

    try:
        main_root = _primary_worktree(start)
    except Exception:
        _allow()  # not a git repo / git unavailable -> don't interfere

    worktrees = os.path.join(main_root, ".claude", "worktrees") + os.sep
    if path == main_root or path.startswith(main_root + os.sep):
        if path.startswith(worktrees):
            _allow()  # inside a nested worktree -> fine
        _deny(
            f"Blocked: {path} is in the protected primary checkout "
            f"({main_root}). Do this work inside a .claude/worktrees/<name> "
            f"worktree instead. To edit main deliberately, set "
            f"CLAUDE_ALLOW_MAIN_EDITS=1 in the environment."
        )

    _allow()  # outside the primary worktree entirely -> not our concern


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Fail open: never let a guard bug block edits everywhere.
        sys.exit(0)
