---
name: worktree
description: >-
  Create and clean up lightweight git worktrees under .claude/worktrees/ for
  isolated, throwaway work (a spike, a parallel agent, a risky refactor). Use
  when the user asks to make, list, or remove a worktree in .claude/worktrees,
  wants an isolated checkout without touching main, or asks to sweep/clean up
  worktrees whose branches have already been merged into main — including the
  teardown that `/pr full` runs after its merge. NOT for the repo's
  container-backed sibling worktrees — those are devtools/worktree.sh (see
  "When NOT to use" below).
---

# Worktrees under `.claude/worktrees/`

These are **lightweight, disposable** worktrees: a second checkout of this repo
on its own branch, living inside `.claude/worktrees/<name>/`. That path is
git-ignored by a tracked line in `.gitignore`, so the checkout itself is never
committed. They get **no devcontainer, no named volumes, no build image** — that
is exactly what makes them cheap to create and tear down.

Run every command from the **main checkout root**
(`…-root/main`), not from inside a worktree.

## Create — and enter with the EnterWorktree tool

Creation is a two-step: lay the worktree down with `git worktree add`, then
**always switch the session into it** with the `EnterWorktree` tool
(`{"path": ".claude/worktrees/<name>"}`). Do not stop after the `git` command
— a worktree the session is not inside is a checkout nobody is using.

```bash
# New branch <name>, forked from current HEAD:
git worktree add .claude/worktrees/<name>

# Fork from an explicit start-point (branch, tag, or SHA):
git worktree add .claude/worktrees/<name> <start-point>

# Branch name different from the directory name:
git worktree add .claude/worktrees/<dir> -b <branch> <start-point>
```

Then:

```
EnterWorktree {"path": ".claude/worktrees/<name>"}
```

`git worktree add .claude/worktrees/foo` creates branch `foo` at the directory
`.claude/worktrees/foo`. If a branch named `foo` already exists, that bare form
fails — attach it instead with `git worktree add .claude/worktrees/foo foo`.

Heads-up before forking: `add` starts from **local** HEAD, which may be behind
`origin`. If main is stale (common right after a PR merges), refresh first with
`devtools/worktree.sh sync`, then create. `sync` locates whichever checkout has
`main` on it, so it works from inside one of these worktrees too — but it
fast-forwards `main` **in that other checkout**, not the branch you are on.

Notes on the tool half:

- `EnterWorktree {"name": "<name>"}` (no prior `git worktree add`) also works
  for the simple case — it creates the worktree *and* enters it in one call.
  But its base ref follows the `worktree.baseRef` setting (default `fresh` =
  `origin/<default-branch>`), not local HEAD, and it cannot take an explicit
  start-point or a branch name different from the directory — use the
  two-step form whenever those matter.
- To leave, use `ExitWorktree` (`action: "keep"` to preserve the checkout,
  `"remove"` to delete it) — but only when the user asks. Worktrees entered
  via `path` are never deleted by `ExitWorktree`; clean those up with the
  commands below.
- To hop into an already-existing worktree, the same
  `EnterWorktree {"path": …}` call is the way in.

## List

```bash
git worktree list                    # all worktrees, their HEADs and branches
git worktree list --porcelain        # machine-readable
```

## Clean up

```bash
# Remove the checkout (refuses if it has uncommitted changes):
git worktree remove .claude/worktrees/<name>

# Discard uncommitted changes and remove anyway:
git worktree remove --force .claude/worktrees/<name>

# Clear stale bookkeeping for any hand-deleted worktree dirs:
git worktree prune
```

Then delete the branch — **never silently lose commits**:

```bash
git branch -d <name>    # refuses an unmerged branch (safe default)
git branch -D <name>    # force-delete; only after confirming the work is saved
```

If `-d` refuses, the branch has commits not merged anywhere. Do **not** reach
for `-D` reflexively: surface it to the user (or push the branch / open a PR)
before discarding.

## Sweep worktrees already merged into main

When asked to clean up merged worktrees, sweep every entry under
`.claude/worktrees/` and remove the ones whose work is already in main.
`origin/main` is the truth, not local `main` — fetch first:

```bash
git fetch origin main --quiet
```

Then, for each worktree directory `wt` (from `git worktree list --porcelain`,
paths under `.claude/worktrees/` only):

1. **Skip if dirty.** `git -C "$wt" status --porcelain` non-empty means
   uncommitted work — leave it alone and report it.
2. **Classify the branch tip** (`git -C "$wt" rev-parse HEAD`):
   - *Merged:* `git merge-base --is-ancestor <tip> origin/main` succeeds.
   - *Squash-merged:* not an ancestor, but every commit is patch-equivalent
     to something upstream — `git cherry origin/main <branch>` prints only
     `-` lines. (PRs here are often squash-merged, so this case is common;
     `gh pr list --state merged --head <branch>` is a second confirmation.)
   - *Unmerged:* anything else — leave it alone and report it.
3. **Remove merged ones:**

```bash
git worktree remove .claude/worktrees/<name>
git branch -d <name>              # merged: -d succeeds
git branch -D <name>              # squash-merged only: -d refuses because the
                                  # SHAs differ; -D is safe *after* step 2
                                  # confirmed the patches are upstream
```

Finish with a one-line-per-worktree report: removed (merged), removed
(squash-merged), kept (dirty), kept (unmerged). Never remove a dirty or
unmerged worktree during a sweep, even with `--force` — that is a user
decision, not a cleanup.

## `/pr full` — ship, then tear the worktree down

`/pr full` is the `pr` skill's end-to-end form: land the work *and* reclaim the
worktree it ran in. The `full` argument is what asks for the teardown — a bare
`/pr` lands the PR and leaves the worktree standing for more commits, while
`/pr full` adds the cleanup below as its final step. Never tear a worktree down
on a bare `/pr` unless the user asks.

After the merge (the `pr` skill's steps 1–6), if the session ran inside a
`.claude/worktrees/` worktree:

1. **Move out first.** `ExitWorktree {"action": "keep"}` — you cannot remove a
   checkout the session occupies, and `keep` returns to the main checkout
   without deleting anything (worktrees entered via `path` are never removed by
   `ExitWorktree`, so this is only a move-out, not the teardown).
2. **Remove it** from the main checkout, exactly as "Sweep … merged into main"
   above: the branch is merged or squash-merged, so `git worktree remove` then
   `git branch -d <name>` (`-D` after a squash merge, since the SHAs differ).

If the branch did *not* merge — the `pr` skill stopped before merging (its
"When NOT to merge" cases) — there is nothing to reclaim: leave the worktree in
place and say so. `full` cleans up a *shipped* worktree, never an unmerged one.

## When NOT to use this skill

This repo has a **second, heavier** worktree system for full container-backed
development: `devtools/worktree.sh add|rm <name>`, which puts worktrees in
**sibling** directories under the repo root (`…-root/<name>`, not
`.claude/worktrees/`) and manages a devcontainer, three named volumes, and a
per-worktree build image for each. That is the **`devbox`** skill — use it when
the user wants to run the full test suite, a toolchain that only exists in the
image, or anything else needing the project's container.

Use **this** skill only for the lightweight `.claude/worktrees/` layout — quick
isolated checkouts, agent scratch space, no container.

## Docker leftovers (rare)

Plain `git worktree remove` on a `.claude/worktrees/` entry leaves no docker
state, because these worktrees never build a container. But if one somehow did
(e.g. someone ran `devcontainer.sh up` inside it), reconcile with:

```bash
devtools/worktree.sh gc --dry-run   # report orphaned containers/volumes/images
devtools/worktree.sh gc             # prune them (prompts first; --yes skips)
```

`gc` explicitly knows about `.claude/worktrees/` entries and cleans up anything
none of the live worktrees owns.
