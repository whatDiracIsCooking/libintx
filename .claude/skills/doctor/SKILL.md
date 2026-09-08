---
name: doctor
description: >-
  Diagnose a degraded environment: run devtools/doctor.sh and turn each
  finding into the fix, plus the failures it cannot see (orphaned build
  workers, a container that ignored your config edit). Use when something
  behaves oddly — commits fail with "No module named pre_commit", tests skip or
  vanish, git says "not a repository" inside a container, a worktree command
  cannot find its docker state — or when the user just asks what is wrong here.
---

# Reading a degraded environment

```bash
devtools/doctor.sh
```

It exits non-zero **only** on a FAIL (the core suite cannot run). A WARN means
some slice of the suite will skip, which is often the correct state of a
machine. So the output is not pass/fail: it is a list of what this environment
cannot do, and the job is to say which of those matter for the task at hand.

Run it **first** when anything behaves oddly, and again after any fix.

## Run it on both sides, and compare

This repo has a host half and a container half, and doctor is the thing that
tells you which one you are on:

```bash
devtools/doctor.sh                                    # the host
devtools/devcontainer.sh shell -c devtools/doctor.sh  # the CUDA container
```

On a bare host, **around a dozen warnings is the expected, healthy state**.
What is genuinely container-only is the clang/CUDA toolchain -- `cmake`,
`ninja`, `clang++`, `clang-scan-deps`, `clang-format`, `nvcc`,
`compute-sanitizer`, `nsys`, `ccache` -- plus all four `/opt/*` dependency
trees, which live in `Dockerfile.cuda` and nowhere else.

Note what is NOT on that list any more: `cmake-format` and `cmake-lint` come
from the `cmakelang[yaml]` dev dependency, and doctor resolves tools through
`VENV_PATHS` (`find_tool` in `devtools/lib.sh`) rather than `PATH`. On the host
they report `[ ok ]` with a `not on PATH; using .../.venv/bin/...` note, which
is normal rather than a finding. That is not a broken machine; it is the
laptop side of a two-sided workflow, good for editing, linting and the Python
suite. The same list appearing *inside* the container is a real problem.

Never report the host list on its own as "the environment is degraded." Say
which side you ran on, and what the other side would cover.

## Finding → what it means → fix

**`[FAIL] not inside a git repository`** — nothing else can be checked. Usually
means a container without the `.git` bind mount: a worktree's `.git` is a file
pointing outside the workspace folder, so `.devcontainer/devcontainer.json`
mounts the main checkout's `.git` at its literal host path. Fix that entry (see
both `.devcontainer/*.json`) and **`rebuild`** — `up` will not apply it.

**`[warn] <file> names volumes that do not start with '<name>-'`** — a
half-applied rename. `PROJECT_NAME` is spelled out in `devtools/config.sh` *and*
in **both** `devcontainer.json` files (JSON cannot source shell), and while they
disagree `worktree.sh rm` and `gc` do not recognise this project's volumes, so
every worktree you tear down leaks its whole set. Both files are checked
separately, so the warning names which one drifted. On a template that has never
been named, edit `name`, the `PROJECT_NAME` build arg and the volume prefixes
in both `.devcontainer/*.json` to match `PROJECT_NAME` in `devtools/config.sh`.

**`[warn] <file> still has the placeholder .git bind mount`** — the same
situation from the other end: `.devcontainer/*.json` bind-mounts the repo's git
common dir at a literal host path, and until that path is real, container
creation fails at mount time with an error naming a directory nobody recognises.
Write it by hand; it is
`git rev-parse --path-format=absolute --git-common-dir`. `rebuild` after, not
`up`.

**`[FAIL] DEVCONTAINER_CONFIG does not exist`** — `devtools/config.sh` points at
a `devcontainer.json` that is not there, and *every* `devcontainer.sh` command
refuses until it is fixed. The two shipped values are
`.devcontainer/cuda/devcontainer.json` (the default; has the C++ toolchain) and
`.devcontainer/devcontainer.json` (Python only).

**`[warn] worktree root not writable` / `primary checkout owned by uid N`**
— `worktree.sh add` cannot create sibling checkouts, and git ops in main fail.
Fix once with the numeric uid: `sudo chown -R $(id -u) <root>`. Not `chown
ubuntu` — the host user and the container's `ubuntu` are usually both uid 1000
under different names, so the name form fails on the host.

**`[warn] N broken/prunable linked worktree(s)`** — a worktree directory was
deleted out from under git. `git worktree prune` clears the bookkeeping; then
check `devtools/worktree.sh gc --dry-run`, because a hand-deleted worktree
usually left its container and volumes behind too.

**`[warn] no cblas.h` / `no lapacke.h`** — install them (`libopenblas-dev`,
`liblapacke-dev`). `libintx.blas` includes both and calls `LAPACKE_dsygvd`,
but `CMakeLists.txt` only `find_package(LAPACK)`s: without the CBLAS header
the tree fails to COMPILE, and without LAPACKE it configures and compiles and
then fails at LINK. Both are on the CUDA and CPU images already.

**`[warn] preset '<name>' does not resolve here`** — almost always the
`default` preset on a machine with no GPU. It sets
`CMAKE_CUDA_ARCHITECTURES=native`, which asks a live device at *configure*
time, so it cannot even configure there. Use
`CMAKE_PRESET=workstation CTEST_PRESET=workstation`, or one of the pinned
architecture presets (`ampere`, `hopper`, `portable`) to cross-build.

**`[warn] <tool> not found -- lost: <what>`** — an optional tool from
`DOCTOR_OPTIONAL_TOOLS` in `devtools/config.sh`. Install it *or* accept that its
slice of the suite will skip — but say which, because that is exactly the
coverage a green run will not have. This list is the project's to maintain: when
a test starts depending on a tool, add it here.

**`[warn] gh cannot see <owner>/<repo>`** — the token authenticates but was
never granted THIS repository. A fine-grained PAT names its repositories one by
one, and `gh auth status` reports a clean login either way, so without this
check the first symptom is `gh pr create` failing with "Could not resolve to a
Repository" *after* the branch is already pushed. Fix it under Repository
access on the token, not by re-authenticating. Blocks the `pr` and `milestone`
skills entirely.

**`[warn] gh present but GH_TOKEN empty`** — `gh` is installed but
unauthenticated, so the PR flow fails at push time rather than at the start.
Export a fine-grained PAT (Contents + Pull requests: read/write) on the **host**
and `rebuild`: the container takes `GH_TOKEN` from the host env at create time,
so an `export` after the container is up does not reach it.

**`[warn] <path> absent`** — a `DOCTOR_REQUIRED_PATHS` entry. Everything
libintx depends on is either vendored in-tree (eigen3, cutlass/cute, doctest,
pybind11) or a system package the images install, so a warning here means a
filtered or shallow clone — not a submodule to init. There are no submodules in
this repo, and doctor's "if it is a submodule" hint does not apply.

**`[warn] N orphaned docker resource(s)`** — containers, volumes and images from
worktrees that no longer exist (a plain `git worktree remove`, or an agent's
`.claude/worktrees/` checkout). Review with `devtools/worktree.sh gc --dry-run`,
then `gc` to remove. The count comes from that same `--dry-run`, so the two
always agree.

## What doctor.sh cannot see

Five failures that look like something else entirely:

1. **A container running your old config.** `up` reuses the running container
   and silently applies no change from `devcontainer.json` or the `Dockerfile`,
   while reporting success and the same container id. If a change "did not take
   effect", `devtools/devcontainer.sh rebuild` before debugging anything else.

2. **Orphaned build/test processes.** Killing `devtools/devcontainer.sh test` on the
   host kills `npx` only; ctest and the compiler jobs keep running
   inside at 100% CPU. Symptom: every later run is inexplicably slower.

   ```bash
   devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
   ```

   `pkill -f ctest` reaches only the driver — kill the test binaries by PID.

3. **A green run that tested nothing.** Skips are silent without `-rs`, and a
   suite whose tests all degrade-to-skip on a missing tool passes cleanly. This
   is the failure doctor exists for, but it only tells you the *tools* are
   missing — the `test` skill covers reading the skip list itself.

4. **A stale CMake cache.** Doctor never configures, so it cannot see that
   `build/` holds a toolchain variable you changed an hour ago. A change that
   "does nothing" wants `devtools/cpp-tier.sh --fresh`; a plain reconfigure
   keeps the old value.

5. **A C++ change with no gate behind it.** The push hook only fires on `.py`,
   and the CI `cpp:` job ships `if: false`. Nothing server-side checks a
   `src/`-only PR until you have a GPU runner — doctor reports tools, not that
   absence.

## Reporting

Say what is degraded and what it costs, not just the counts: "no failures, 21
warnings — all the C++ toolchain, because this is the host and it lives in the
CUDA image; the Python suite runs here, the C++ tier does not." A bare "doctor
passes" throws away the entire point of the command, and a bare "21 warnings"
reads as alarming when it is the expected state.
