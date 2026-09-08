#!/usr/bin/env bash
# Report what is degraded in this environment, so "why did half the suite skip"
# is one command rather than a green run you cannot trust. Runs on the host and
# inside the container. Exits non-zero only on a FAIL (the core suite cannot
# run); a WARN means a slice of the suite will skip, which is often fine.
#
#   devtools/doctor.sh
#
# What counts as optional here is project-specific and comes from
# devtools/config.sh: DOCTOR_OPTIONAL_TOOLS and DOCTOR_REQUIRED_PATHS. Add the
# project's own device compiler, submodule or fixture tree there rather than
# editing this script.
set -uo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

fails=0
warns=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n'  "$1"; warns=$((warns+1)); }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n'  "$1"; fails=$((fails+1)); }
note() { printf '         %s\n' "$1"; }

# --- repo -----------------------------------------------------------------
echo "repo"
TOP=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$TOP" ]; then
  fail "not inside a git repository"
  echo; echo "cannot check anything else outside the repo"; exit 1
fi
ok "git repo at $TOP"

# The <root>/<branch> layout is anchored at the PRIMARY checkout, not at
# whichever worktree doctor happens to be run from: the parent of the shared
# git common dir is the primary worktree, and its parent is the root. Deriving
# these from $TOP was wrong in exactly the place it matters -- run from
# .claude/worktrees/<name> it called .claude/worktrees the "worktree root" and
# looked for main at .claude/worktrees/main.
COMMON=$(git -C "$TOP" rev-parse --git-common-dir 2>/dev/null)
case "$COMMON" in /*) : ;; *) COMMON="$TOP/$COMMON" ;; esac
PRIMARY=$(cd "$(dirname "$COMMON")" 2>/dev/null && pwd) || PRIMARY=$TOP

# The worktree root must be writable or `devtools/worktree.sh add` cannot make
# sibling checkouts.
ROOT=$(dirname "$PRIMARY")
if [ -w "$ROOT" ]; then
  ok "worktree root writable ($ROOT)"
else
  warn "worktree root not writable: $ROOT"
  note "worktree.sh add will fail; fix once: sudo chown $(id -u) $ROOT"
fi

# PROJECT_NAME is duplicated in every devcontainer.json (JSON cannot source
# shell), and if they drift, worktree.sh rm/gc stop recognising this project's
# volumes and silently leak them. Both variants are checked: the light one is
# what a bare `devcontainer up` picks, so a rename that missed it leaks from
# whichever side you forgot.
for rel in .devcontainer/devcontainer.json .devcontainer/cuda/devcontainer.json; do
  DCJSON="$TOP/$rel"
  [ -f "$DCJSON" ] || continue
  if grep -q "source=$PROJECT_NAME-" "$DCJSON"; then
    ok "PROJECT_NAME '$PROJECT_NAME' matches $rel volume names"
  elif grep -q 'type=volume' "$DCJSON"; then
    warn "$rel names volumes that do not start with '$PROJECT_NAME-'"
    note "worktree.sh rm/gc will not recognise them; align the two"
    note "  config: devtools/config.sh PROJECT_NAME   json: $DCJSON"
  fi
done

# Which of them devcontainer.sh will actually drive. A path that does not exist
# is a hard stop for every container command, so it is a FAIL rather than a
# warning.
case "${DEVCONTAINER_CONFIG:-}" in
  "") note "DEVCONTAINER_CONFIG unset: the CLI picks .devcontainer/devcontainer.json" ;;
  /*) DCACTIVE=$DEVCONTAINER_CONFIG ;;
  *) DCACTIVE="$TOP/$DEVCONTAINER_CONFIG" ;;
esac
if [ -n "${DCACTIVE:-}" ]; then
  if [ -f "$DCACTIVE" ]; then
    ok "devcontainer config: $DEVCONTAINER_CONFIG"
  else
    fail "DEVCONTAINER_CONFIG does not exist: $DCACTIVE"
    note "every devtools/devcontainer.sh command will refuse; fix devtools/config.sh"
  fi
fi

# The .git bind mount is an absolute host path you write by hand. Left
# as the placeholder it fails container creation at mount time -- an error that
# names a path nobody recognises rather than the config that produced it.
for rel in .devcontainer/devcontainer.json .devcontainer/cuda/devcontainer.json; do
  DCJSON="$TOP/$rel"
  [ -f "$DCJSON" ] || continue
  if grep -q '/absolute/path/to/your-root/main' "$DCJSON"; then
    warn "$rel still has the placeholder .git bind mount"
    note "point it at this repo's common git dir:"
    note "  git rev-parse --path-format=absolute --git-common-dir"
  fi
done

# --- worktrees ------------------------------------------------------------
# Two failure modes that are local usability damage rather than lost work
# (committed work is safe on the remote), so these are warn/note, never fail:
# a primary checkout owned by another uid, and linked worktrees whose admin
# state has broken.
echo "worktrees"

owner=$(stat -c %u "$PRIMARY" 2>/dev/null)
if [ -n "$owner" ] && [ "$owner" != "$(id -u)" ]; then
  warn "primary checkout owned by uid $owner, not $(id -u): $PRIMARY"
  note "git and worktree ops there fail; fix: sudo chown -R $(id -u) $ROOT"
else
  ok "primary checkout owned by current uid ($PRIMARY)"
fi
if [ ! -d "$ROOT/main" ]; then
  note "no sibling main checkout at $ROOT/main (not the <root>/<branch> layout here)"
fi

# `prunable` means git found a linked worktree whose gitdir points nowhere --
# the working tree was removed out from under it. A worktrees/<name> admin dir
# with no readable `gitdir` file is the other half. `git worktree prune` clears
# both.
prunable=$(git worktree list --porcelain 2>/dev/null | grep -c '^prunable')
corrupt=0
if [ -d "$COMMON/worktrees" ]; then
  for wtd in "$COMMON"/worktrees/*/; do
    [ -d "$wtd" ] || continue
    [ -r "${wtd}gitdir" ] || corrupt=$((corrupt+1))
  done
fi
broken=$((prunable + corrupt))
if [ "$broken" -gt 0 ]; then
  warn "$broken broken/prunable linked worktree(s) ($prunable prunable, $corrupt corrupt admin dir)"
  note "clean up: git worktree prune; then recreate via devtools/worktree.sh"
else
  ok "no broken/prunable linked worktrees"
fi

# --- build toolchain (the core suite) -------------------------------------
# libintx is C++ end to end: there is no venv, no lockfile and no Python test
# suite to check here. What decides whether the tree builds at all is the
# compiler, the generator, and the two system headers libintx.blas includes.
echo "build toolchain"
if command -v cmake >/dev/null 2>&1; then
  ok "cmake $(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
else
  fail "cmake not found -- nothing configures without it"
fi
if [ -f "$TOP/CMakePresets.json" ]; then
  ok "CMakePresets.json present (preset=$CMAKE_PRESET, ctest=${CTEST_PRESET:-none})"
  if ! cmake --preset "$CMAKE_PRESET" -N >/dev/null 2>&1; then
    warn "preset '$CMAKE_PRESET' does not resolve here"
    note "the 'default' preset needs CUDA and a live device (native arch);"
    note "on a bare host use CMAKE_PRESET=workstation CTEST_PRESET=workstation"
  fi
else
  warn "no CMakePresets.json -- devtools/build.sh and cpp-tier.sh drive presets"
fi

# libintx.blas includes <cblas.h> and <lapacke.h> and calls LAPACKE_dsygvd, but
# CMakeLists.txt only find_package(LAPACK)s and links LAPACK::LAPACK. On a
# platform where the LAPACK it finds is not also the LAPACKE (a plain OpenBLAS
# on Linux, say), the tree configures cleanly and then fails at LINK time with
# an undefined LAPACKE_dsygvd -- a long way from the missing package. Both
# halves are checked separately because they fail at different phases.
cblas_found=""
for d in /usr/include /usr/local/include /usr/include/"$(uname -m)"-linux-gnu \
         /opt/homebrew/include; do
  [ -f "$d/cblas.h" ] && { cblas_found="$d/cblas.h"; break; }
done
if [ -n "$cblas_found" ]; then
  ok "cblas.h: $cblas_found"
elif [ -n "${LIBINTX_CBLAS_H:-}" ]; then
  note "cblas.h not on the default include path; LIBINTX_CBLAS_H is set"
else
  warn "no cblas.h found -- libintx.blas fails to COMPILE (md3/md4 link it)"
  note "install a CBLAS (libopenblas-dev), or set -DLIBINTX_CBLAS_H=<header>"
fi
lapacke_found=""
for d in /usr/include /usr/local/include /usr/include/"$(uname -m)"-linux-gnu \
         /opt/homebrew/include; do
  [ -f "$d/lapacke.h" ] && { lapacke_found="$d/lapacke.h"; break; }
done
if [ -n "$lapacke_found" ]; then
  ok "lapacke.h: $lapacke_found"
elif [ "$(uname -s)" = "Darwin" ]; then
  note "no lapacke.h; on macOS Accelerate supplies LAPACKE (LIBINTX_APPLE_ACCELERATE)"
else
  warn "no lapacke.h found -- libintx.blas fails to LINK on LAPACKE_dsygvd"
  note "install liblapacke-dev, or set -DLIBINTX_LAPACKE_H=<header>"
fi

# --- optional tools (project-declared) ------------------------------------
echo "optional tools"
config_lines "$DOCTOR_OPTIONAL_TOOLS"
if [ "${#CONFIG_LINES[@]}" -eq 0 ]; then
  note "none declared (DOCTOR_OPTIONAL_TOOLS in devtools/config.sh)"
fi
for entry in "${CONFIG_LINES[@]}"; do
  tool=${entry%%:*}
  what=${entry#*:}
  # find_tool, not `command -v`: the cmakelang tools come from the project venv
  # and are on PATH inside the image but not on the host. See devtools/lib.sh.
  if toolpath=$(find_tool "$tool"); then
    ver=$("$toolpath" --version 2>/dev/null | head -1)
    ok "$tool${ver:+: $ver}"
    command -v "$tool" >/dev/null 2>&1 || note "not on PATH; using $toolpath"
  else
    warn "$tool not found -- lost:${what:+ $what}"
  fi
done

# gh is the one optional tool with a second half: installed but unauthenticated
# is a distinct state, and the PR flow fails at push time rather than at start.
if command -v gh >/dev/null 2>&1 && [ -z "${GH_TOKEN:-}" ]; then
  warn "gh present but GH_TOKEN empty -- gh is unauthenticated"
  note "export a fine-grained PAT (Contents + Pull requests: R/W) on the host"
# A token can authenticate as the right user and still be unable to see THIS
# repo: a fine-grained PAT names its repositories one by one, and one that has
# not been given this one 404s. `gh auth status` reports a clean login either
# way, so the first sign of trouble is `gh pr create` failing with "Could not
# resolve to a Repository" after the branch is already pushed. One API call
# here turns that into a warning you get before starting.
elif command -v gh >/dev/null 2>&1; then
  # Match on github.com FIRST, then extract. Stripping known URL prefixes and
  # accepting anything left containing a slash was wrong: a clone from a local
  # path has an origin like /home/you/projects/thing, which survives every
  # substitution and still looks like `owner/repo` to a `*/*` glob. The probe
  # then asked GitHub for `repos//home/you/...`, got a 404, and reported "the
  # token lacks access to this repository" about a repository that is not on
  # GitHub at all. Same for a GitLab remote, and for a fresh `git init` with no
  # origin yet.
  ORIGIN=$(git -C "$TOP" remote get-url origin 2>/dev/null || true)
  case "$ORIGIN" in
    *github.com[:/]*)
      SLUG=$(printf '%s' "$ORIGIN" | sed -E 's#^.*github\.com[:/]+##; s#\.git$##; s#/+$##')
      # owner/repo and nothing else: two non-empty segments, no extra slashes.
      if printf '%s' "$SLUG" | grep -qE '^[^/]+/[^/]+$'; then
        if ${GH_PROBE_TIMEOUT:-timeout 10} gh api "repos/$SLUG" >/dev/null 2>&1; then
          ok "gh can reach $SLUG"
        else
          warn "gh cannot see $SLUG -- the PR and milestone flows will fail"
          note "the token authenticates but lacks access to this repository;"
          note "add it under Repository access on the fine-grained PAT"
        fi
      else
        note "origin looks like github but is not owner/repo: $ORIGIN"
      fi
      ;;
    "") note "no origin remote yet -- nothing to check gh against" ;;
    *) note "origin is not a github remote; skipping the gh reachability check" ;;
  esac
fi

# --- required paths (submodules, fixture trees) ---------------------------
config_lines "$DOCTOR_REQUIRED_PATHS"
if [ "${#CONFIG_LINES[@]}" -gt 0 ]; then
  echo "required paths"
  for entry in "${CONFIG_LINES[@]}"; do
    rpath=${entry%%:*}
    what=${entry#*:}
    # Absolute entries are taken as-is: the prebuilt dependency trees this
    # project's `default` CMake preset needs live at /opt inside the image, not
    # under the checkout. Relative ones stay repo-relative (submodules,
    # fixtures), so doctor works from any worktree.
    case "$rpath" in /*) abs=$rpath ;; *) abs="$TOP/$rpath" ;; esac
    if [ -e "$abs" ]; then
      ok "$rpath present"
    else
      warn "$rpath absent --${what:+ $what}"
      case "$rpath" in
        /*) ;;
        *) note "if it is a submodule: git submodule update --init --recursive" ;;
      esac
    fi
  done
fi

# --- docker hygiene (report-only) -----------------------------------------
# Worktree containers/volumes/images leak when a worktree is removed with plain
# `git worktree remove` (bypassing worktree.sh rm), or via an agent's worktrees
# under .claude/worktrees/. Surface the count instead of letting it accrete.
# The enumeration lives in `worktree.sh gc` -- called in --dry-run, which
# neither prunes nor deletes, so there is one source of truth.
echo "docker hygiene"
if command -v docker >/dev/null 2>&1; then
  wt="$TOP/devtools/worktree.sh"
  if [ -x "$wt" ]; then
    n=$("$wt" gc --dry-run 2>/dev/null \
        | sed -n 's/^orphans: \([0-9]*\).*/\1/p')
    if [ "${n:-0}" -gt 0 ]; then
      warn "$n orphaned docker resource(s) from removed worktrees"
      note "review and remove: devtools/worktree.sh gc"
    else
      ok "no orphaned containers/volumes/images"
    fi
  else
    note "devtools/worktree.sh not executable -- skipping orphan check"
  fi
else
  note "docker not found here (expected inside the container; run on the host)"
fi

# --- summary --------------------------------------------------------------
echo
if [ "$fails" -gt 0 ]; then
  printf '\033[31m%d FAIL\033[0m, %d warn\n' "$fails" "$warns"
  exit 1
elif [ "$warns" -gt 0 ]; then
  printf 'no failures, \033[33m%d warn\033[0m (some tests will skip)\n' "$warns"
else
  printf '\033[32mall good\033[0m\n'
fi
