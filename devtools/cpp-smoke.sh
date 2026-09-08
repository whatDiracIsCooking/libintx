#!/usr/bin/env bash
# The C++ smoke gate: an incremental build plus the cheap end of the suite.
#
#   devtools/cpp-smoke.sh
#
# What it does, and why it is shaped this way:
#   * INCREMENTAL configure + build, reusing devtools/build.sh's cpp_build --
#     the one copy of the cmake phases. Warm (you have been building) this is
#     seconds; cold (a fresh worktree) it is the full kernel explosion, which
#     for libintx is minutes. It never passes --fresh.
#   * runs the host tests named by SMOKE_TESTS in devtools/config.sh, and then
#     the ones named by SMOKE_GPU_TESTS if a device is actually present.
#
# It builds with SMOKE_PRESET (`workstation` by default) rather than
# CMAKE_PRESET, so a push from a machine with no GPU is still gated: the
# `default` preset's CMAKE_CUDA_ARCHITECTURES=native queries a live device at
# CONFIGURE time and cannot even configure without one.
#
# This is a BUILD + SMOKE gate on purpose. It catches "does the tree still
# compile, link and get the easy answers right" without the minutes that get a
# gate bypassed; the full run -- every angular-momentum class of md2/md3/md4,
# every K engine case at the configured LMAX -- stays in devtools/cpp-tier.sh.
#
# It SKIPS (exit 0, not a failure) when cmake or the generator is absent.
# Skipping keeps such a push off --no-verify; CI and cpp-tier.sh remain the
# backstop.
#
# Bypass a one-off with `git push --no-verify`.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# After lib.sh: build.sh skips re-sourcing it (REPO_ROOT is set) and only
# defines cpp_build, cpp_build_dir, cpp_require_build_tools and run_phase.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build.sh"
cd "$REPO_ROOT"

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage_from_header "${BASH_SOURCE[0]}"; exit 0; }

preset=$SMOKE_PRESET
jobs=$BUILD_JOBS

skip() {
  echo "cpp-smoke: $1"
  echo "cpp-smoke: skipping (CI and devtools/cpp-tier.sh remain the backstop)"
  exit 0
}

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 ||
    skip "$tool not found -- the build toolchain lives in the container"
done

report_dir=$SLOW_TIER_REPORTS
case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_LOG="$report_dir/smoke-$stamp-$rev-$preset.log"

echo "cpp-smoke: $REPO_ROOT @ $rev  preset=$preset  (incremental build + smoke)"
echo "cpp-smoke: full log -> $BUILD_LOG"

build_dir=$(cpp_build_dir "$preset")

rc=0
set +e

echo "cpp-smoke: building (incremental, preset=$preset, -j ${jobs:-auto}) ..."
cpp_build "$preset" 0 0 "$jobs" >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "cpp-smoke: BUILD FAILED (rc=$rc) -- see $BUILD_LOG" >&2
  tail -25 "$BUILD_LOG" >&2
  set -e
  exit "$rc"
fi

# Run one ctest selection: append the full output to the log, echo the summary
# lines, return ctest's status. A selection that matches nothing is a failure,
# not a pass -- "No tests were found" is how a renamed test silently stops
# being gated, and libintx's EXCLUDE_FROM_ALL test targets make that easy to
# do by accident.
run_ctest() {
  local label=$1 selection=$2
  [ -n "$selection" ] || return 0
  echo "--- $label: ctest -R $selection ---" >>"$BUILD_LOG"
  local out rc2
  out=$(ctest --test-dir "$build_dir" --output-on-failure -R "$selection" 2>&1)
  rc2=$?
  printf '%s\n' "$out" >>"$BUILD_LOG"
  printf '%s\n' "$out" | grep -E 'tests passed|tests failed|No tests were found' |
    sed "s/^/cpp-smoke:   /"
  if printf '%s\n' "$out" | grep -q 'No tests were found'; then
    echo "cpp-smoke: $label selection matched no tests -- check SMOKE_TESTS in devtools/config.sh" >&2
    return 1
  fi
  return $rc2
}

echo "cpp-smoke: host tests ..."
run_ctest host "$SMOKE_TESTS" || rc=$?

# Probe the driver device NODES, not nvidia-smi: NVML routinely fails with
# "Failed to initialize NVML: Unknown Error" inside a container whose CUDA
# runtime -- and thus /dev/nvidia* -- works perfectly.
if [ -n "$SMOKE_GPU_TESTS" ] &&
   [ -e /dev/nvidiactl ] && ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  echo "cpp-smoke: GPU smoke ..."
  run_ctest gpu "$SMOKE_GPU_TESTS" || rc=$?
else
  echo "cpp-smoke: no CUDA device node (/dev/nvidia*) -- GPU smoke skipped"
fi

set -e

if [ "$rc" -ne 0 ]; then
  echo "cpp-smoke: FAILED (rc=$rc) -- see $BUILD_LOG (bypass one push with --no-verify)" >&2
  exit "$rc"
fi
echo "cpp-smoke: PASS"
