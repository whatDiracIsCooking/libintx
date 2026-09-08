#!/usr/bin/env bash
# Configure, build and ctest the tree -- the full run, of which
# devtools/cpp-smoke.sh is the cheap subset.
#
#   devtools/cpp-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test]
#                        [--tidy] [-j N] [--report DIR] [-- <ctest args>]
#
# It is NOT a push gate: libintx instantiates one translation unit per
# (bra,ket) angular-momentum pair, so a cold build costs minutes even at a
# modest LIBINTX_MAX_L, and a gate that costs minutes is one people learn to
# bypass with --no-verify. That is what devtools/cpp-smoke.sh is for. Run this
# one before opening a PR, on a machine with a card -- CI builds no CUDA at
# all.
#
# THE DEFAULT PRESET NEEDS A GPU. nvcc lives in Dockerfile.cuda and `default`
# sets CMAKE_CUDA_ARCHITECTURES=native, which asks a live device at CONFIGURE
# time -- so the normal way to call this is from inside that container (which
# DEVCONTAINER_CONFIG in config.sh already selects):
#
#   devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
#
# On a bare host use --preset workstation, which builds and tests the CPU half
# and needs no device. `devtools/doctor.sh` lists which situation you are in.
#
# Flags:
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh). The test preset follows it when one of
#                   the same name exists, and is skipped with a note when not
#                   -- only default, workstation, debug and asan have one.
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   --no-test       configure and build only.
#   --tidy          after a successful build, run clang-tidy over the .cc
#                   implementation units using the build's
#                   compile_commands.json. Advisory: it reports findings and
#                   does NOT fail the run. libintx ships no .clang-tidy, so
#                   without one this is clang-tidy's defaults on a tree that
#                   was never written to them -- expect noise. Off by default.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). This is a wide
#                   build -- a too-high N OOMs rather than failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#   -- <args>       everything after `--` is passed to ctest verbatim
#                   (e.g. `-- -R Constants` to scope the run). ctest has
#                   one entry per gtest case, so -R matches suite names.
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# Deliberately NOT what docker/compose.yaml's `build`/`test` services do, even
# though the commands overlap: compose owns the batch path (it brings its own
# container, tees to .log/, and is the thing CI calls), this owns the
# already-inside-a-container path. Changing the preset list means touching
# CMakePresets.json, which both read -- neither hardcodes one.
#
# The configure and build phases themselves are NOT here: they live in
# devtools/build.sh, which this sources for cpp_build, cpp_build_dir,
# cpp_require_build_tools and run_phase. `devtools/build.sh` run directly is the
# same two phases without a ctest -- one copy of the cmake invocations, used by
# both. This tier adds ctest and the advisory clang-tidy on top.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# After lib.sh: build.sh skips re-sourcing it (REPO_ROOT is already set) and
# only defines its functions. run_phase comes from here too, so the ctest and
# clang-tidy phases below tee into the same BUILD_LOG the build did.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
test_preset=$CTEST_PRESET
fresh=0
clean=0
run_tests=1
tidy=0
jobs=$BUILD_JOBS
report_dir=$SLOW_TIER_REPORTS
declare -a passthrough=()
preset_explicit=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; preset_explicit=1; shift 2;;
    --fresh)   fresh=1; shift;;
    --clean)   clean=1; shift;;
    --no-test) run_tests=0; shift;;
    --tidy)    tidy=1; shift;;
    -j|--jobs) jobs=$2; shift 2;;
    --report)  report_dir=$2; shift 2;;
    --)        shift; passthrough=("$@"); break;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)         echo "cpp-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

# --preset switches the test preset with it, or the run silently tests the
# wrong build directory: ctest --preset default reads build/, while
# --preset asan built into build-asan/.
[ "$preset_explicit" -eq 1 ] && test_preset=$preset

cpp_require_build_tools || exit 1

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/cpp-$stamp-$rev-$preset.log"
# run_phase (from build.sh) tees here, so configure, build, ctest and tidy all
# land in this one file.
BUILD_LOG=$log

echo "cpp-tier: $REPO_ROOT @ $rev  preset=$preset -> $log"

# Where this preset puts compile_commands.json, for --tidy below. build.sh's
# cpp_build_dir asks CMake rather than guessing, because `default` uses build/
# but the others use build-<preset>/, and a hardcoded build/ would have --tidy
# silently lint the previous preset's database.
build_dir=$(cpp_build_dir "$preset")

# `jq -e` is not assumed to be present, so the "does this test preset exist"
# question is answered by asking ctest, which is. --show-only lists the tests a
# preset would run and exits non-zero when the preset is unknown.
has_test_preset() {
  [ -n "$test_preset" ] || return 1
  ctest --preset "$test_preset" --show-only >/dev/null 2>&1
}

rc=0
set +e

# Configure + build, both phases from build.sh (the one place they are spelled
# out). ctest and the advisory tidy below are this tier's own.
cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?

if [ "$rc" -eq 0 ] && [ "$run_tests" -eq 1 ]; then
  if has_test_preset; then
    run_phase ctest ctest --preset "$test_preset" "${passthrough[@]}" || rc=$?
  else
    echo "cpp-tier: no test preset '${test_preset:-<empty>}' -- built only" \
      | tee -a "$log"
    echo "          (CMakePresets.json defines test presets for default," \
      "workstation, debug and asan)" | tee -a "$log"
  fi
fi

# clang-tidy is deliberately advisory, and here it is barely more than a
# convenience: libintx ships no `.clang-tidy`, so this runs the tool's defaults
# over a tree that was never written to them, on a fork whose upstream will not
# take the churn. Add a `.clang-tidy` with a narrow check list before treating
# any of its output as a to-do, and only then consider making it a gate.
if [ "$rc" -eq 0 ] && [ "$tidy" -eq 1 ]; then
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "cpp-tier: clang-tidy not found -- skipping --tidy" | tee -a "$log"
  else
    mapfile -t tidy_files < <(git -C "$REPO_ROOT" ls-files 'src/*.cc')
    if [ "${#tidy_files[@]}" -eq 0 ]; then
      echo "cpp-tier: no src/*.cc to tidy" | tee -a "$log"
    else
      # Advisory: the phase's own status is reported but never folded into rc.
      run_phase clang-tidy clang-tidy -p "$build_dir" --quiet "${tidy_files[@]}" \
        || echo "cpp-tier: clang-tidy reported findings (advisory)" | tee -a "$log"
    fi
  fi
fi

set -e

verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
summary="cpp-tier $verdict  rev=$rev  preset=$preset  stamp=$stamp"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
