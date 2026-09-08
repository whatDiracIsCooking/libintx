#!/usr/bin/env bash
# Configure and build the tree -- the build half of devtools/cpp-tier.sh, on
# its own so a build-without-ctest run has a name of its own and cpp-tier does
# not carry its own copy of the two cmake phases.
#
#   devtools/build.sh [--preset NAME] [--fresh] [--clean] [-j N] [--report DIR]
#
# THE DEFAULT PRESET NEEDS A GPU. nvcc lives in Dockerfile.cuda and
# `default` sets CMAKE_CUDA_ARCHITECTURES=native, which asks a live device at
# CONFIGURE time -- so the normal way to call this is from inside that
# container (which DEVCONTAINER_CONFIG in config.sh already selects):
#
#   devtools/devcontainer.sh shell -c devtools/build.sh
#
# On a bare host use --preset workstation, which builds the CPU half of libintx
# and needs no device at all. `devtools/doctor.sh` lists which of the two
# situations you are in.
#
# The build presets name `all all.tests`, which matters: libintx declares every
# test executable EXCLUDE_FROM_ALL, so a build that does not name all.tests
# produces the libraries and no test binaries.
#
# Flags (a subset of cpp-tier.sh's, minus everything about running tests):
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh).
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). libintx explodes
#                   its MD kernels into one translation unit per (bra,ket)
#                   angular-momentum pair, so this is a wide build -- a too-high
#                   N OOMs rather than failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# Two lives. Run directly, this is the build-only tier: configure + build,
# logged, with its own PASS/FAIL summary line (tagged `build`, so it is
# distinct from cpp-tier's `cpp-tier` line in the same summary.log). SOURCED --
# which is how devtools/cpp-tier.sh uses it -- it defines run_phase,
# cpp_build_dir, cpp_require_build_tools and cpp_build and does nothing else, so
# the tier's configure, build and ctest all tee into ONE log (BUILD_LOG) and the
# tier writes the single summary line. Nothing here duplicates a phase cpp-tier
# also owns; there is exactly one copy of the cmake invocations, and it is here.
set -euo pipefail

# Sourced from cpp-tier.sh, which has already loaded lib.sh (and with it
# config.sh); only pull it in when run standalone, so we do not re-run its git
# probes on every source.
if [ -z "${REPO_ROOT:-}" ]; then
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
fi

# Where run_phase tees. A caller that sources this sets it to its own log so the
# whole tier lands in one file; left as /dev/null a stray source still runs.
: "${BUILD_LOG:=/dev/null}"

# Echo one phase's header and streamed output into BUILD_LOG, and return the
# command's status (not tee's -- PIPESTATUS[0]). Shared with cpp-tier.sh's
# ctest and clang-tidy phases, which is why it lives here rather than in either
# script.
run_phase() {
  local name=$1; shift
  echo "--- $name: $* ---" | tee -a "$BUILD_LOG"
  "$@" 2>&1 | tee -a "$BUILD_LOG"
  return "${PIPESTATUS[0]}"
}

# The binary directory a preset configures into. Every preset has one of its
# own -- `default` keeps build/, the rest use build-<preset>/ -- and a
# hardcoded build/ would point ctest and --tidy's compile_commands.json at
# whichever preset was built there last.
#
# Two sources, in order, because neither is available everywhere:
#
#   1. `cmake --preset <p> -N`, which prints "Binary directory: ..." -- but
#      only from CMake 4.x. On 3.x (Ubuntu 24.04 ships 3.28) it prints the
#      cache variables and nothing else, and the old sed silently produced an
#      empty string, fell back to build/, and sent ctest at a directory that
#      does not exist. That failure reads as "Failed to change working
#      directory", a long way from the version difference that caused it.
#   2. CMakePresets.json itself, read with python3, resolving `inherits` and
#      the ${sourceDir} macro the same way CMake does.
#
# Only if both are unavailable does it fall back to build/.
cpp_build_dir() {
  local preset=$1 d
  d=$(cmake --preset "$preset" -N 2>/dev/null |
    sed -n 's/^.*Binary directory: *//p' | head -1)
  if [ -z "$d" ] && command -v python3 >/dev/null 2>&1; then
    d=$(python3 - "$REPO_ROOT" "$preset" <<'PY' 2>/dev/null
import json, sys, os
root, want = sys.argv[1], sys.argv[2]
path = os.path.join(root, "CMakePresets.json")
presets = {}
try:
    with open(path) as f:
        for p in json.load(f).get("configurePresets", []):
            presets[p["name"]] = p
except Exception:
    sys.exit(1)
seen = set()
def binary_dir(name):
    if name in seen or name not in presets:
        return None
    seen.add(name)
    p = presets[name]
    if "binaryDir" in p:
        return p["binaryDir"]
    inherits = p.get("inherits", [])
    if isinstance(inherits, str):
        inherits = [inherits]
    for parent in inherits:
        d = binary_dir(parent)
        if d:
            return d
    return None
d = binary_dir(want)
if not d:
    sys.exit(1)
print(d.replace("${sourceDir}", root))
PY
    )
  fi
  [ -n "$d" ] && echo "$d" || echo "$REPO_ROOT/build"
}

# Fail early, and toward the fix, when the toolchain is not on this machine --
# the whole clang/CMake/nvcc stack lives in the image, not on a bare host.
cpp_require_build_tools() {
  local tool
  for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "build: $tool not found -- this needs the container:" >&2
      echo "  devtools/devcontainer.sh shell -c devtools/build.sh" >&2
      echo "(or install cmake + ninja and use --preset workstation)" >&2
      return 1
    }
  done
}

# Configure then build one preset, teeing both phases into BUILD_LOG. Returns
# the first failing phase's status; runs with the caller's errexit setting, so
# wrap the call in `set +e; cpp_build ... || rc=$?` as both entry points do.
#   cpp_build <preset> <fresh 0|1> <clean 0|1> <jobs-or-empty>
cpp_build() {
  local preset=$1 fresh=$2 clean=$3 jobs=$4

  local configure=(cmake --preset "$preset")
  [ "$fresh" -eq 1 ] && configure+=(--fresh)
  run_phase configure "${configure[@]}" || return $?

  local build=(cmake --build --preset "$preset")
  [ -n "$jobs" ] && build+=(-j "$jobs")
  [ "$clean" -eq 1 ] && build+=(--clean-first)
  run_phase build "${build[@]}" || return $?
}

# --- standalone entry point ------------------------------------------------
# Only when executed, never when sourced.

build_main() {
  local preset=$CMAKE_PRESET
  local fresh=0 clean=0 jobs=$BUILD_JOBS
  local report_dir=$SLOW_TIER_REPORTS

  while [ $# -gt 0 ]; do
    case "$1" in
      --preset)  preset=$2; shift 2;;
      --fresh)   fresh=1; shift;;
      --clean)   clean=1; shift;;
      -j|--jobs) jobs=$2; shift 2;;
      --report)  report_dir=$2; shift 2;;
      -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
      *)         echo "build: unknown flag: $1" >&2; exit 2;;
    esac
  done

  cpp_require_build_tools || exit 1

  case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
  mkdir -p "$report_dir"
  local stamp rev
  stamp=$(date +%Y%m%d-%H%M%S)
  rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  BUILD_LOG="$report_dir/build-$stamp-$rev-$preset.log"

  echo "build: $REPO_ROOT @ $rev  preset=$preset -> $BUILD_LOG"

  local rc=0
  set +e
  cpp_build "$preset" "$fresh" "$clean" "$jobs" || rc=$?
  set -e

  local verdict summary
  verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
  summary="build $verdict  rev=$rev  preset=$preset  stamp=$stamp"
  echo "$summary" | tee -a "$BUILD_LOG"
  echo "$summary" >> "$report_dir/summary.log"
  exit "$rc"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  build_main "$@"
fi
