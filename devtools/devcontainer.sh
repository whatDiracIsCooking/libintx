#!/usr/bin/env bash
# Drive this worktree's devcontainer. There is usually no `devcontainer` binary
# on PATH, so every call goes through npx.
#
#   devtools/devcontainer.sh up          bring it up, reusing an existing one
#   devtools/devcontainer.sh rebuild     recreate from scratch
#   devtools/devcontainer.sh shell       open a bash shell inside
#   devtools/devcontainer.sh claude      open a shell inside and launch claude
#   devtools/devcontainer.sh test [...]  run the test suite inside
#   devtools/devcontainer.sh gpu         check the GPU is reachable inside
#   devtools/devcontainer.sh down        stop and remove the container
#
# CPUSET / CPUS (from config.sh, or set for one call) bound the container's
# share of the host on `up` and `rebuild`:
#
#   CPUSET=0-11 devtools/devcontainer.sh up      pin to host cores 0-11
#   CPUS=8      devtools/devcontainer.sh up      cap at 8 cores' worth
#
# Unlike BUILD_JOBS, which only bounds the compiler, these bound everything the
# container runs -- compiles started from a `shell` included.
#
# WHICH container: DEVCONTAINER_CONFIG in config.sh, defaulting to the CUDA
# variant -- the one that can actually build the C++ tree. Override per call:
#
#   DEVCONTAINER_CONFIG=.devcontainer/devcontainer.json devtools/devcontainer.sh up
#
# Containers key on the workspace folder path, so each worktree gets its own
# container and its own named volumes. The script resolves the workspace from
# its own location, so it does the right thing whichever checkout you call it
# from, and from any cwd.
#
# Project-specific values (the test command, the ctest preset) come from
# devtools/config.sh -- edit that, not this.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKSPACE=$REPO_ROOT
CLI=(npx -y @devcontainers/cli)

# Which devcontainer.json to drive. Defaults in config.sh to the CUDA variant
# -- the one with the C++ toolchain -- and an environment variable still wins
# for one call. A RELATIVE path is resolved against the repo root here, not the
# cwd: the CLI resolves --config relative to where you invoked it, so a bare
# `.devcontainer/cuda/devcontainer.json` would otherwise only work from the
# root. Empty means "let the CLI find it", i.e. .devcontainer/devcontainer.json.
DC_CONFIG=()
if [ -n "${DEVCONTAINER_CONFIG:-}" ]; then
  case "$DEVCONTAINER_CONFIG" in
    /*) dc_config_path=$DEVCONTAINER_CONFIG ;;
    *) dc_config_path=$REPO_ROOT/$DEVCONTAINER_CONFIG ;;
  esac
  if [ ! -f "$dc_config_path" ]; then
    echo "error: DEVCONTAINER_CONFIG does not exist: $dc_config_path" >&2
    exit 1
  fi
  DC_CONFIG=(--config "$dc_config_path")
fi

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# This workspace's container ids, one per line. `--all` includes stopped ones,
# which is what `down` wants -- an exited container from a crashed run keeps
# the label. The bare form is the running container, the only thing `docker
# update` can act on. Empty when nothing matches.
container_ids() {
  local ps=(docker ps -q)
  [ "${1:-}" = --all ] && ps=(docker ps -aq)
  "${ps[@]}" --filter "label=devcontainer.local_folder=$WORKSPACE"
}

# Apply CPUSET/CPUS to the running container with `docker update` -- live, no
# restart, so it also works as a way to re-pin a container that is already up.
# A no-op when neither is set, and a warning rather than an error when nothing
# is running: the limits are a refinement of `up`, not a precondition for it.
#
# `docker update` changes only the limits it is given, so a later call with
# just CPUSET leaves an earlier CPUS quota in place, and unsetting both here
# clears nothing -- it just stops applying. To get back to an unbounded
# container, `rebuild` it (a fresh container starts with no limits).
apply_cpu_limits() {
  [ -z "${CPUSET:-}" ] && [ -z "${CPUS:-}" ] && return 0
  local id flags=()
  # `|| true`: docker may be absent or its daemon down, and a missing CPU
  # pin is not a reason to fail a command that has already done its work.
  id=$(container_ids | head -1 || true)
  if [ -z "$id" ]; then
    echo "cpu-limit: no running container for $WORKSPACE (skipped)" >&2
    return 0
  fi
  [ -n "${CPUSET:-}" ] && flags+=(--cpuset-cpus "$CPUSET")
  [ -n "${CPUS:-}" ] && flags+=(--cpus "$CPUS")
  docker update "${flags[@]}" "$id" >/dev/null
  echo "cpu-limit: ${CPUSET:+cpuset-cpus=$CPUSET }${CPUS:+cpus=$CPUS }-> $id" >&2
}

# Is the GPU actually reachable inside the running container? Probes the CUDA
# driver the same way testkit.cuda_driver.HAVE_GPU does (cuInit != 0 -> no
# device). Prints "<n>" device count on success, nothing on failure; returns
# non-zero when the container has GPU nodes but the driver can't use them.
# No-op success when there are no GPU nodes at all (a host without --gpus), so
# non-GPU setups are unaffected.
gpu_device_count() {
  local id; id=$(container_ids | head -1 || true)
  [ -z "$id" ] && return 0
  docker exec "$id" sh -c 'ls /dev/nvidia[0-9]* >/dev/null 2>&1' || return 0
  docker exec "$id" python3 -c 'import ctypes,sys
lib=ctypes.CDLL("libcuda.so.1")
if lib.cuInit(0)!=0: sys.exit(1)
c=ctypes.c_int(0); lib.cuDeviceGetCount(ctypes.byref(c)); print(c.value)' 2>/dev/null
}

# Warn (loudly, to stderr) when the running container has GPU nodes but the CUDA
# driver reports no device -- the mid-life device-cgroup revocation a host driver
# reload or `systemctl daemon-reload` causes (see doctor.sh / CLAUDE.md). A plain
# `up` reuses that dead container; only a recreate restores it. Called after
# `up`/`rebuild` so the fix is obvious instead of surfacing as mystery skips.
warn_if_gpu_revoked() {
  local id; id=$(container_ids | head -1 || true)
  [ -z "$id" ] && return 0
  docker exec "$id" sh -c 'ls /dev/nvidia[0-9]* >/dev/null 2>&1' || return 0
  gpu_device_count >/dev/null && return 0
  echo "devcontainer: WARNING -- GPU nodes are present but the CUDA driver reports no device." >&2
  echo "devcontainer:   the container's --gpus device-cgroup grant was revoked (host driver" >&2
  echo "devcontainer:   reload or a systemd daemon-reload). Recover by recreating it:" >&2
  echo "devcontainer:       devtools/devcontainer.sh rebuild" >&2
  echo "devcontainer:   (a plain 'up' reuses this container and will NOT restore the GPU.)" >&2
}

# The .git bind mount in devcontainer.json is an absolute host path that
# a checkout at a new path must fill in. Until then it is a literal placeholder, and docker's
# --mount type=bind refuses a source that does not exist. What the devcontainer
# CLI does with that refusal is the problem: it prints a node stack trace and
# the ENTIRE failing `docker run` command line, which by then has -e
# GH_TOKEN=<your token> in it. So the first thing a fresh checkout
# does is put a credential on the terminal, and into whatever issue the error
# gets pasted into.
#
# Catch it here instead, before the CLI is ever invoked.
placeholder_git_mount='/absolute/path/to/your-root/main'
check_git_mount() {
  grep -q "$placeholder_git_mount" "$dc_config_path" 2>/dev/null || return 0
  echo "error: $dc_config_path still has the placeholder .git bind mount." >&2
  echo "       docker cannot bind a source that does not exist, so the" >&2
  echo "       container will not start. Fix it once:" >&2
  echo >&2
  echo "         edit the .git bind mount in both .devcontainer/*.json" >&2
  echo >&2
  echo "       or point it by hand at:" >&2
  echo "         $(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo '<this repo>/.git')" >&2
  exit 1
}

# The CLI prints the whole `docker run` invocation on failure, GH_TOKEN and
# all. Everything below routes through this so a secret cannot reach the
# terminal; the exit status is the CLI's, not sed's.
run_cli() {
  local rc=0
  "${CLI[@]}" "$@" > >(sed -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g') \
    2> >(sed -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g' >&2) || rc=$?
  # Let the redirections drain before the caller reads the next line.
  wait
  return "$rc"
}

case "${1:-}" in
  up)
    shift
    check_git_mount
    run_cli up --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" "$@"
    apply_cpu_limits
    # `up` may have reused a container whose GPU grant was revoked mid-life; a
    # reuse cannot restore it, so flag it here and point at `rebuild`.
    warn_if_gpu_revoked
    ;;
  rebuild)
    # Needed after editing devcontainer.json or the Dockerfile: a plain `up`
    # reuses the running container and silently applies none of it, while
    # reporting success and the same container id. The workspace is a bind
    # mount and the caches are named volumes, so both survive; anything in the
    # container's own filesystem does not.
    shift
    check_git_mount
    run_cli up --workspace-folder "$WORKSPACE" \
      --remove-existing-container "${DC_CONFIG[@]}" "$@"
    apply_cpu_limits
    # A fresh container should have the GPU; if this still warns, the problem is
    # host-side (driver/cgroup), not a stale container -- see doctor.sh.
    warn_if_gpu_revoked
    ;;
  shell)
    shift
    # An INTERACTIVE shell cannot go through the CLI. Two things break it, and
    # both are in this script's own plumbing: `devcontainer exec` allocates no
    # pty (`shell -c tty` prints "not a tty"), and run_cli pipes both streams
    # through sed, so bash sees a non-terminal stderr. The result is a shell
    # that prints no prompt and block-buffers its output while running your
    # keystrokes invisibly -- indistinguishable from a hang. So the
    # no-argument form talks to `docker exec` directly.
    #
    # Skipping run_cli costs nothing here: the GH_TOKEN redaction exists for
    # the CLI's failure output, which prints the whole `docker run` line, and
    # `docker exec` prints no such thing. `-w "$WORKSPACE"` is required -- the
    # image WORKDIR is /workspace, which is not this checkout -- and `-t` has
    # to be conditional, or a non-terminal stdin dies with "the input device
    # is not a TTY". No `-u`: docker exec already lands as `ubuntu` here.
    if [ "$#" -eq 0 ]; then
      id=$(container_ids | head -1)
      if [ -z "$id" ]; then
        echo "no container up for $WORKSPACE -- run: $0 up" >&2
        exit 1
      fi
      tty_flags=(-i)
      [ -t 0 ] && tty_flags+=(-t)
      exec docker exec "${tty_flags[@]}" -w "$WORKSPACE" "$id" bash -l
    fi
    # With arguments there is no pty to want, and the CLI path keeps its
    # redaction.
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" bash "$@"
    ;;
  claude)
    shift
    # `shell` with no argument, but launching claude in it -- so the same
    # `docker exec` path and for the same reasons (see `shell` above): the CLI
    # allocates no pty and run_cli block-buffers stderr through sed, either of
    # which turns an interactive claude into an invisible hang. `bash -lc`, not
    # a bare `claude`, so the login profile that puts claude on PATH is sourced
    # first -- a plain `docker exec ... claude` lands with the image's minimal
    # env and "claude: command not found". Trailing "$@" forwards any extra
    # args (`devcontainer.sh claude --continue`) straight to claude.
    id=$(container_ids | head -1)
    if [ -z "$id" ]; then
      echo "no container up for $WORKSPACE -- run: $0 up" >&2
      exit 1
    fi
    tty_flags=(-i)
    [ -t 0 ] && tty_flags+=(-t)
    exec docker exec "${tty_flags[@]}" -w "$WORKSPACE" "$id" \
      bash -lc 'exec claude "$@"' bash "$@"
    ;;
  test)
    shift
    # TEST_CMD may carry its own arguments, so split it as a command line
    # rather than on whitespace alone.
    config_args "$TEST_CMD"
    cmd=("${CONFIG_ARGS[@]}")
    # ctest, not pytest: --output-on-failure because a bare "Failed" line tells
    # you nothing, and --preset so it reads the tree the presets configured.
    # Anything the caller passes is appended and wins.
    [ -n "${CTEST_PRESET:-}" ] && cmd+=(--preset "$CTEST_PRESET")
    cmd+=(--output-on-failure)
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${cmd[@]}" "$@"
    ;;
  gpu)
    # On-demand version of the check `up`/`rebuild` run automatically: is the
    # GPU reachable inside the running container right now?
    id=$(container_ids | head -1 || true)
    if [ -z "$id" ]; then
      echo "no running container for $WORKSPACE (bring it up first)" >&2
      exit 1
    fi
    if ! docker exec "$id" sh -c 'ls /dev/nvidia[0-9]* >/dev/null 2>&1'; then
      echo "gpu: no /dev/nvidia* in the container (started without --gpus, or host has no GPU)"
      exit 0
    fi
    if n=$(gpu_device_count) && [ -n "$n" ]; then
      echo "gpu: reachable inside container $id -- $n device(s)"
    else
      warn_if_gpu_revoked
      exit 1
    fi
    ;;
  down)
    ids=$(container_ids --all)
    if [ -z "$ids" ]; then
      echo "no container for $WORKSPACE"
    else
      # One id per line, and there can be more than one, so loop: a single
      # quoted "$ids" would reach docker as one newline-joined name and fail
      # with "No such container" while leaving both behind.
      while read -r id; do
        [ -n "$id" ] || continue
        docker rm -f "$id"
      done <<<"$ids"
    fi
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
