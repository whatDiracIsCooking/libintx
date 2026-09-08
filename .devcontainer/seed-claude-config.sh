#!/usr/bin/env bash
# Seed this container's Claude Code config from the host's, once, at create.
#
# $HOME/.claude.json is not carried by the ~/.claude mount -- it sits next to
# that directory, not inside it -- so without this the container comes up as a
# fresh install and asks you to log in. The host copy is bind-mounted read-only
# at $SRC and each container takes its own mutable copy: that file is one
# monolithic JSON (every project's allowedTools and trust state in a single
# `projects` map) rewritten whole on each change, so sharing it read-write
# between the host and several worktree containers is last-writer-wins.
set -uo pipefail

SRC=/home/ubuntu/.claude.json.host
DST=/home/ubuntu/.claude.json

if [ -e "$DST" ]; then
  echo "seed-claude-config: $DST already present, left as is"
  exit 0
fi
if [ ! -r "$SRC" ]; then
  echo "seed-claude-config: no readable $SRC, skipping (expect a login prompt)"
  exit 0
fi

cp "$SRC" "$DST" || exit 0

# Drop the host's installation bookkeeping. It records a native install at
# ~/.local/bin/claude, a path that does not exist in here -- claude comes from
# the devcontainer feature, at /usr/bin/claude -- and the startup health check
# turns that into
#   claude command at /home/ubuntu/.local/bin/claude missing or broken
# With installMethod absent it derives to "unknown" and that check does not run.
#
# autoUpdates is pinned false rather than dropped: it defaults to *true* when
# absent (`e.autoUpdates ?? true`), and the feature installs claude npm-global
# under root-owned /usr/lib/node_modules, so an enabled auto-update fails with
#   Auto-update failed: no write permission to npm prefix
# The claude in here is provisioned by the image, like every other tool in it;
# it should not be updating itself behind the image's back.
#
# Everything that matters for staying logged in (oauthAccount,
# hasCompletedOnboarding) is untouched.
python3 - "$DST" <<'PY'
import json, sys

path = sys.argv[1]
with open(path) as fh:
    cfg = json.load(fh)
for key in ("installMethod", "autoUpdatesProtectedForNative"):
    cfg.pop(key, None)
cfg["autoUpdates"] = False
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
PY

echo "seed-claude-config: seeded from $SRC"
