# Devcontainers

| File | What it is |
|---|---|
| `cuda/devcontainer.json` | **The development container.** `../../Dockerfile.cuda` plus `--gpus all`: CUDA 12 devel, g++, CMake, Ninja, ccache, OpenBLAS + LAPACKE. The default, and the only one that can build anything under `src/libintx/gpu`. |
| `devcontainer.json` | The CPU variant: `../Dockerfile`, no device request. g++, CMake, Ninja, OpenBLAS/LAPACKE — enough to build and test the whole HOST half of libintx (boys, md2/md3/md4, the host K engine) on a machine with no GPU. Not enough for anything under `src/libintx/gpu`. |
| `seed-claude-config.sh` | Copies the agent's host config into the container once, at create time. |

## Which one you get

`DEVCONTAINER_CONFIG` in `devtools/config.sh`, defaulting to the CUDA one. A
relative path is resolved against the repo root, so it works from any cwd and in
any worktree. Override for one call:

```bash
DEVCONTAINER_CONFIG=.devcontainer/devcontainer.json devtools/devcontainer.sh up
```

`devtools/doctor.sh` prints the active config, and **FAILs** if it points at a
file that is not there — every `devcontainer.sh` command refuses until that is
fixed.

Note that the bare `devcontainer` CLI and any IDE "Reopen in Container" action
ignore all of this and pick `.devcontainer/devcontainer.json`, the CPU one.
Point the IDE at `cuda/devcontainer.json` explicitly, or you will get a
container with no nvcc, where the `default` preset fails at configure on
`CMAKE_CUDA_ARCHITECTURES=native` with a message about no CUDA compiler rather
than about which container you are in.

## Per-project edits

Both JSON files carry `// EDIT` markers on the four things a checkout at a new
path must change (there is no `bootstrap.sh` in this fork, so they are yours to
edit) — this list is
what it is doing, and what to check by hand if you ever rename again.

1. `"name"` — what the container is called.
2. The `PROJECT_NAME` build arg — stamped on the image as a label, which is how
   `devtools/worktree.sh gc` recognises this project's build images and leaves
   every other repo's alone.
3. The three `source=libintx-…` volume names — `worktree.sh rm` and `gc`
   reconstruct these from `PROJECT_NAME` in `devtools/config.sh`, so the two
   must agree. `devtools/doctor.sh` checks both files separately and names the
   one that drifted; while they disagree, every worktree you tear down leaks its
   whole set of volumes.
4. The `.git` bind mount — the absolute path of your main checkout's `.git`
   common dir. A worktree's `.git` is a *file* pointing into
   `.git/worktrees/<name>`, outside the workspace folder, and there is no
   devcontainer variable for it; without the mount, every git command inside a
   worktree container fails with "not a git repository", taking
   `postCreateCommand` — and with it the submodule init and the pre-commit hook
   install — down with it. When the workspace *is* the main repo this mounts
   `.git` onto itself, a harmless no-op. Delete the entry only if you will never
   use `devtools/worktree.sh`; a bind mount of a path that does not exist fails
   container creation.

JSON cannot source shell, which is why (2) and (3) are duplicated here rather
than read from `devtools/config.sh`.

## Rebuilding

```bash
devtools/devcontainer.sh rebuild
```

Use `rebuild`, not `up`, after editing either JSON file or a Dockerfile: `up`
reuses the running container, applies none of the change, and reports success
with the same container id. The workspace is a bind mount and the caches are
named volumes, so both survive a rebuild.

The CUDA image is large and its first build is long — the CUDA devel base image
is several GB before a single apt package lands. Subsequent rebuilds hit the
layer cache unless you changed something early in the file.

## Bounding it

`CPUSET` (host cores) and `CPUS` (a quota) in `devtools/config.sh` are applied by
`up` and `rebuild` via `docker update` — live, no restart. They bound everything
in the container, unlike `BUILD_JOBS`
(`cmake --build` jobs), which bound only their own runner. Two worktree
containers pinned to disjoint cores are the only way concurrent worktrees stop
stealing each other's timings.
