# docker/ — the batch path

One-shot runs that bring their own container, tee everything to `.log/`, and
exit. The other front end is the devcontainer (`.devcontainer/cuda/` driven by
`devtools/devcontainer.sh`), which is the interactive and agent-facing one.

**Both run the same image, `../Dockerfile.cuda`**, so a result here and a result
in there mean the same thing. Neither is deprecated: batch when you want to read
a result and throw the container away, devcontainer when you want to stay
inside.

## Build the image

From the **repo root**, not from here:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.cuda -t libintx:latest .
```

To widen the device architecture baked into the image:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.cuda \
  --build-arg CUDA_ARCH="80;86;90" -t libintx:latest .
```

## Run

`HOST_UID`/`HOST_GID` are **required** (`:?` in `compose.yaml`, not a default):
a default of 1000 on a host where you are not uid 1000 writes root-owned build
output into the bind mount, and that failure surfaces a long way from here.

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
dc() { docker compose -f docker/compose.yaml "$@"; }

dc run --rm build              # configure + build only
dc run --rm test               # configure + build + ctest
dc run --rm asan               # the host tree under AddressSanitizer
dc run --rm compute-sanitizer  # the GPU tests under compute-sanitizer
```

## Services

| Service | What it does |
|---|---|
| `build` | `cmake --preset $BUILD_PRESET -B $BUILD_DIR`, then `cmake --build --target all all.tests`. |
| `test` | The same, then `ctest --output-on-failure`. |
| `asan` | The `asan` preset (CPU only, `LIBINTX_SANITIZE_ADDRESS=ON`) end to end. |
| `compute-sanitizer` | Builds with `$BUILD_PRESET`, then runs each test matching `$SANITIZER_TESTS` under `compute-sanitizer --tool=$SANITIZER_TOOL`. |

## Environment

| Variable | Meaning |
|---|---|
| `HOST_UID` / `HOST_GID` | Required. The uid:gid the container runs as. |
| `LIBINTX_IMAGE` | Image to use (default `libintx:latest`). |
| `BUILD_PRESET` | Configure preset (default `default`). |
| `BUILD_DIR` | Compose's own binary directory (default `build-compose`). |
| `CTEST_ARGS` | Extra arguments for `ctest`, e.g. `-R kengine`. |
| `RECONFIGURE` / `CLEAN` / `REBUILD` | `--fresh` / `--clean-first` / both, on `build`. |
| `SANITIZER_TOOL` | `memcheck` (default), `racecheck`, `initcheck`, `synccheck`. |
| `SANITIZER_TESTS` | ctest `-R` selection for the sanitizer sweep (default `gpu`). |

`docker/env.example` is a starting point; copy it to `docker/.env`.

## Two things that will bite you

**Compose and the devcontainer cannot share a build directory.** The
devcontainer mounts the workspace at its *host* path while compose mounts it at
`/workspace`, and a CMake cache records absolute source and binary directories.
A tree configured by one is rejected by the other with

```
The current CMakeCache.txt directory ... is different than the directory ...
where CMakeCache.txt was created
```

Caches are not relocatable, so compose configures into its own `build-compose*/`
via `-B $BUILD_DIR` instead of the preset's `binaryDir`. That is what makes
`devcontainer.sh rebuild` followed by `docker compose run --rm test` work.

**`--target all all.tests`, always.** libintx declares every test executable
`EXCLUDE_FROM_ALL` and gathers them under the `all.tests` custom target. A plain
`cmake --build` produces the libraries and no test binaries, and `ctest` then
reports "No tests were found", which reads like a configuration problem rather
than a missing target. Every build line in `compose.yaml` names both.

## Logs

Everything is teed into `.log/` at the repo root, timestamped:

```
.log/configure.{ts}.out.txt  / .log/configure.{ts}.err.txt
.log/build.{ts}.out.txt      / .log/build.{ts}.err.txt
.log/ctest.{ts}.out.txt      / .log/ctest.{ts}.err.txt
.log/compute-sanitizer.{ts}.txt
```

`devtools/build.sh` and `devtools/cpp-tier.sh` write elsewhere
(`.slow-tier-reports/`) and are not driven by this file at all.
