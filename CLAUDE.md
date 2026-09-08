# CLAUDE.md

## What this is

LibintX evaluates molecular integrals of many-body operators over Gaussian AOs,
with the accelerated (CUDA/HIP) path as the point and a CPU path that works
everywhere. This checkout is a **fork** of the Valeev Group's repository; the
public mirror is <https://github.com/ValeevGroup/libintx>. Upstream calls it
EXPERIMENTAL until 1.0 and means it — APIs move.

The tree is C++17/20 with ordinary headers. No modules, no `import std`, no
FetchContent: the heavy dependencies are **vendored in-tree** (`include/eigen3`,
`include/cute` + `include/cutlass`, `python/pybind11`, `tests/doctest.h`), so a
clone is self-contained and there is nothing to `submodule update`.

## Layout

| Path | What |
|---|---|
| `src/libintx/` | `shell.h`, `orbital.h`, `array.h`, `math.h`, `tensor.h`, `simd.h` — the value types everything else is written against. Plus `blas.cc` and the two engine interfaces, `jengine.h` and `kengine.h`. |
| `src/libintx/boys/` | The Boys function: Chebyshev interpolation + asymptotic tail. Host and (`gpu/chebyshev.h`) device. |
| `src/libintx/ao/md/` | The **host** McMurchie–Davidson engines: `IntegralEngine<2>` (overlap/kinetic/nuclear), `<3>`, `<4>`, the Hermite machinery (`hermite.h`, `r1/`), a plain reference (`reference.h`), and the host K engine (`kengine.cc`). |
| `src/libintx/gpu/` | The device half. `api/` wraps CUDA/HIP behind one `gpu::` namespace; `md/` is the device `IntegralEngine<3>`/`<4>` plus the device K engine; `jengine/md/` is the DF J engine. |
| `src/libintx/kengine/md/driver.h` | The K build shared by the host and device engines: shell-pair binning, screening, the eight-fold digest. |
| `tests/` | doctest executables plus `test.h` (random bases, Eigen tensors, `ReferenceValue`). |
| `python/` | pybind11 bindings (`-DLIBINTX_PYTHON=ON`) and `pywfn`, a small pure-Python molecule/basis helper with JSON basis sets and `.xyz` geometries. |
| `devtools/`, `.devcontainer/`, `docker/` | Container and build tooling — see below. |

## Building

```bash
cmake --preset workstation && cmake --build --preset workstation   # CPU only
ctest --preset workstation --output-on-failure

devtools/devcontainer.sh shell -c devtools/cpp-tier.sh             # the GPU tree
```

`CMakePresets.json` is a fork addition (upstream drives raw `cmake -B build`
lines, and its CI still does). Presets: `default` (CUDA, native arch, `build/`),
`workstation` (CPU only), `debug`, `asan`, `ampere`/`hopper`/`portable`, `hip`.
Every preset has its own `binaryDir` — a CMake cache records absolute source and
binary directories, so two presets sharing one directory silently reconfigure it
back and forth on every switch.

Four things that will cost you an afternoon otherwise:

- **Test binaries are `EXCLUDE_FROM_ALL`.** `tests/CMakeLists.txt` gathers them
  under the `all.tests` custom target, so a plain `cmake --build <dir>` produces
  the libraries and **no tests**, and ctest then says `No tests were found` —
  which reads like a broken configuration. Every build preset here names
  `all all.tests`; a bare `cmake --build` does not.
- **`CMAKE_CUDA_ARCHITECTURES=native` queries a live device at *configure*
  time.** No GPU, no configure — even for a change that touches no CUDA. That
  is the `default` preset; use `workstation` on a bare host, or a pinned
  architecture preset to cross-build.
- **`LIBINTX_MAX_L` is a configure-time decision that fixes what the tests can
  cover.** It defaults to 3 and instantiates one translation unit per
  (bra,ket) angular-momentum pair — `(2*LMAX+1)^2` of them for md4 alone. Lower
  it to iterate (`-DLIBINTX_MAX_L=2` roughly halves the build); a green run at
  `MAX_L=2` has not compiled or executed a single f-shell kernel. `LIBINTX_MAX_X`
  (auxiliary, defaults to `MAX_L+1`), `LIBINTX_MAX_K` (primitives, 10) and
  `LIBINTX_GPU_MAX_SHMEM` are the other knobs; they land in
  `libintx/config.h` as `LMAX`, `XMAX`, `KMAX`.
- **`libintx.blas` needs CBLAS *and* LAPACKE, which are separate packages on
  Linux.** `blas.cc` includes `<cblas.h>` and calls `LAPACKE_dsygvd`, but
  `find_package(LAPACK)` locates the Fortran library. Without `libopenblas-dev`
  the tree fails to compile; without `liblapacke-dev` it configures, compiles,
  and *then* fails at link on an undefined `LAPACKE_dsygvd`. This fork adds a
  `find_library(lapacke)` to `src/libintx/CMakeLists.txt` so the second half is
  wired automatically where the package exists (it is a no-op under Accelerate
  and MKL, which already carry the C interface). Both images install both.

## The engines

Three layers, and it is worth knowing which one you are in.

**`IntegralEngine<N>`** — the integral primitives. `ao::IntegralEngine<N>` in
`src/libintx/ao/engine.h` is the abstract base; `md::IntegralEngine<N>` is the
host implementation and `gpu::md::IntegralEngine<N>` the device one. They share
one `compute` contract:

```cpp
compute(Operator, const std::vector<Index2> &bra,
        const std::vector<Index2> &ket,
        BraKet<const double*> norms,
        double *V, const std::array<size_t,2> &dims);
```

`V` is a column-major `(M*NA*NB) x (NC*ND*N)` matrix for `M` bra pairs by `N`
ket pairs, so element `(ij, na, nb, nc, nd, kl)` is at
`ij + M*(na + NA*(nb + NB*(nc + NC*(nd + ND*kl))))`. `norms` are per-pair
Schwarz bounds, or `{}` for none.

**A batch is one bin.** Every pair handed to a single `compute` call must agree
on angular momentum, the solid-harmonic flag *and* the contraction degree
`K_ab = nprim(a)*nprim(b)`: `ao/md/basis.cc` packs the batch as one flat array
of `K_ab` primitive pairs and `libintx_assert(a.K*b.K == K)` aborts otherwise.
Nothing pads. That three-way binning is what `kengine::md::make_pair_classes`
exists to do.

**`JEngine`** (`src/libintx/jengine.h`) — the density-fitted Coulomb engine.
`gpu::make_jengine(basis, df_basis, V_linv, screening)`; GPU only, no host
implementation. It reads D and writes J through shell-block tile callbacks so a
distributed caller never has to materialise either matrix.

**`KEngine`** (`src/libintx/kengine.h`) — the exchange engine, added in this
fork.

```
K[mu,nu] = sum_{lambda,sigma} (mu lambda | nu sigma) D[lambda,sigma]
```

Same tile-callback shape as `JEngine`, deliberately, so a caller already
driving J drives K the same way. Unlike J it is *conventional*, not
density-fitted: four centres, no auxiliary basis, no `V^-1`.
`F = H + J - K/2` for a closed shell, with J and K contracted against the same
`D = 2 C_occ C_occ^T`.

Two implementations, one algorithm:

| | Host | Device |
|---|---|---|
| Factory | `libintx::md::make_kengine` | `libintx::gpu::make_kengine` |
| Header | `src/libintx/ao/md/kengine.h` | `src/libintx/gpu/kengine.h` |
| Integrals | `md::IntegralEngine<4>` | `gpu::md::IntegralEngine<4>` |
| Built into | `libintx.md4` | `libintx.gpu.md4` |

Both are thin: they supply an engine type and a buffer, and
`src/libintx/kengine/md/driver.h` does the rest. The device path produces the
`(ab|cd)` batch into host-registered memory and digests on the host — the
straightforward split, not the final one; a device-side digest would not change
the interface.

### Two things about the K digest specifically

**The eight-fold orbit is enumerated, not weighted.** `K[mu,nu]` sums over
*every* index tuple, so a unique quartet contributes once per distinct member of
its permutation orbit. Carrying per-case degeneracy factors is where a K digest
normally goes wrong, because `a==b`, `c==d` and `(ab)==(cd)` each collapse a
*different* subset. `digest()` instead enumerates the eight permutations,
deduplicates on the permuted shell tuple, and gives every survivor weight one —
which is why it carries the permutation index around rather than just the
permuted shells: the index says where in `V` that member's value lives.

If a K result is wrong by a small integer factor in a systematic pattern, it is
this, not the integrals. The integrals have their own test
(`libintx.md4.test` against `libintx::md::reference`).

**Contraction coefficients must be primitive-normalized.** A basis-set library's
coefficients are defined against normalized primitives. For a contracted shell
that is *not* an overall per-basis-function scale — the primitives carry
different exponents — so feeding raw coefficients through is a physically
different basis, not a rescaled one, and the error survives all the way to the
SCF energy rather than showing up as an obviously broken integral.
`libintx::make_basis(..., normalize=true)` applies `gto::normalized` and that is
the convention the K engine assumes; if you build a `Basis` by hand with
`normalize=false`, K, the one-electron integrals and J must all agree on that
choice. (This is the same trap that produced a ~6.6 Ha error in the project this
engine was ported from, where it was first misdiagnosed as a d-shell kernel bug.)

### Screening

`KEngine::Screening` is `JEngine::Screening` minus `max1()` — a conventional K
build has no auxiliary bra, so only pair bounds mean anything. `max2(i,j)` is
the Schwarz bound `sqrt(max |(ij|ij)|)`; a quartet is dropped when
`max2(i,j)*max2(k,l)*max|D|` fails `skip()`.

`make_schwarz_screening(basis, threshold)` builds one (there is a host and a
device version; the result is plain host data either way, so they are
interchangeable). It evaluates one diagonal quartet per call — a batch would
have to be square in the shell pairs to keep its diagonal, which is `|pairs|`
times the integral work for the same `|pairs|` numbers. It runs once per
geometry, not once per SCF iteration.

Note the density factor for K is the largest of the **cross** blocks
(`D[ac]`, `D[ad]`, `D[bc]`, `D[bd]`, …), not `D[ab]`: K couples the bra indices
to the ket indices, which is exactly what makes a J screen too loose for it.

## Running tests

One suite: doctest executables under `tests/`, registered with `add_test()`,
driven by `ctest`. (`python/tests/` exists but needs the pybind11 bindings,
which are off by default.)

```bash
ctest --preset workstation --output-on-failure     # everything
ctest --preset workstation -R kengine              # one
ctest --preset workstation -N                      # list without running
devtools/cpp-tier.sh                               # configure + build + ctest, logged
```

Only `default`, `workstation`, `debug` and `asan` have **test** presets. With
any other configure preset, `cpp-tier.sh` builds and then reports nothing to
ctest — that is not a pass.

**A green run is bounded by the `LIBINTX_MAX_L` it was configured with.**
`test::enabled(...)` in `tests/test.h` and the `if (LMAX < 3) return;` guard in
the K engine's `f` case skip silently above it. Say which `MAX_L` a run was, and
say whether the GPU half was in the build at all — on a machine with no card it
was not.

**One test fails on `main`, before this fork's changes.** At
`LIBINTX_MAX_L=3` on x86-64 with the default Release flags (`-Ofast
-ffast-math -march=native`, AVX512 here), `libintx.md2.test` reports 2 failed
assertions out of 1,370,880 — both `libintx.md2.Nuclear` at `(3|3)`, e.g.
`-1.4718871549` against a reference of `-1.4718871542`. That is a relative
error of ~5e-10 against a `1e-10` tolerance: a precision limit in the f-shell
nuclear-attraction path under fast-math, not a wrong answer. Verified identical
on a pristine `origin/main` checkout, so do not attribute it to a change of
yours; upstream has hit the same class of thing before (`7f1efe5`, "Lower
precision for boys unit test if Apple"). It does not reproduce at
`LIBINTX_MAX_L=2`, which is why CI is green.

The K engine tests:

- `tests/libintx.kengine.test.cc` — the host engine against a brute-force sum
  over *every* shell quartet with no permutational symmetry, built from
  `libintx::md::reference`. It shares no code with the engine, so agreement is
  evidence rather than tautology. Cases: ss / sp / spd / f, contracted shells of
  differing depth, the screened path, K's symmetry, and `AllSum`.
- `tests/libintx.gpu.kengine.test.cc` — the device engine against the host one,
  plus the two Schwarz passes against each other. What it pins down is the
  device engine and the pinned-memory path; the driver itself is pinned down by
  the host test above.

## devtools/

Ported from a sibling project and driven entirely by **`devtools/config.sh`** —
the project name, which devcontainer, which presets, the job counts, the CPU
bounds, the smoke selection, the doctor lists. Edit that file, not the scripts.

| Script | What it does |
|---|---|
| `config.sh` | The settings above. Sourced by everything else. |
| `lib.sh` | Shared helpers (config splitting, tool discovery). |
| `doctor.sh` | Report what is degraded here. Non-zero only on a FAIL. |
| `devcontainer.sh` | `up` / `rebuild` / `shell` / `test` / `down` for this worktree's container. |
| `worktree.sh` | `add` / `rm` / `sync` / `gc` / `list` for sibling container-backed worktrees. |
| `build.sh` | Configure + build, logged. Also the one copy of the two cmake phases, which `cpp-tier.sh` and `cpp-smoke.sh` source. |
| `cpp-tier.sh` | `build.sh`'s configure + build, then ctest, logged. |
| `cpp-smoke.sh` | The cheap gate: incremental build + `SMOKE_TESTS`, plus `SMOKE_GPU_TESTS` when a device node exists. |
| `hooks/protect-main.py` | PreToolUse guard: blocks agent edits to the primary checkout. |

**The upstream set's pytest-shaped scripts are deliberately absent** —
`prepush-tests.sh`, `slow-tier.sh`, `cmake-lint.sh` — along with the
`pyproject.toml`/`uv.lock` machinery, because libintx has no Python package, no
venv and no pre-commit config. Do not reintroduce them as empty shells; the gate
here is the C++ one.

Two duplications are unavoidable and worth knowing about:

- **`PROJECT_NAME` is spelled out again in both `.devcontainer/*.json`** (and in
  their `PROJECT_NAME` build args), because JSON cannot source shell. If they
  drift, `worktree.sh rm` and `gc` stop recognising this project's volumes and
  silently leak them. `doctor.sh` checks both files.
- **The `.git` bind mount in both `devcontainer.json` files is an absolute host
  path**, currently the placeholder `/CHANGE/ME/libintx/main/.git`. A worktree's
  `.git` is a *file* pointing into the main checkout's `.git/worktrees/<name>`,
  which is not under the workspace folder, so without that mount every git
  command inside a worktree container fails with "not a git repository". There
  is no devcontainer variable for "the repo behind this worktree" — edit it for
  your checkout.

`protect-main.py` is enabled by default via `.claude/settings.json`: an agent's
`Write`/`Edit` to the primary checkout is denied, with `.claude/worktrees/`
carved out. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing
main.

## Containers: one image, two front ends

`Dockerfile.cuda` is the image — CUDA 12 devel, g++, CMake, Ninja, ccache,
OpenBLAS + LAPACKE, gh. It is built from the **repo root**, not from `docker/`.

- **`docker/compose.yaml`** — batch. `build` / `test` / `asan` /
  `compute-sanitizer`, one-shot runs that tee to `.log/` and exit. Requires
  `HOST_UID`/`HOST_GID` in the environment (`:?`, not a default, so a forgotten
  export fails loudly instead of writing root-owned files).
- **`.devcontainer/cuda/`** driven by `devtools/devcontainer.sh` — interactive,
  per worktree, with its own container, volumes and CPU bounds.

**They do not share a build directory, and cannot.** The devcontainer mounts the
workspace at its *host* path; compose mounts it at `/workspace`, and a CMake
cache records absolute source and binary directories. Compose configures with
`-B $BUILD_DIR` into its own `build-compose*/` for exactly this reason.

The root `Dockerfile` is the CPU image. Unlike its counterpart in the project
this tooling came from, it is **not** crippled: it builds and tests the entire
host half of libintx. What it cannot do is anything under `src/libintx/gpu`.

Use `rebuild`, not `up`, after editing `devcontainer.json` or a `Dockerfile`:
`up` reuses the running container, applies none of the change, and reports
success with the same container id.

**Three "how parallel" knobs, and only two bound a container.** `BUILD_JOBS` is
`cmake --build`'s and reaches nothing else — a compile started from a `shell`
still takes the whole box. `CPUSET` (host cores) and `CPUS` (a quota) are
applied by `devcontainer.sh` on `up`/`rebuild` via `docker update`, live. They
belong to the container, which is keyed on the workspace folder, so they are per
worktree.

## What is gated, and what is not

- **`git push`** — nothing automatic. `devtools/cpp-smoke.sh` is written to be a
  pre-push hook and is runnable by hand, but this fork ships no
  `.pre-commit-config.yaml`, so wire it up yourself if you want it.
- **CI** (`.github/workflows/ci.yml`) runs two jobs, both CPU and both at
  `-DLIBINTX_MAX_L=2`: the upstream macOS/clang Debug job, and a Linux/g++
  Release job this fork adds to gate the host K engine. **Neither builds any
  CUDA**, so a change under `src/libintx/gpu/` is checked by nothing
  server-side. `devtools/cpp-tier.sh` with the `default` preset, on a machine
  with a card, is the real gate — know that rather than discover it.
- No linter and no formatter runs anywhere. `clang-format` and `cmake-format`
  are installed in the images and run by hand if at all.

## Fork changes, and where they touch upstream

Keep this list current; it is what a rebase onto upstream has to reconcile.

**New files** — `src/libintx/kengine.h`, `src/libintx/kengine/md/driver.h`,
`src/libintx/ao/md/kengine.{h,cc}`, `src/libintx/gpu/kengine.h`,
`src/libintx/gpu/md/kengine.cc`, `tests/libintx.{,gpu.}kengine.test.cc`,
`CMakePresets.json`, `CLAUDE.md`, `.gitignore`, `.editorconfig`, `devtools/`,
`.claude/`, `.devcontainer/`, `docker/`, `Dockerfile`, `Dockerfile.cuda`,
`.dockerignore`.

**Edits to upstream files:**

- `src/libintx/gpu/md/engine.h`, `md3.cc`, `md4.cc` — **a fix, not a feature.**
  `ao::IntegralEngine<N>::compute` gained a `BraKet<const double*> norms`
  parameter when the host MD engines were rewritten (upstream `99f0bd4`), but
  the device engines were not updated: they still declared the old five-argument
  signature and marked it `override`, so the CUDA half of the tree **did not
  compile** against its own base class. Restored here, with the argument
  accepted and unused (the device kernels do not screen on pair norms yet) so
  one interface drives both engines — which is what lets the K engine's driver
  be written once. `tests/libintx.gpu.md{3,4}.{test,perf}.cc` updated to pass
  `{}`.
- `src/libintx/CMakeLists.txt` — `find_library(lapacke)`, see above; plus
  `jengine.h`/`kengine.h` added to the installed headers.
- `src/libintx/ao/md/CMakeLists.txt`, `src/libintx/gpu/md/CMakeLists.txt`,
  `tests/CMakeLists.txt` — build the new sources and tests.
- `.github/workflows/ci.yml` — the added Linux job. The macOS job is untouched.
- `README.md` — a short section on the K engine, above "Using".

Because the device tree did not compile before this fork, **the GPU K engine and
the GPU MD signature fix have not been executed anywhere** — there is no CUDA
toolkit or device in the environment they were written in. They are compile-
correct by construction and untested. Run
`ctest --preset default -R 'gpu\.(md|kengine)'` on a machine with a card before
trusting either.
