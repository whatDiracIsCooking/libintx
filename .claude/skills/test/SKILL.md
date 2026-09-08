---
name: test
description: >-
  Run libintx's doctest suite the right way and interpret the result honestly —
  pick the preset (workstation on a bare host, default in the CUDA container),
  remember that test binaries are EXCLUDE_FROM_ALL, and ALWAYS check what the
  configured LIBINTX_MAX_L actually covered before calling a run green. Use
  when the user asks to run tests, check a change, or verify an edit.
---

# Running tests, and trusting the result

libintx has **one** suite: doctest executables under `tests/`, registered with
`add_test()` and driven by `ctest`. There is no Python tier (`python/tests/`
exists but needs the pybind11 bindings built, which is off by default).

Two rules matter more than the commands.

## Rule 1: a build with no `all.tests` has no tests

Every test executable in `tests/CMakeLists.txt` is declared
`EXCLUDE_FROM_ALL` and collected under the `all.tests` custom target. So:

```bash
cmake --build --preset workstation          # libraries only. NO test binaries.
ctest --preset workstation                  # "No tests were found"
```

That message reads like a broken configuration and is actually a missing
target. The build presets in `CMakePresets.json` name `all all.tests`, and so
does `devtools/build.sh` through them, so use the presets rather than a bare
`cmake --build <dir>`.

## Rule 2: a green run only covers the `LIBINTX_MAX_L` it was configured with

`LIBINTX_MAX_L` (default 3) fixes the angular momentum the kernel tables are
instantiated for, at **configure** time. `tests/test.h`'s `test::enabled(...)`
and the `if (LMAX < 3) return;` guards in the K engine test then quietly skip
every case above it. A run configured with `-DLIBINTX_MAX_L=2` is green without
having compiled or executed a single f-shell kernel.

So when reporting a result, say which `LIBINTX_MAX_L` it was, and say whether
the GPU half was in the build at all. "All tests passed" on a laptop means the
host tree at whatever L that build was configured for — the device engines were
not built, let alone run.

## The commands

```bash
# Bare host / CPU only -- the host boys, md2/md3/md4 and K engines.
cmake --preset workstation && cmake --build --preset workstation
ctest --preset workstation --output-on-failure

# The whole thing, from inside the CUDA container.
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh

# One test.
ctest --preset workstation -R kengine --output-on-failure

# What exists without running it.
ctest --preset workstation -N
```

`devtools/cpp-tier.sh` is configure + build + ctest in one, logged to
`.slow-tier-reports/cpp-<stamp>-<rev>-<preset>.log` with a PASS/FAIL line
appended to `summary.log`. **Quote the log path when reporting a failure** —
the summary has none of the diagnostics. `devtools/build.sh` is the same
configure + build with no ctest.

## Picking a preset

| Preset | When |
|---|---|
| `workstation` | A bare host, a laptop, CI. CPU only; no device needed. |
| `default` | Inside the CUDA container, on the machine with the card. |
| `ampere` / `hopper` / `portable` | Build host and run host differ. |
| `debug` / `asan` | Chasing a bug. |

**`default` cannot configure without a GPU.** It sets
`CMAKE_CUDA_ARCHITECTURES=native`, which queries a live device at *configure*
time — so on a machine with no card it fails before compiling anything, even
for a change that touches no CUDA. That is what `workstation` is for.

Only `default`, `workstation`, `debug` and `asan` have **test** presets. With
any other configure preset, ctest has nothing to drive; that is not a pass.

## Reading a failure

doctest prints the failing `CHECK` with both values and the index the test
tagged it with (`test::ReferenceValue::at(...)`), e.g.

```
CHECK( k(i,j) == expect )
with expansion: 0.4172 == 0.4181 @ [ 3 7 ]
```

The index is the reference's own, so it points at a basis-function pair, not at
a line of the engine. For the K engine specifically, a failure that is
*systematically* a small integer multiple of the reference is a digest
degeneracy bug (the eight-fold permutation orbit), not an integral bug — the
integrals themselves are covered by `libintx.md4.test` against
`libintx::md::reference`.

## What the push gate does and does not run

`devtools/cpp-smoke.sh` builds incrementally with `SMOKE_PRESET`
(`workstation`) and runs only `SMOKE_TESTS` from `devtools/config.sh` — the
cheap host cases. It skips itself, not the push, when cmake or the generator is
missing. It is a smoke gate: the full sweep of every angular-momentum class
stays in `devtools/cpp-tier.sh`, and that is the one to run before opening a
PR.
