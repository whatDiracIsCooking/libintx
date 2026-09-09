# LibintX

LibintX is a library for accelerated evaluation of molecular integrals of many-body operators over Gaussian atomic orbitals. The primary purpose of LibintX is to enable efficient evaluation of 2-body operators in Gaussian AO integrals on accelerated architectures like the CUDA-capable graphical processing units (GPU). However, it can also be used on conventional/central processing units (CPUs).

Until version 1.0 this will remain EXPERIMENTAL code development; expect things to break and APIs to change.  https://github.com/ValeevGroup/libintx/ is the public mirror of the private development repo.  Don't make PRs against the public mirror; if you wish to collaborate send us a request.

# Installation

## Prerequisites
- CMake
- A C++ compiler with support for the 2017 C++ standard ([the list of compilers with partial or full support for C++17](https://en.cppreference.com/w/cpp/compiler_support/17))
- CUDA toolkit, version 11 or higher (optional)

Other CMake parameters:
- LIBINTX_MAX_K - maximum primitives
- LIBINTX_MAX_L - maximum angular momentum
- LIBINTX_MAX_X - maximum auxillary angular momentum
- LIBINTX_GPU_MAX_SHMEM - maximum GPU shared memory per *threadblock*

## Building
- configure: \
    `cd libintx` \
    `cmake -B ./build` \
    `cd ./build` \
  with CUDA: \
    `cmake -DLIBINTX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 .` \
  with HIP: \
    `export PATH=$PATH:/opt/rocm/bin # or wherever hipcc lives`
    `cmake -DLIBINTX_HIP=1 .` \
  with libint2 as perf reference: \
    `cmake -DLIBINTX_LIBINT2=ON .`
- build: \
    `cmake --build .`
- 3-center tests and benchmarks: \
    `cmake --build . --target libintx.gpu.md3.test` \
    `./tests/libintx.gpu.md3.test`
    `cmake --build . --target libintx.gpu.md3.perf` \
    `./tests/libintx.gpu.md3.benchmarks`
- 4-center tests and benchmarks: \
    `cmake --build . --target libintx.gpu.md4.test` \
    `./tests/libintx.gpu.md4.test`
    `cmake --build . --target libintx.gpu.md4.perf` \
    `./tests/libintx.gpu.md4.benchmarks`

# Python
- Configure with LIBINTX_PYTHON=TRUE
- Build `libintx-python` target

# Conventional J and K engines — fork addition

Upstream's only Coulomb engine is density fitted. This fork adds *conventional*
(integral-direct) four-centre J and K engines alongside it, both behind the
existing shell-block tile interfaces:

```
J[mu,nu] = sum_{lambda,sigma} (mu nu | lambda sigma) D[lambda,sigma]
K[mu,nu] = sum_{lambda,sigma} (mu lambda | nu sigma) D[lambda,sigma]
```

```cpp
#include "libintx/ao/md/jengine.h"   // host J
#include "libintx/ao/md/kengine.h"   // host K
#include "libintx/gpu/jengine.h"     // device J (DF and direct)
#include "libintx/gpu/kengine.h"     // device K

auto screening = libintx::md::make_schwarz_screening(basis, 1e-12f);

auto j = libintx::md::make_jengine(basis, screening);
j->J(read_density_tile, write_coulomb_tile, allsum);

auto k = libintx::md::make_kengine(basis, screening);
k->K(read_density_tile, write_exchange_tile, allsum);
```

`libintx::gpu::make_jengine_direct` and `libintx::gpu::make_kengine` are the
same calls on the device MD engine. All four share
`src/libintx/fock/md/driver.h` — one shell-pair binning, one screening, one
eight-fold permutation orbit, differing only in which two of the four permuted
slots accumulate and which two contract against the density — and the host
engines are what the device ones are tested against.

The density-fitted engine is still there, now spelled
`libintx::gpu::make_df_jengine` (`make_jengine` remains as an alias). It is a
different algorithm behind the same `libintx::JEngine` interface: it needs an
auxiliary basis and `V^-1`, and carries that basis's fitting error. The
conventional J above needs neither, and pairs exactly with the conventional K
for `F = H + J - K/2`.

Contraction coefficients must be primitive-normalized (which is what
`libintx::make_basis(..., normalize=true)` does) — see `CLAUDE.md`.

There is also a **density-fitted** K engine behind the same `KEngine`
interface, alongside the integral-direct one rather than in place of it. It
takes an auxiliary basis and a metric transform, the way `JEngine` does:

```cpp
auto k = libintx::md::make_df_kengine(basis, df_basis, V_linv, screening);
k->K(read_density_tile, write_exchange_tile, allsum);
```

`V_linv(X, n)` replaces a row-major `naux x n` block with `V^-1 X`, where
`V[P,Q] = (P|Q)`; it is the caller's, exactly as it is for
`make_df_jengine`. `libintx::gpu::make_df_kengine` is the device counterpart,
and both share `src/libintx/fock/md/df.h` — three-centre integrals and two
GEMMs per auxiliary function, where `driver.h` is four-centre integrals and
the permutation orbit. Unlike the direct engine this is an approximation — it
reproduces a direct K only to the quality of the auxiliary basis — so the two
are alternatives to pick between, not implementations to check against each
other.

## Gradients: what libintx owns, and what the caller does

Analytic nuclear gradients are planned as **per-term derivatives**, not as a
total force. For a closed shell,

```
dE/dX = sum D_uv d(T+V)_uv/dX  +  [J and K terms]  -  sum W_uv dS_uv/dX  +  dE_nn/dX
```

libintx is an integrals library, so it ships the derivative of each term it
computes and stops there. Two pieces of that expression are deliberately
**outside** it, and their absence is a scope decision rather than an omission:

- **Nuclear repulsion.** `E_nn` and `dE_nn/dX` are arithmetic over point
  charges with no integral, no basis and no Gaussian in them. They appear
  nowhere in this tree and are not planned to.
- **Assembly, `D`, and `W`.** Nothing here composes the per-term derivatives
  into a `3*natom` force, because doing so needs the density `D` and the
  energy-weighted density `W = 2 C_occ e C_occ^T` (closed-shell RHF) that only
  the caller's SCF has. The gradient engines read `D` through the same tile
  callbacks `JEngine` and `KEngine` use; `W` never enters an engine at all,
  and the Pulay term `sum W_uv dS_uv/dX` is the caller's contraction against
  the overlap derivative.

**What exists so far** is the *integral* layer of two of the terms, on the
device, all through a `compute1` entry point that writes the value layout with
the Cartesian component as one more, slowest index.

- `dS/dX`, as `ao::IntegralEngine<2>::compute1(Operator::Overlap, ...)` -- the
  Pulay derivative the caller contracts against `W`. It computes only the
  **bra** derivative: a two-centre integral depends on the centres only through
  `r_a - r_b`, so `dS/dB = -dS/dA` elementwise. A caller scatters `+V` onto the
  bra shell's atom and `-V` onto the ket shell's, and must *accumulate* -- a
  pair with both shells on one atom hits the same slot twice.
- `d(ab|cd)/dX` and `d(P|cd)/dX`, as
  `IntegralEngine<4>::compute1(Coulomb, centre, ...)` and
  `IntegralEngine<3>::compute1(...)`. These take a centre selector, because
  three- and four-centre integrals have no `d/dA = -d/dB` identity to collapse
  them -- translational invariance relates only the sum over all centres. The
  four-centre engine computes all four; the three-centre one computes the two
  ket centres, and `d/dP = -(d/dC + d/dD)` gives the auxiliary one.

Everything else in the expression above is still planned -- no derivative J or
K, no gradient assembly, no host derivative kernel of any kind -- and
`compute1` throws for it rather than return zeros.

What libintx does test is that its own terms compose: the assembled gradient
against central differences of the assembled energy expression, at a fixed `D`
and `W` that need not be converged or physically meaningful. That check is
exact -- at fixed `D` and `W` the expression above is an ordinary function of
the geometry -- but it cannot see whether a caller's `W` is the right `W` for
its wavefunction. That one is on the caller.

# Using
Still work in progress.  Read through test programs and/or contact Andrey, asadchev@gmail.com

# Developers
LibintX is developed by the [Valeev Group](http://valeevgroup.github.io/) at [Virginia Tech](http://www.vt.edu).

# License

LibintX is freely available under the terms of the LGPL v3+ licence. See the included LICENSE file for details. If you are interested in using LibintX under different licensing terms, please contact us.

# How to Cite

See the enclosed LICENSE file.

# Acknowledgements

Development of LibintX is made possible by the support provided by the Department of Energy Exascale Computing Project ([NWChemEx subproject](https://github.com/NWChemEx-Project)).
