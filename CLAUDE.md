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
| `src/libintx/` | `shell.h`, `orbital.h`, `array.h`, `math.h`, `tensor.h`, `simd.h` — the value types everything else is written against. Plus `blas.cc`, the two engine interfaces `jengine.h` and `kengine.h`, and `screening.h` (the pair bounds they share). |
| `src/libintx/boys/` | The Boys function: Chebyshev interpolation + asymptotic tail. Host and (`gpu/chebyshev.h`) device. |
| `src/libintx/ao/md/` | The **host** McMurchie–Davidson engines: `IntegralEngine<2>` (overlap/kinetic/nuclear), `<3>`, `<4>`, the Hermite machinery (`hermite.h`, `r1/`), a plain reference (`reference.h`), the host conventional J and K engines (`jengine.cc`, `kengine.cc`, `screening.cc`), and the host **density-fitted** K engine (`df.kengine.cc`). |
| `src/libintx/gpu/` | The device half. `api/` wraps CUDA/HIP behind one `gpu::` namespace; `md/` is the device `IntegralEngine<3>`/`<4>` plus the device K engines (`kengine.cc` direct, `df.kengine.cc` density-fitted) and the device *conventional* J engine; `onebody/` is the device `IntegralEngine<2>` and the pieces its operator kernels share, with `overlap/`, `kinetic/` and `potential_en/` its three operator kernels (`overlap/` also carries the tree's one derivative kernel, `dS/dX`); `jengine/md/` is the **DF** J engine, a different algorithm in its own target; `eri/` materialises the full ERI tensor. |
| `src/libintx/fock/md/` | `driver.h` is the conventional Fock build shared by all four of those engines: shell-pair binning, screening, the eight-fold digest, parameterised on the scatter. `df.h` is the density-fitted K build, which reuses the binning and the tile plumbing but has no digest at all. |
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
Nothing pads. That three-way binning is what `fock::md::make_pair_classes`
exists to do.

**`JEngine`** (`src/libintx/jengine.h`) — the Coulomb engine *interface*. It
says nothing about how J is built; three factories implement it. It reads D and
writes J through shell-block tile callbacks so a distributed caller never has to
materialise either matrix.

- `gpu::make_df_jengine(basis, df_basis, V_linv, screening)` — the upstream
  **density-fitted** engine: two three-centre passes around an `AllSum` on the
  fitted vector `X_Q`, never forming a four-index `(mu nu|lambda sigma)`. GPU
  only, in `libintx.gpu.jengine`. Needs an auxiliary basis and a caller-supplied
  metric solve, and carries that basis's fitting error. `gpu::make_jengine` is
  kept as an inline alias so existing callers still compile.
- `md::make_jengine(basis, screening)` and
  `gpu::make_jengine_direct(basis, screening, stream)` — the **conventional**
  (integral-direct) engines added in this fork: four centres, no auxiliary
  basis, no metric, no fitting error.

`src/libintx/gpu/jengine/os/` is a fourth, Obara–Saika DF J engine and is **dead
code**: no `CMakeLists.txt` references it, and all three of its includes
(`libintx/cuda/{eri.h,jengine.h,api/api.h}`) predate the `cuda/` → `gpu/` rename
and no longer exist. Do not mistake it for a starting point.

**`KEngine`** (`src/libintx/kengine.h`) — the exchange engine, added in this
fork.

```
J[mu,nu] = sum_{lambda,sigma} (mu nu | lambda sigma) D[lambda,sigma]
K[mu,nu] = sum_{lambda,sigma} (mu lambda | nu sigma) D[lambda,sigma]
```

Same tile-callback shape as `JEngine`, deliberately, so a caller already
driving J drives K the same way. `F = H + J - K/2` for a closed shell, with J
and K contracted against the same `D = 2 C_occ C_occ^T` — and, if J is the
conventional one, out of the same integrals, so there is no fitting error on one
side to reconcile against an exact other side.

Four conventional engines, one algorithm:

| | Host | Device |
|---|---|---|
| Factory (J) | `libintx::md::make_jengine` | `libintx::gpu::make_jengine_direct` |
| Factory (K) | `libintx::md::make_kengine` | `libintx::gpu::make_kengine` |
| Headers | `src/libintx/ao/md/{j,k}engine.h` | `src/libintx/gpu/{jengine,kengine}.h` |
| Integrals | `md::IntegralEngine<4>` | `gpu::md::IntegralEngine<4>` |
| Built into | `libintx.md4` | `libintx.gpu.md4` |

All four are thin: they supply an engine type, a buffer and a term, and
`src/libintx/fock/md/driver.h` does the rest. The device path produces the
`(ab|cd)` batch into host-registered memory and digests on the host — the
straightforward split, not the final one; a device-side digest would not change
the interface.

**And two density-fitted K engines**, `libintx::md::make_df_kengine` and
`libintx::gpu::make_df_kengine`, behind the same `KEngine` interface. They are
*additive*: the direct engines are untouched, both families answer the same
`K()`, and a caller picks one at construction. K now has the same
direct-or-DF choice J has had — see below.

### The density-fitted K build

```
(mu lambda | nu sigma) ~= sum_PQ (P|mu lambda) [V^-1]_PQ (Q|nu sigma)
A[P,mu,nu] = (P|mu nu),  B = V^-1 A,  K[mu,nu] = sum_P (B_P D A_P)[mu,nu]
```

so `src/libintx/fock/md/df.h` is three-centre integrals plus two GEMMs per
auxiliary function, where `driver.h` is four-centre integrals plus the
permutation digest. It reuses the driver's pair binning, `Matrix`, `Density`
and tile plumbing; what it does *not* share is the orbit, because a DF build
never forms a quartet.

| | Host | Device |
|---|---|---|
| Factory | `libintx::md::make_df_kengine` | `libintx::gpu::make_df_kengine` |
| Integrals | `md::IntegralEngine<3>` | `gpu::md::IntegralEngine<3>` |
| Built into | `libintx.md3` | `libintx.gpu.md3` |

**They are in the 3-centre libraries, not the 4-centre ones** — they need
`IntegralEngine<3>` and nothing four-centre. So **a caller that wants both K
engines links both `libintx.md3` and `libintx.md4`**; `make_df_kengine` will
not link from `libintx.md4` alone. Three things more:

- **`V^-1` is the caller's, not the engine's**, exactly as it is for
  `make_df_jengine`. `KEngine::MetricTransform` is `void(double *X, size_t n)`
  and must replace a **row-major** `naux x n` block with `V^-1 X` — the DF J
  engine's `V_linv` generalised from one column to n, because a K build has to
  transform the whole `(P|mu nu)` tensor rather than a single vector. It is
  required; `K()` throws without one.
- **It is an approximation and the direct engine is not.** DF K reproduces
  direct K only to the quality of the auxiliary basis, so the two are not
  cross-checkable to round-off. What *is* exact — and what
  `libintx.df.kengine.exact_fit` tests — is the case where every product
  density lies in the auxiliary span: uncontracted s shells with exponents
  `a1`, `a2` have products at exponent `a1+a2` on their centre of charge, so
  three auxiliary functions span a two-shell basis exactly and DF then has to
  match the direct engine to round-off.
- **Memory is `2*naux*nbf^2` doubles**, both `A` and `B` in core, and the
  contraction is `2*naux*nbf^3`. A DF-K in an SCF is usually written against
  the occupied coefficients, which drops the `nbf^3` to `nbf^2*nocc` — but
  `KEngine` hands the engine a density through tile callbacks, not a `C_occ`,
  and a general `D` has no factorisation to exploit through that interface. So
  this is the right form *for this interface*; an occupied-space variant needs
  a different one.

### The device one-electron engine

`gpu::md::IntegralEngine<2>` (`src/libintx/gpu/onebody/`) is the device
counterpart of the host `md::IntegralEngine<2>`. The engine type, the factory,
the batch upload, the point-charge upload and the `(A|B)` dispatch table are all
wired, and **all three operators have kernels**: `Operator::Overlap`
(`src/libintx/gpu/overlap/`), `Operator::Kinetic` (`src/libintx/gpu/kinetic/`)
and `Operator::Nuclear` (`src/libintx/gpu/potential_en/`). Nothing reaches the
`(A|B)` fallback table but `Operator::Coulomb`, which is not a two-centre
operator this engine implements; `compute` throws for it rather than hand back
a buffer of zeros.

Each operator is one line in `gpu/onebody/md2.cc` dispatching to its own
translation unit, which owns the `(LMAX+1)^2` `(A|B)` instantiations its kernel
is compiled into (`gpu/{overlap,kinetic,potential_en}/*.h` are that whole
interface: one non-template launcher taking an uploaded bin). A function
template crossing that boundary would have to be explicitly instantiated over a
table whose size is a configure-time decision, which is why the dispatch is
split in two rather than done once. `block_size<A,B,DB,DA>()` in
`gpu/onebody/kernel.h` is the one piece of that per-operator boilerplate the
three share.

Three things the scaffolding settles, which all three kernels are the consumers
of:

- **Namespace.** `libintx::gpu::md`, the same as the device Coulomb engines,
  even though the files sit outside `gpu/md/`. The algorithm is still
  McMurchie-Davidson; a second namespace for the same method would buy nothing
  and cost a rename across four directories.
- **`E2` is shared, and carries its extra ket degree.** The device Hermite
  expansion used to be file-private inside `gpu/md/basis.cu`; it is
  `src/libintx/gpu/md/e2.h` now, `E2<A,B,DB>`, with `DB` the extra ket degree
  an operator needs (`0` for the Coulomb path and overlap, `2` for kinetic,
  which is what the host spells `libintx::md::E2<T,A,B+2,0>`). It stays under
  `gpu/md/` because both paths use it and the Coulomb path should not depend on
  a one-electron directory. Its `init()` parallelises the recursion over the
  Hermite index and so **requires `Block::size() >= A+B+DB+1`** -- a real
  constraint on every caller's block shape, documented on the class and
  asserted at compile time. `tests/libintx.gpu.e2.test.cu` checks it against
  the host `libintx::md::E2` for both `DB` values over all `(A,B)`.
- **The batch is raw primitives, not baked Hermites.** `gpu::md::make_basis`
  bakes E into a `[ab,p]` buffer over `nherm2(A+B)` because that is what md3
  and md4 consume; no single baked buffer serves all three one-electron
  operators without waste (overlap wants `E(a,b,0)`, kinetic wants `(A,B+2)`,
  only the nuclear kernel wants the full index, which is why it is the one that
  might have justified the exception). So `gpu/onebody/basis.h`
  uploads `Gaussian2` -- the same POD, promoted out of `basis.cu` into
  `gpu/md/basis.h` -- and each kernel builds the slice of E it wants in shared
  memory, via the block-level skeleton in `gpu/onebody/kernel.h`.

**And a fourth entry point, `compute1`** -- the first geometric derivative,
`Operator::Overlap` only so far. It is a separate virtual on
`ao::IntegralEngine<2>` rather than a fifth `Operator` or a defaulted `deriv`
argument, because a derivative is a derivative *of* an operator and because a
default argument on a virtual binds to the static type. The `1` is the
derivative order, not a centre count. Output is the `compute` layout with the
component as one more, slowest index,

```
V[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV],   x = 0,1,2 -> d/dA_x
```

so the `x = 0` block has exactly the shape `compute` writes. **Only the bra
derivative is computed**: a two-centre integral depends on the centres only
through `r_a - r_b`, so `dS/dB = -dS/dA` elementwise and the caller scatters
`+V` onto the bra shell's atom and `-V` onto the ket shell's -- accumulating,
because a pair with both shells on one atom hits the same slot twice. The host
`md::IntegralEngine<2>::compute1` throws (there is no host derivative kernel at
all), and the device one throws for kinetic, nuclear and Coulomb.

`set()` uploads the point charges once per geometry, not once per `compute`,
and a second `set()` replaces the first. The output layout matches the host
byte for byte -- `V[ij + (na + nb*npure(A))*ldV]`, `ldV = ijs.size()`, into
host memory the caller has registered -- so `test::check2` and any caller are
drop-in.

**The overlap kernel** is the cheapest of the three: no Boys function, no
Hermite `R` tensor, only the `t = 0` expansion coefficient, so

```
S_ab = (pi/p)^(3/2) * prod_x E^{i_x j_x}_0,   p = a + b
```

is `libintx::md::overlap` (`src/libintx/ao/md/md2.cc`) with its loop over the
Cartesian components spread across a thread block. One block per shell pair,
`32*ceil(ncart(A)*ncart(B)/32)` threads capped at 128, each thread owning its
own element of the Cartesian accumulator so there are no atomics; the primitive
loop, E in shared memory, the cartesian-to-pure pass and the output write are
all `gpu/onebody/kernel.h`'s. The batch must be solid-harmonic, asserted --
the host engine writes the pure layout unconditionally, so there is no agreed
answer for a Cartesian one to be checked against.

Two things it does **not** settle. The block shape is the issue's proposal, not
a measured optimum: pairs along `threadIdx.x` and components along
`threadIdx.y`, the way md4's `md_v0_kernel_base` bins them, should win for a
small `(A|B)` with `K = 1`, where a whole block per pair has almost nothing to
do. Nobody has measured it. And the `Pure = false` branch of
`onebody::compute2` is still uninstantiated by anything -- including, contrary
to what the gradient issue predicted, by the overlap derivative below.

**The overlap gradient kernel** (`overlap1` in the same `gpu/overlap/` files)
is `dS/dA_x`. Differentiating a Cartesian primitive with respect to its own
centre raises and lowers one Cartesian index,

```
d/dA_x (a|  =  2*alpha*(a+1_x|  -  i_x*(a-1_x|
```

and because `S` factorises axis by axis it is the same product with one axis
replaced:

```
dS/dA_x = (pi/p)^(3/2) * [ 2a*E^{i_x+1,j_x}_0 - i_x*E^{i_x-1,j_x}_0 ]
                       * prod_{y != x} E^{i_y j_y}_0
```

Two generalisations of the shared skeleton carry it, and nothing else changes:
`compute2` gained `DA`, an extra **bra** degree (E is built as `E2<A+DA,B,DB>`;
480 doubles at `(3|3)` against overlap's 336), and `NC`, the number of output
components, which widens the Cartesian accumulator to `NC*ncart(A)*ncart(B)` and
runs the cartesian-to-pure pass once per component. Both default to `0` and `1`,
so the three value kernels are untouched.

**It does not shift a shell, and that is what keeps it off the normalization
trap.** The textbook route builds an `L+1` copy of the bra shell; `gto::normalized`
carries an `L`-dependent factor, so re-normalizing that copy is a silently
different basis and a smooth, plausible, wrong gradient. Here the raised index
is read out of `E` inside the kernel, so the batch, its contraction coefficients
and the accumulator all stay at `L` and the parent's coefficients are the only
ones in play. For the same reason there is **no Cartesian output path**: the
solid-harmonic transform is linear with constant coefficients, so it commutes
with `d/dA_x` and each component is transformed exactly as a value is, even
though `T_pure(L)*(a+1_x)` is not a pure `L+1` function. The `(A|B)` table stays
`(LMAX+1)^2` and the solid-harmonic guard stays as it is -- both of which the
issue expected to have to change.

Note `pair.C` carries `K_ab = exp(-a*b/p*|AB|^2)`, which depends on the bra
centre too. That is not a missing term: the raising relation is an identity on
the *primitive function*, so `S_{a+1_x,b}` evaluated with the same `K_ab` --
which is what `C*E(a+1_x,b,0)` is -- already carries the whole `A`-dependence
through `E`'s own recursion. Getting this wrong looks like a gradient that is
right for a one-centre pair and wrong everywhere else.

**The kinetic kernel** (`src/libintx/gpu/kinetic/`) is the same skeleton with
`DB = 2` and a three-term body. Differentiating the ket twice turns

```
T_ab = -1/2 <a| nabla^2 |b>
     = (pi/p)^(3/2) * [ b*(2*B+3)*t0 - 2*b^2*t1 - (1/2)*t2 ]
```

into a fixed combination of *overlap* integrals with the ket degree shifted by
`+2`, `0` and `-2` along each axis (`t1`, `t0`, `t2`), which is
`libintx::md::kinetic` (`src/libintx/ao/md/md2.cc`) spread across a block the
same way. `E2<A,B,2>` is 648 doubles (5.2 KB) at `(3|3)` against overlap's 336;
the block shape, the primitive loop, the pure pass and the output write are
unchanged, and `block_size<A,B,DB>()` moved into `gpu/onebody/kernel.h` when
the second kernel wanted it.

**It does not replicate the host's transpose branch, deliberately.**
`libintx::md::compute2` (`ao/md/md2.cc:246-257`) evaluates `(A|B)` as
`kinetic<B,A>` on the swapped, sign-flipped pair whenever `B > A`, writing
through a transposing accessor, on the grounds that the `+2` is cheaper on the
smaller index. On this skeleton that trade runs backwards: `E2<A,B,2>` is
`(A+1)*(B+3)*(A+B+3)*3` doubles and the swap makes it `(B+1)*(A+3)*(...)`, a
difference of `2*(A-B)` per `(A+B+3)*3` — so the swapped table is strictly
*larger* exactly when `B > A`, which is precisely when the host swaps (6
doubles per axis against 12 at `(s|f)`), and `E2::init` syncs 5 times against
11 there. On top of that `compute2` fixes the accumulator as
`U[ia + ib*ncart(A)]`, so a swap would have to be undone in every write of the
operator functor — the one line the issue calls the most bug-prone in the host
path, bought for nothing. Swapping the *other* way, when `A > B`, would be a
real saving by both counts, but it needs the skeleton to build E on the swapped
pair rather than the functor to transpose its writes; nobody has measured
whether that is worth it.

**The electron-nuclear potential kernel** (`src/libintx/gpu/potential_en/`) is
the substantial one of the three. Overlap and kinetic are products of
one-dimensional `E` coefficients; `1/r` is not, so this kernel carries the two
pieces the Coulomb path carries -- the Boys function and the Hermite `R` tensor
-- and a per-molecule parameter set that lives on the device between `compute`
calls. Per primitive pair, mirroring `libintx::md::nuclear`
(`src/libintx/ao/md/md2.cc`):

```
s[m] = F_m(p*|PC|^2) * (-2p)^m,  m = 0..A+B,  PC = P - R_C
R[t] += -Z_C * r1(PC, s)[t]                         over every nucleus C
V_ab  = (2*pi/p) * sum_{t <= a+b} R[t] * E^{a b}_t
```

Neither hard piece is reimplemented. `gpu::boys()` (`src/libintx/gpu/boys.h`) is
the tree's **one** device Chebyshev table and already covers order `A+B <=
2*LMAX`; a second table would be a duplicate upload of the same interpolation
data. `libintx::md::r1::visit` (`src/libintx/ao/md/r1.h`) is
`LIBINTX_GPU_ENABLED` and is literally what the device Coulomb kernels call, so
the recursion cannot drift between the two- and four-centre paths.

Three things specific to it:

- **It is the one operator that reads `E` over the full `nherm2(A+B)` Hermite
  index**, not just degree 0 -- which is the whole triangle `E2<A,B,0>` stores
  anyway, so `DB` is still 0 and the shared skeleton is unchanged. (Whether
  reusing md3/md4's baked `[ab,p]` buffer would beat rebuilding `E` in shared
  memory is the open question the scaffolding issue flagged; this does the
  latter, for uniformity with overlap.)
- **The nuclei are spread across the block, and `R` is a shared-memory
  `atomicAdd` reduction** over `nherm2(A+B)` slots (84 at `(3|3)`). The issue's
  recommended starting point was a strictly serial nuclei loop; that leaves 127
  of 128 threads idle through the part that dominates the kernel --
  `r1::visit<A+B>` once per nucleus per primitive pair, against a contraction
  that is one pass over `ncart(A)*ncart(B)`. Summation order is therefore not
  reproducible run to run, which is immaterial at 1e-10 and is not a
  correctness question: every term is added exactly once. **Nothing here has
  been benchmarked** -- there is no device in the environment it was written in
  -- so read the block shape as a starting point, not a measured optimum.
- **`compute(Nuclear, ...)` before any `set()` throws**, rather than reading an
  empty `device::vector` and returning zeros; and `set()` re-uploads every time,
  so a second `set()` with different centres is not silently ignored. Both are
  real bug classes for a geometry optimisation or a finite-difference gradient,
  and both have a test case.

### Three things about the shared digest

**J and K differ in one statement, and nothing else.** `fock::md::digest()` is
parameterised on which two of the four permuted slots accumulate into the output
and which two contract against the density:

```
CoulombDigest   out (0,1), in (2,3)    j(x0+n0, x1+n1) += v[off]*d(x2+n2, x3+n3)
ExchangeDigest  out (0,2), in (1,3)    k(x0+n0, x2+n2) += v[off]*d(x1+n1, x3+n3)
```

Everything else — the binning, the batching, the orbit — is one piece of code
on purpose: the orbit logic is exactly the part that must not drift between the
two builds. `digest()` takes a *pack* of `Term`s rather than one, so a single
sweep can feed J and K at once and halve the integral cost of a Fock build.
`tests/libintx.jengine.test.cc`'s `fused` case exercises that path; no engine
uses it yet.

**The eight-fold orbit is enumerated, not weighted.** A Fock term sums over
*every* index tuple, so a unique quartet contributes once per distinct member of
its permutation orbit. Carrying per-case degeneracy factors is where such a
digest normally goes wrong, because `a==b`, `c==d` and `(ab)==(cd)` each
collapse a *different* subset. `digest()` instead enumerates the eight
permutations, deduplicates on the permuted shell tuple, and gives every survivor
weight one — which is why it carries the permutation index around rather than
just the permuted shells: the index says where in `V` that member's value lives.

If a J or K result is wrong by a small integer factor in a systematic pattern,
it is this, not the integrals. The integrals have their own test
(`libintx.md4.test` against `libintx::md::reference`). The DF K build has no
digest and no orbit — it never sees a quartet — so this failure mode is
specific to the conventional engines.

**Contraction coefficients must be primitive-normalized.** A basis-set library's
coefficients are defined against normalized primitives. For a contracted shell
that is *not* an overall per-basis-function scale — the primitives carry
different exponents — so feeding raw coefficients through is a physically
different basis, not a rescaled one, and the error survives all the way to the
SCF energy rather than showing up as an obviously broken integral.
`libintx::make_basis(..., normalize=true)` applies `gto::normalized` and that is
the convention these engines assume; if you build a `Basis` by hand with
`normalize=false`, J, K and the one-electron integrals must all agree on that
choice. (This is the same trap that produced a ~6.6 Ha error in the project this
engine was ported from, where it was first misdiagnosed as a d-shell kernel bug.)

### Screening

`libintx::PairScreening` (`src/libintx/screening.h`) is the shared shape:
`max2(i,j)` is the Schwarz bound `sqrt(max |(ij|ij)|)`, plus `max()` and
`skip()`. `KEngine::Screening` is an alias of it, and `JEngine::Screening`
*derives* from it, adding only the DF engine's per-auxiliary-shell `max1(int)` —
so one screening object serves the conventional J engine, the K engine, or both.

`make_schwarz_screening(basis, threshold)` builds one (there is a host and a
device version; the result is plain host data either way, so they are
interchangeable). It evaluates one diagonal quartet per call — a batch would
have to be square in the shell pairs to keep its diagonal, which is `|pairs|`
times the integral work for the same `|pairs|` numbers. It runs once per
geometry, not once per SCF iteration.

**The density factor comes from the term's own `in[]` slots**, per orbit member,
not from a fixed list — which makes it automatically right for both builds and
tighter than a hand-written K bound. For K the reachable blocks are the
**cross** ones (`D[ac]`, `D[ad]`, `D[bc]`, `D[bd]`, …) because K couples the bra
indices to the ket indices, which is exactly what makes a J screen too loose for
it; for J they are only `D[ab]` and `D[cd]`, which is strictly tighter and is
where a direct J earns its cost. The two orbit members that write a matrix
element and its transpose read the same density block (transposed), so for the
symmetric D an SCF hands these engines they carry the same bound and a screened
build stays symmetric.

The DF K engines take the same `PairScreening` and get less out of it: their
integrals are three-centre, so there is no second pair to bound and no
`max1()` on this interface to bound the auxiliary bra with. A pair goes on
`max2(i,j)` against the largest possible partner — the same product the direct
engines apply, never more aggressive. Wiring `JEngine::Screening::max1()`
through is the obvious way to tighten it, and needs a screening type that has
it.

## Gradients: planned, and where the scope line is

Nothing in the tree computes a derivative integral yet -- host or device, one-
or two-electron. Analytic gradients are tracked as a DAG of issues; what
matters for anyone starting on them is that libintx ships **per-term
derivatives** and never a total force:

```
dE/dX = sum D_uv d(T+V)_uv/dX + [J and K] - sum W_uv dS_uv/dX + dE_nn/dX
```

**Nuclear repulsion and gradient assembly are the caller's**, deliberately.
`E_nn` involves no integral, and assembly needs `D` and the energy-weighted
density `W` that only an SCF has. Neither is missing by accident; see README's
"Gradients: what libintx owns, and what the caller does".

Two structural facts, both verified against the code, that shape all of it:

- **The device Coulomb kernels are pure-only, and that is the hard part.**
  `gpu/md/basis.cu` bakes the solid-harmonic transform into the batch before
  the kernel runs, and `gpu/md/md.kernel.h:386` contracts against a
  `npure(A)*npure(B)`-wide transform fixed at compile time. Since
  `T_pure(L)*(a+1_x)` is not a pure `L+1` function, the textbook shifted-shell
  route (`d/dA_x (a| = 2a*(a+1_x| - i_x*(a-1_x|`) has nowhere to put its
  Cartesian intermediate. The kernel *is* generic in that transform, though --
  it does not know it is applying a pure transform -- so baking `dE/dX` into
  that slot is the cheaper route than adding a Cartesian output path. Either
  way it is kernel work, not driver work.
- **The one-electron skeleton does not have that problem**, and `dS/dX` is
  done -- `gpu::md::IntegralEngine<2>::compute1`, see "The overlap gradient
  kernel" above. It did not even need the Cartesian output branch the
  scaffolding was expected to instantiate: the extra unit of bra angular
  momentum is spent inside `E` (`compute2`'s `DA`), never on the shell, so the
  output stays pure and the pure transform commutes with the derivative.
  `dT/dX` and `dV/dX` are the same shape on the same `compute1` interface and
  are independent of the ERI side; `dV/dX` additionally has the
  Hellmann-Feynman term below.

Three smaller things that will otherwise be rediscovered:

- **`Basis<Shell>` carries no atom index**, and neither does
  `Nuclear::Operator::Parameters::centers`. `make_basis` builds from `(Z,r)`
  atoms and drops the mapping. A gradient needs a shell-to-atom map, and two
  shells on one centre must accumulate into the same slot -- a bug the
  translational-invariance check cannot see.
- **A derivative scatter indexed by atom must accumulate, not assign.** A
  quartet with two shells on the same atom hits one accumulator twice, which is
  the normal case in a molecule. This is a second source of the
  wrong-by-a-small-integer-factor symptom, independent of the orbit-weighting
  one described above.
- **`dV/dX` has a Hellmann-Feynman term** -- the operator moves, not just the
  basis functions -- and it needs one more Boys order than `gpu::boys()`
  carries (`A+B+1` reaches `2*LMAX+1` at the top). A finite-difference test
  that displaces only shell centres passes with that term entirely absent.

## The full-ERI formats

`src/libintx/gpu/eri/` is the other end of the tradeoff from the engines above:
instead of contracting integrals as they are produced, it writes the **whole**
four-index tensor out as an `nbf^2 x nbf^2` matrix, so that a Fock build's J and
K each become one GEMV against `vec(D)`.

```
G_J[(mu,nu),(lambda,sigma)] = (mu nu | lambda sigma)      J = G_J . vec(D)
G_K[(mu,nu),(lambda,sigma)] = (mu lambda | nu sigma)      K = G_K . vec(D)
```

with the composite index `munu = mu*nbf + nu`. `gpu::eri::jformat(basis, G,
stream)` and `kformat(...)` fill a device buffer the caller owns and
`format_size(basis)` sizes;
`format_fits()` answers whether it will fit, because `8*nbf^4` bytes is 800 MB
at `nbf=100` and 34 GB at 256. Past a few hundred basis functions this approach
is simply not available and the direct engines are the only option.

Four things worth knowing:

- **Both matrices are symmetric.** `G_J` obviously; `G_K` because its transpose
  is `(lambda mu|sigma nu)`, equal to `(mu lambda|nu sigma)` by the within-pair
  symmetries. So row- and column-major readings agree, `dsymv` applies, and
  neither needs a transposed twin.
- **`G_K` is `G_J` with axes 1 and 2 transposed** —
  `G_K[(mu,nu),(l,s)] = G_J[(mu,l),(nu,s)]` — but it is a second buffer rather
  than a second reading of the first, because the row K needs is scattered
  through `G_J` with stride `nbf` in one index and 1 in another, which is
  exactly what a GEMV cannot express. The permutation is applied where the
  scatter already picks a destination, so it costs nothing.
- **Nothing screens.** A dropped quartet would leave a zero the GEMV cannot
  tell from a real one. The buffer is zeroed and then filled completely, and
  every element is written exactly once, which is why the scatter uses plain
  stores rather than atomics.
- **The eight-fold orbit is the Fock driver's, literally.** `eri/format.h`
  shares `fock::md::make_pair_classes` for the binning and repeats the
  permutation table only because a `__global__` function cannot read a host
  `constexpr` array; a `static_assert` ties the two tables together. Everything
  in "Three things about the shared digest" above applies here verbatim — the
  orbit is enumerated and deduplicated, never weighted — and a result wrong by
  a small integer factor means that, not the integrals.

## Running tests

One suite: doctest executables under `tests/`, registered with `add_test()`,
driven by `ctest`. (`python/tests/` exists but needs the pybind11 bindings,
which are off by default.)

```bash
ctest --preset workstation --output-on-failure     # everything
ctest --preset workstation -L fast                 # the per-change tier
ctest --preset workstation -L slow                 # the angular-momentum sweeps
ctest --preset workstation -R kengine              # one
ctest --preset workstation -N                      # list without running
devtools/cpp-tier.sh                               # configure + build + ctest, logged
```

Only `default`, `workstation`, `debug` and `asan` have **test** presets. With
any other configure preset, `cpp-tier.sh` builds and then reports nothing to
ctest — that is not a pass.

**A green run is bounded by the `LIBINTX_MAX_L` it was configured with.**
`test::enabled(...)` in `tests/test.h` and the `if (LMAX < 3) return;` guard in
the J and K engines' `f` cases skip silently above it. Say which `MAX_L` a run
was, and
say whether the GPU half was in the build at all — on a machine with no card it
was not.

**Every test carries one label — `fast`, `slow` or `gpu` — and a `TIMEOUT`.**
`add_libintx_test` in `tests/CMakeLists.txt` derives both from the test's name:
anything with `.gpu.` in it is `gpu`, the three `libintx.md<n>.test` sweeps are
`slow` (that list is `LIBINTX_SLOW_TESTS`, the one thing to maintain), the rest
is `fast`. So `-L fast` is the per-change loop and `-L slow` is the (bra,ket)
angular-momentum sweeps — 8 s against 41 s of the 49 s full run at
`LIBINTX_MAX_L=2`. The timeouts **scale with the configured `LIBINTX_MAX_L`**
(300 s fast / 1200 s slow at 2, 2400 / 9600 at 3, ×8 per unit of L after that),
because ctest's default 1500 s is *shorter than a legitimate md4 run* at the
default `MAX_L=3` and so reported a real failure as `***Timeout` — see below.
They are backstops for a wedged machine, not budgets; a slow test that trips one
has hung.

**`libintx.md4.test` genuinely fails at `LIBINTX_MAX_L=3`. It does not merely
run long.** A complete run reaches the last combination `(33|33)` and reports 1
failed assertion out of 693,600 (issue #15). This file used to describe that
purely as outrunning the ctest timeout, which implied it passes given time; it
does not. With the timeout above, `***Timeout` on md4 now means a wedged machine
and a `FAIL` means the assertion. At `MAX_L=2` it passes (80,736 assertions),
which is why both CI jobs are green.

**The sweeps' runtime is the test oracle, not libintx.**
`libintx::md::reference::E` in `src/libintx/ao/md/reference.h` — the Hermite
expansion coefficient — is a three-way recursion costing ~1,800 nested calls at
`i=j=3`, driven once per Cartesian axis, per Hermite index, per primitive
quartet; a backtrace of the two-hour run in issue #15 lands in it. It is
memoised now: a small LRU set of fixed-size `(i,j,k)` tables tagged by the
`(a,b,R)` triple, `thread_local`, and **host-only** — the table is inside
`#ifndef __CUDA_ARCH__` because `E` is `LIBINTX_GPU_ENABLED` and a static table
in a `__device__` function does not compile, so the device keeps the plain
recursion. Full ctest at `MAX_L=2` went 94.7 s → 49.4 s, md4 alone 44.0 → 16.8.
At `MAX_L=3` a complete md4 run is **27 min** against the 1 h 56 min in issue
#15 — same verdict, 1 failed assertion of 693,600, and still 8% over ctest's
old 1500 s default, which is why the scaled timeout above is not optional.

**That memoisation is value-preserving but not bit-preserving under `-Ofast`.**
Built `-O2 -march=native`, memoised and unmemoised oracles agree byte-for-byte
over every printed assertion value in `libintx.md{2,3,4}.test` (doctest `-s`).
Under the default Release flags they do not: `-ffast-math` lets the optimiser
reassociate and contract across the recursion while it is inlined into itself,
and it cannot once a call returns a cached value. The *engine* side of every
comparison is unchanged; the oracle side moves by up to ~1e-11 relative, which
is well inside the `1e-9`/`1e-10` tolerances — but it is the same order as the
margin in the two known high-L failures, so do not read a change in *which*
assertion fails at `MAX_L=3` as evidence about the defect.

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

The J and K engine tests. The three K tests share `tests/kengine.test.h` —
dense matrices, the tile callbacks, and the reference three-centre integrals
and Coulomb metric the DF cases need. (The J tests still carry their own copy
of the scaffolding.)

- `tests/libintx.kengine.test.cc` — the host K engine against a brute-force sum
  over *every* shell quartet with no permutational symmetry, built from
  `libintx::md::reference`. It shares no code with the engine, so agreement is
  evidence rather than tautology. Cases: ss / sp / spd / f, contracted shells of
  differing depth, the screened path, K's symmetry, and `AllSum`.
- `tests/libintx.jengine.test.cc` — the same shape for the host J engine, with
  its own brute-force reference. Plus a `fused` case that drives
  `fock::md::build` with both terms in one sweep and checks it against the two
  engines run separately — that is what pins down the "a pack of terms, not
  one" generalization.
- `tests/libintx.gpu.e2.test.cu` — the device `E2` against the host
  `libintx::md::E2`, over all `(A,B)` up to `LMAX` and for both the `DB = 0`
  and the `DB = 2` (kinetic) ket bound. It was the one piece of the one-electron
  device path checkable before any operator kernel existed, and it stays a
  separate test because all three operators are built on it.
- `tests/libintx.gpu.md2.test.cc` — the harness the three operator issues add
  cases to, a direct mirror of `libintx.md2.test`: the same
  `libintx::md::reference::compute2<Op>` sweep over every `(A|B)` and the same
  `{1,1}/{1,5}/{3,5}` contraction sweep, plus a comparison against the **host**
  `md::IntegralEngine<2>` on the same input — the reference pins the values, the
  host engine pins the output layout. All three operators are turned on, so
  what the `scaffolding` case still covers is the engine around them: the
  factory, the one-bin batching invariant, the point-charge upload through
  `set()`, and `Operator::Coulomb` — the one `Operator` with no two-centre
  kernel, and now the only thing that reaches the `(A|B)` fallback table —
  saying so. **That subcase is a semantic conflict between the three operator
  PRs**: git merges three patches each deleting a different `CHECK_THROWS`
  without complaint, leaving a stale line asserting that an implemented
  operator still throws. Read it by hand after any merge. Three
  operator-specific cases sit alongside the sweep:
  `libintx.gpu.md2.Overlap.normalization` is the other half of overlap's check —
  `S == S^T`, and for a shell scaled to unit norm `S == I` against itself; a
  normalization mistake in `S` comes out symmetric, positive definite and
  plausible, and is invisible to the reference sweep because the reference would
  carry the same mistake. `libintx.gpu.md2.Kinetic.transpose` is kinetic's:
  `(A|B)` and `(B|A)` computed as two separate batches and checked to be
  transposes of each other, at every `(A,B)` up to `LMAX` and with the two shell
  families at different contraction depth, plus `T == T^T` within one `(L|L)`
  bin. It references nothing — it is the direct check on the host's transpose
  branch, which the device kernel does not replicate.
  Two cases cover the derivative path.
  `libintx.gpu.md2.Overlap.gradient` is the `(A|B)` sweep against **central
  finite differences** of `libintx::md::reference::compute2<Overlap>` -- a
  five-point stencil at `h = 0.0025`, whose own error is 1.4e-9 relative
  against the case's 1e-7 tolerance, so what the test measures is the kernel
  and not the oracle. `libintx.gpu.md2.Overlap.gradient.translation` references
  nothing: only the bra derivative is computed, and `S(a,b) = S(b,a)` makes the
  bra derivative of the swapped `(B|A)` bin the *ket* derivative of this one, so
  `G_(A|B)[ij,na,nb,x] == -G_(B|A)[ji,nb,na,x]` states `dS/dA + dS/dB = 0`
  elementwise while walking both index orders and both contraction depths. Its
  one-centre subcase is a statement about the caller's scatter, not the kernel:
  `dS/dA` for such a pair is generally *not* zero above `L = 0`, only the sum
  over the two centres is.
  `libintx.gpu.md2.Nuclear.parameters` is everything
  about the point-charge set the `(A|B)` sweep cannot see: `V == V^T`, a second
  `set()` with different centres actually changing the answer (and setting the
  first back reproducing it), a `Z = 0` nucleus contributing nothing, and a
  nucleus placed exactly on a shell's centre — the `T = 0` limit of the Boys
  function, where a naive `1/sqrt(T)` asymptotic form blows up.
- `tests/libintx.gpu.kengine.test.cc` and
  `tests/libintx.gpu.jengine.direct.test.cc` — the device engines against the
  host ones, plus the two Schwarz passes against each other. What they pin down
  is the device engines and the pinned-memory path; the driver itself is pinned
  down by the host tests above. (`tests/libintx.gpu.jengine.test.cc`, without
  `.direct`, is upstream's **DF** J engine test — a different engine.) The
  device DF K engine is checked in the same `gpu.kengine` test, against the
  host DF one.
- `tests/libintx.df.kengine.test.cc` — the host **DF** K engine against the DF
  contraction written out as loops over reference three-centre integrals. Two
  things worth knowing about how it is set up:
  - The metric transform it passes is a **random non-symmetric** matrix, not
    `V^-1`. The engine's contract is "replace X with W X", and a symmetric
    `V^-1` cannot tell a correct application from a transposed one.
  - `exact_fit` is the one case where DF and direct have to agree to
    round-off — see "The density-fitted K build" above. It is what separates
    the DF *mathematics* from the assembly the other cases check; drop one
    auxiliary function from it and it fails by ~1e-3.

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
  Release job this fork adds to gate the host J and K engines. **Neither builds any
  CUDA**, so a change under `src/libintx/gpu/` is checked by nothing
  server-side. `devtools/cpp-tier.sh` with the `default` preset, on a machine
  with a card, is the real gate — know that rather than discover it.
- No linter and no formatter runs anywhere. `clang-format` and `cmake-format`
  are installed in the images and run by hand if at all.

## Fork changes, and where they touch upstream

Keep this list current; it is what a rebase onto upstream has to reconcile.

**New files** — `src/libintx/kengine.h`, `src/libintx/screening.h`,
`src/libintx/fock/md/driver.h`, `src/libintx/ao/md/{jengine,kengine,screening}.{h,cc}`
(`screening.h`/`.cc` and the two engine pairs), `src/libintx/gpu/kengine.h`,
`src/libintx/gpu/screening.h`, `src/libintx/gpu/md/buffer.h`,
`src/libintx/gpu/md/{jengine,kengine,screening}.cc`,
`src/libintx/fock/md/df.h`, `src/libintx/ao/md/df.kengine.cc`,
`src/libintx/gpu/md/df.kengine.cc`,
`src/libintx/gpu/md/e2.h`,
`src/libintx/gpu/onebody/{basis.h,basis.cc,engine.h,kernel.h,md2.cc,CMakeLists.txt}`,
`src/libintx/gpu/overlap/{overlap.h,overlap.cu}`,
`src/libintx/gpu/kinetic/{kinetic.h,kinetic.cu}`,
`src/libintx/gpu/potential_en/{potential_en.h,potential_en.cu}` (none of the
three has a `CMakeLists.txt` of its own -- the sources join `libintx.gpu.md2`,
which is declared in `gpu/onebody/CMakeLists.txt`),
`tests/libintx.{,df.,gpu.}kengine.test.cc`, `tests/kengine.test.h`,
`tests/libintx.jengine.test.cc`,
`tests/libintx.gpu.jengine.direct.test.cc`,
`tests/libintx.gpu.e2.test.cu`, `tests/libintx.gpu.md2.test.cc`
(which now also carries the two `Overlap.gradient` cases),
`src/libintx/gpu/eri.h`, `src/libintx/gpu/eri/{CMakeLists.txt,format.h,eri.cc,
eri.jformat.cu,eri.kformat.cu}`,
`tests/libintx.gpu.eri.{j,k}format.test.cc`,
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
- `src/libintx/gpu/engine.h` — `gpu::IntegralEngine<2>` and its
  `integral_engine<2>` factory declaration, alongside the existing `<3>` and
  `<4>`. Nothing existing changes shape.
- `src/libintx/ao/engine.h` — `ao::IntegralEngine<2>::compute1`, the first
  geometric derivative, plus the `overlap1` convenience wrapper. A new pure
  virtual, so both engines implement it: the device one for
  `Operator::Overlap`, the host one by throwing
  (`src/libintx/ao/md/{engine.h,md2.cc}`).
- `src/libintx/gpu/onebody/kernel.h` — `compute2` gained `DA` (extra bra
  degree; `E2<A+DA,B,DB>`) and `NC` (output components), both defaulted so the
  three value kernels are unchanged, and `block_size` gained `DA`. See "The
  overlap gradient kernel" above.
- `src/libintx/gpu/overlap/{overlap.h,overlap.cu}` — the `overlap1` launcher
  and its `OverlapD1` kernel body, in the same translation unit as the value
  kernel and the same `(LMAX+1)^2` table.
- `src/libintx/gpu/onebody/{engine.h,md2.cc}` — `compute1` on the device
  engine: `Operator::Overlap` to `onebody::overlap1`, everything else throwing.
- `src/libintx/gpu/md/basis.h`, `src/libintx/gpu/md/basis.cu` — `Gaussian2` and
  the device `E2` were file-private inside `basis.cu` and are shared with the
  one-electron engine now: `Gaussian2` moved into `basis.h` next to the other
  device basis PODs, `E2` into the new `gpu/md/e2.h` with its ket bound
  generalised (`E2<A,B,DB>`; `DB = 0` is what `basis.cu` uses, so its behaviour
  is unchanged) and its thread-group contract written down.
- `src/libintx/gpu/CMakeLists.txt` — `add_subdirectory(onebody)`, plus
  `engine.h` added to the installed headers (`onebody/engine.h` includes it).
- `src/libintx/CMakeLists.txt` — `find_library(lapacke)`, see above; plus
  `jengine.h`/`kengine.h`/`screening.h` added to the installed headers.
- `src/libintx/jengine.h` — `JEngine::Screening` now derives from
  `libintx::PairScreening` instead of declaring `max2`/`max`/`skip` itself. An
  existing DF `Screening` implementation is unaffected: same four pure virtuals,
  same names.
- `src/libintx/gpu/jengine.h`, `src/libintx/gpu/jengine/md/jengine.cc` — the DF
  factory is renamed `gpu::make_df_jengine` now that there are two GPU J
  engines to tell apart, with an inline `gpu::make_jengine` forwarding to it so
  no caller breaks. The header also declares the conventional
  `gpu::make_jengine_direct`, so both GPU J engines are visible from one place.
- `src/libintx/ao/md/CMakeLists.txt`, `src/libintx/gpu/md/CMakeLists.txt`,
  `src/libintx/gpu/CMakeLists.txt`, `tests/CMakeLists.txt` — build the new
  sources and tests. `gpu/md`'s also installs `basis.h` and `e2.h` now that
  both are public.
- `src/libintx/gpu/api/api.{h,cc}` — `device::memory_info()`, a wrapper over
  `cuda/hipMemGetInfo` so `gpu::eri::format_fits` can answer whether an
  `nbf^4` buffer will fit before the allocation fails.
- `tests/libintx.gpu.kengine.test.cc` — `(double)` cast on the Schwarz
  comparison. `float == ReferenceValue` is an ambiguous overload; the file had
  never been compiled, because nothing in the environment it was written in had
  CUDA.
- `.github/workflows/ci.yml` — the added Linux job. The macOS job is untouched.
- `README.md` — a short section on the conventional J and K engines, above
  "Using".

The device DF K engine is in `libintx.gpu.md3`, which now links
`libintx.blas` for the contraction — the one new library dependency this fork
adds to the device tree.

Because the device tree did not compile before this fork, **the GPU engines, the
GPU MD signature fix, the `gpu/eri` formats and the one-electron engine have not
been *executed on a device* anywhere** — there is no CUDA toolkit or device in
the environment they were written in. The device translation units under
`gpu/md/` (`jengine.cc`, `kengine.cc`, `df.kengine.cc`, `screening.cc`), both of
`gpu/onebody/` (`basis.cc`, `md2.cc`) and the GPU tests that are plain C++ do at
least compile: they are plain C++ over opaque stream handles, so
`g++ -fsyntax-only` with a hand-written `libintx/gpu/api/config.h` type-checks
them without a toolkit. That is where the `(double)` cast above came from. It is
not a substitute for running them — do
`ctest --preset default -R 'gpu\.(md|md2|e2|kengine|jengine|eri)'` on a machine
with a card before trusting any of it.

The `__device__` headers (`gpu/md/e2.h`, `gpu/onebody/kernel.h`) go through the
same check with a handful of stand-ins for `__device__`, `__shared__`,
`blockIdx` and `cooperative_groups` — worth doing, but a type check, not a
compile: nvcc's shared-memory and launch rules are not exercised by it, and
neither is anything with `<<<...>>>` in it.

**All three one-electron operator kernels, the overlap gradient and
`gpu/eri` are the places with more than that behind them**, and in every case
only for the part that is hardware-independent.

The three operator kernel *bodies* were run on the host: the same `__device__`
shims, extended with a `dim3`/`blockIdx` stub, a fake `cooperative_groups` and
a launch that runs `Block::size()` independent contexts per block with
`__shared__` mapped to function-local `static` and `sync()` to a real barrier —
so E2's parallel recursion, the primitive loop, the Cartesian accumulation, the
cartesian-to-pure pass and the output layout all execute with the thread
structure the device sees. Each `.cu` is compiled verbatim there but for the
one `<<<...>>>` line, so the launch configuration is exactly what is *not*
covered.

Against `libintx::md::reference::compute2<Op>` overlap and kinetic reproduce
every `(A|B)` up to `LMAX = 3` over the `{1,1}/{1,5}/{3,5}` contraction sweep
to 1e-10 (overlap also `S == S^T` and `S == I` for a normalized shell against
itself). The harness is demonstrably looking: transposing one accumulator
index, swapping two axes of E, replacing kinetic's `2*B+3` with `2*A+3`, or its
ket exponent with the bra's each makes it fail, the `2*A+3` and transpose
mutations only on the off-diagonal `(A|B)` — which is the argument for sweeping
both index orders rather than the diagonal.

`overlap1` went through the same shim -- `__shared__` mapped to a
function-local `static`, one real `std::thread` per lane and a `std::barrier`
for `sync()`, blocks run sequentially -- with `overlap.cu` compiled verbatim but
for its two `<<<...>>>` lines. Against the same
`libintx::md::reference::compute2<Overlap>` differentiated by a five-point
central stencil it reproduces `dS/dA_x` for every `(A|B)` up to `LMAX = 3` over
the `{1,1}/{1,5}/{3,5}` contraction sweep to 1.4e-9 relative -- the stencil's own
error -- and `dS/dA + dS/dB = 0` elementwise to 1e-11. The value kernel was run
in the same harness, so the `DA`/`NC` generalisation of the skeleton is checked
against a kernel that predates it: `S` still agrees to 3.2e-13. Five mutations
each break it by O(1) and none of them touches `S`: dropping the `2` in
`2*alpha`, dropping the `-i_x` lowering term, using the ket exponent for the
bra's, transposing the `E` lookup, and writing a component into the wrong slot.
`kinetic.cu` compiles unchanged through the same shim; `potential_en.cu`
instantiates `compute2` there and then fails only on what this particular shim
does not carry -- `atomicAdd` and the device Boys table -- so the skeleton
change is checked against all four kernels built on it.

Kinetic's contexts are ucontext coroutines, round-robin, so every one reaches
its next `sync()` before any runs past it — which *is* the barrier, and is what
makes the full sweep affordable where 128 OS threads on 4 cores is not. That
serialises the code between syncs, so it cannot see a data race a missing
`sync()` would allow; the same harness has a genuinely concurrent `std::thread`
mode (the form the other two kernels were run under), and kinetic was run under
it too, over every `(A|B)` and every `K` at a reduced batch size.

`potential_en.cu` went through its own instance of the same shim, extended with
an `atomicAdd` over `std::atomic_ref` and a host stand-in for `gpu::boys()` (the device
Chebyshev table's `compute` is behind `#if __CUDACC__` and its constructor
uploads to a device; the stand-in forwards to `libintx::boys::chebyshev`, the
same table the reference uses). Against
`libintx::md::reference::compute2<Nuclear>` it reproduces every `(A|B)` up to
`LMAX = 3` over the `{1,1}/{1,5}/{3,5}` contraction sweep to better than 3e-13
relative, including a `Z = 0` nucleus and a nucleus placed exactly on the pair's
centre of charge. Transposing one axis of the `E` lookup, or dropping the
`(-2p)^m` scaling of the Boys values, makes every case fail — which is what says
the harness is looking. As with the other two, the `<<<...>>>` line is the one thing
it does not cover, and `atomicAdd` on shared doubles is a *device* instruction
that the host stand-in only models: it needs compute capability 6.0 or later,
which every architecture the presets target is.

`gpu/eri`'s *combinatorics* — the eight-fold orbit, the
class-pair batching, the same-class triangle rule and the two index layouts —
were checked by running `eri/format.h` itself on the host (the same shim as
above, extended with `__global__`/`dim3` stubs and a launch-grid loop) against a
synthetic integral carrying exactly the ERI's symmetry group: both formats
reproduce a brute-force reference at four batch bounds, and accumulating instead
of assigning gives the same answer, which is what pins "every element is written
exactly once". That still says nothing about the CUDA half — launch
configuration, occupancy, or whether the kernel runs at all.
