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
| `src/libintx/gpu/` | The device half. `api/` wraps CUDA/HIP behind one `gpu::` namespace; `md/` is the device `IntegralEngine<3>`/`<4>` -- values and, through `compute1`, first geometric derivatives -- plus the device K engines (`kengine.cc` direct, `df.kengine.cc` density-fitted) and the device *conventional* J engine; `onebody/` is the device `IntegralEngine<2>` and the pieces its operator kernels share, with `overlap/`, `kinetic/`, `potential_en/` and `coulomb2/` its four operator kernels — the first three each also carrying that operator's first geometric derivative, `dS/dX`, `dT/dX` and `dV/dX`, and `coulomb2/` being the two-centre metric `(P|Q)`, the one two-electron operator on this engine and the one with no derivative yet; `jengine/md/` is the **DF** J engine, a different algorithm in its own target; `eri/` materialises the full ERI tensor. |
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

Every `IntegralEngine<N>` also has a **`compute1`**, the first geometric
derivative: same arguments plus a centre selector (`<2>` has none -- it has
only one independent centre), same output with the Cartesian component as one
more, slowest index. Only the device engines implement it; the host ones
throw. See "Gradients" below.

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
wired, and **every `Operator` has a kernel**: `Operator::Overlap`
(`src/libintx/gpu/overlap/`), `Operator::Kinetic` (`src/libintx/gpu/kinetic/`),
`Operator::Nuclear` (`src/libintx/gpu/potential_en/`) and `Operator::Coulomb`
(`src/libintx/gpu/coulomb2/`). **Nothing reaches the `(A|B)` fallback table any
more.** It stays because that is what a fifth operator lands in before it has a
kernel -- and it throws rather than hand back a buffer of zeros.

Each operator is one line in `gpu/onebody/md2.cc` dispatching to its own
translation unit, which owns the `(A|B)` instantiations its kernel is compiled
into (`gpu/{overlap,kinetic,potential_en,coulomb2}/*.h` are that whole
interface: one non-template launcher taking an uploaded bin). A function
template crossing that boundary would have to be explicitly instantiated over a
table whose size is a configure-time decision, which is why the dispatch is
split in two rather than done once. `block_size<A,B,DB,DA>()` in
`gpu/onebody/kernel.h` is the one piece of that per-operator boilerplate the
one-electron kernels share.

**The three one-electron tables are `(LMAX+1)^2`; the Coulomb one is
`(max(LMAX,XMAX)+1)^2`,** because `(P|Q)` is the density-fitting metric and its
shells come from the *auxiliary* basis, which reaches `XMAX`
(`LIBINTX_MAX_X`, default `LMAX+1`). `compute` therefore dispatches Coulomb
*before* the `L <= LMAX` assertion the other three carry, and `coulomb2.cu`
asserts against `max(LMAX,XMAX)` instead.

Three things the scaffolding settles, which the three one-electron kernels are
the consumers of:

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
all three one-electron operators. It is a separate virtual on
`ao::IntegralEngine<2>` rather than a fifth `Operator` or a defaulted `deriv`
argument, because a derivative is a derivative *of* an operator and because a
default argument on a virtual binds to the static type. The `1` is the
derivative order, not a centre count. Output is the `compute` layout with the
component as one more, slowest index,

```
V[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV],   x = 0,1,2 -> d/dA_x
```

so the `x = 0` block has exactly the shape `compute` writes. **For `S` and `T`
only the bra derivative is computed**: those depend on the centres only through
`r_a - r_b`, so `dS/dB = -dS/dA` elementwise and the caller scatters `+V` onto
the bra shell's atom and `-V` onto the ket shell's -- accumulating, because a
pair with both shells on one atom hits the same slot twice.

**`V` is the exception and it has a second overload**,
`compute1(op, ijs, dV, dVC)`. The electron-nuclear potential depends on the
nuclear positions too, so `dV/dB` is *not* `-dV/dA`; what vanishes is the
three-way sum `dV/dA + dV/dB + sum_C dV/dR_C`. The Hellmann-Feynman term is
indexed by *nucleus*, which does not fit the buffer above, so it goes to a
second one -- `ncenters` consecutive copies of that block, the nucleus as one
more slowest index:

```
dVC[ij + (na + nb*npure(A) + x*npure(A)*npure(B) + c*3*npure(A)*npure(B))*ldV]
```

The 3-argument overload **throws** for `Nuclear` rather than write the shell
half and drop the rest: a `V` gradient without the Hellmann-Feynman term is
smooth, plausible and wrong by the dominant part of the force on a charged
atom. The host `md::IntegralEngine<2>::compute1` throws for everything (there is
no host derivative kernel at all), and the device one throws for Coulomb, which
is not a two-centre operator it implements at all.

**`c` is the atom index, by convention.**
`Nuclear::Operator::Parameters::centers` carries a charge and a position and no
atom index -- the same gap `Basis<Gaussian>` has for shells. Rather than take a
separate map, `centers[i]` *is* atom `i` and `dVC`'s slowest index is that same
`i`; `ao/engine.h` documents it on the struct. A caller that reorders or filters
its atoms on the way in applies the same permutation on the way out, and an atom
carrying both a nucleus and basis functions -- every atom in a normal molecule
-- takes a contribution from `dVC` *and* from the shell derivatives into the
same slot, so the scatter accumulates.

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

**The kinetic gradient kernel** (`kinetic1` in the same `gpu/kinetic/` files)
is `dT/dA_x`, and it needed no new mechanism at all -- it is the overlap
gradient's `DA = 1`/`NC = 3` on top of kinetic's `DB = 2`, i.e.
`compute2<Block,A,B,2,true,1,3>` over an `E2<A+1,B,2>`. Differentiating with
respect to the bra centre does not touch the three-term structure: `T_ab` is a
fixed linear combination of one-dimensional overlap products, `d/dA_x` acts on
the bra Cartesian index alone, so `dT/dA_x` is the same
`b*(2*B+3)*t0 - 2*b^2*t1 - (1/2)*t2` with every factor on axis `x` replaced by
`2*alpha*E^{i+1,j}_0 - i*E^{i-1,j}_0` and the other two axes untouched. Three
things that are easy to get wrong and each have a mutation behind them:

- **`2*B+3` does not move.** It is the *ket* shell's angular momentum, which
  raising the bra index leaves alone. Writing `2*A+3` is invisible on every
  diagonal `(L|L)` -- exactly as it is in the value kernel -- which is why the
  gradient sweep walks both index orders.
- **Every exponent in the body is still the ket's.** `pair.a` appears exactly
  once, in the raising relation; `b`, `2*b^2` and the `j(j-1)` lowering
  coefficients are unchanged.
- **The `+2` and `-2` ket-shifted terms are differentiated too**, not just
  `t0`. The kernel builds, per axis, the three ket degrees `j`, `j+2`, `j-2` as
  both the plain coefficient and its bra derivative -- eighteen `E` lookups
  shared by all three output components -- and then picks derivative on the
  differentiated axis, value on the other two.

**Shared memory grows and the block shape does not.** `E2<A+1,B,2>` is 900
doubles (7,200 B) at `(3|3)` against the value kernel's 648 (5,184 B) -- both
measured, not estimated -- and with the `NC = 3` accumulator (300 doubles,
2,400 B) plus the pure-transform scratch a block is about 10 KB, comfortably
inside `LIBINTX_GPU_MAX_SHMEM` (49152). `block_size<A,B,2,1>()` static_asserts
`A+DA+B+DB+1 <= 32`, which is 10 at `(3|3)`, so the existing 128-thread shape
carries over unchanged.

**The host's transpose branch is not replicated here either**, and a derivative
makes the case against it worse than the value kernel's: the raised index is no
longer symmetric between bra and ket, so a transposing accessor would have to
know which centre was differentiated.
`libintx.gpu.md2.Kinetic.gradient.translation` is the direct check that a swap
has not crept back in.

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
the tree's **one** device Chebyshev table; a second would be a duplicate upload
of the same interpolation data. It is
`boys::gpu::Chebyshev<7, max(4*LMAX, 2*LMAX+XMAX)+1, 117, 117*7>` -- sized for
the *four-centre* Coulomb path, so it spans orders `0 .. max(4*LMAX,
2*LMAX+XMAX)`, which is 13 at `LMAX = 3` against the `A+B <= 2*LMAX = 6` this
kernel needs and the `A+B+1 = 7` its derivative needs. **Nothing had to be
extended for the gradient**, contrary to what issue #33 assumed; the table
exposes `Boys::orders` now so a kernel `static_assert`s its own order against it
rather than reading past the end. `libintx::md::r1::visit`
(`src/libintx/ao/md/r1.h`) is `LIBINTX_GPU_ENABLED` and is literally what the
device Coulomb kernels call, so the recursion cannot drift between the two- and
four-centre paths.

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

**The electron-nuclear gradient kernels** (`potential_en1` in the same files)
are two launches, because `dV/dX` is two different quantities.

`NuclearD1` is the bra half, `dV/dA_x`. The nuclear attraction is linear in the
bra primitive, so the same raising relation the overlap gradient uses passes
straight through it -- the raised and lowered integrals share `p`, `P`, `PC` and
the `R` tensor, and only `E` changes:

```
dV_ab/dA_x = (2*pi/p) * sum_t [ 2a*E^{a+1_x,b}_t - a_x*E^{a-1_x,b}_t ] * R_t
```

so it is `compute2<...,DA=1,NC=3>` with `E2<A+1,B,0>`, exactly like `overlap1`.

`NuclearD1C` is the Hellmann-Feynman half, `dV/dR_C,x` -- the term no other
one-electron operator here has, because `1/|r - R_C|` depends on the nuclear
position directly. `R_t(PC)` is the `t`-fold derivative of the Boys kernel with
respect to `PC`, so differentiating it raises the Hermite index by one and
`PC = P - R_C` supplies the sign:

```
dV_ab/dR_C,x = -(2*pi/p) * sum_t E^{ab}_t * R^C_{t+1_x}
```

Three things worth knowing:

- **The extra Boys order is not the Hellmann-Feynman term's alone.** `a+1_x`
  reaches Hermite degree `A+B+1` too, so *both* halves want `R` over
  `nherm2(A+B+1)` and `r1::visit<A+B+1>` over Boys values through `m = A+B+1`.
  Both `static_assert(A+B+1 < Boys::orders)`; see the table sizing above.
- **The nucleus is `blockIdx.y`.** One block per (pair, nucleus), so the
  per-nucleus output never has to live in shared memory -- `ncenters*3*ncart(A)
  *ncart(B)` accumulators is not a shared-memory shape for a real molecule --
  and the grid is `ncenters` times wider, which is the axis that fills a device
  when a batch is small. The price is `E` rebuilt once per nucleus rather than
  once per pair, and `r1::visit` on thread 0 while the block waits (there is one
  nucleus per block and the recursion does not divide). **None of this has been
  benchmarked.** The bra half keeps the value kernel's shape: nuclei across the
  block, `R` reduced by shared-memory `atomicAdd`.
- **A `Z = 0` nucleus exits the block early**, uniformly, and its output block
  is zeros -- which is the answer, not a skipped one.

**The two-centre Coulomb kernel** (`src/libintx/gpu/coulomb2/`) is the metric
`(P|Q)` a density-fitted J or K build needs before it can hand
`gpu::make_df_jengine` or `gpu::make_df_kengine` a `V^-1`. Until it landed there
was no device route to `V` at all -- `tests/kengine.test.h` builds it from the
host four-centre reference. Per primitive pair (`a` on `r_A`, `b` on `r_B`):

```
alpha = a*b/(a+b),  PQ = r_A - r_B
s[m]  = F_m(alpha*|PQ|^2) * (-2*alpha)^m,   m = 0..A+B
R[t]  = r1(PQ, s)[t]
(a|b) = C_a*C_b * 2*pi^(5/2)/(a*b*sqrt(a+b))
        * sum_{t<=A} sum_{u<=B} E^A_t E^B_u (-1)^|u| R[t+u]
```

It is the one **two-electron** operator on a two-*centre* engine, and that is
what makes it structurally different from the other three rather than a fourth
copy of them:

- **It does not use `gpu/onebody/kernel.h`'s `compute2`, deliberately.** That
  skeleton is written against a *product density*: it hands the operator body
  one `E2<A+DA,B,DB>` built from `(a, b, r_A-r_B)` and a coefficient
  `C = C_a*C_b*exp(-a*b/(a+b)*|r_A-r_B|^2)`. `P` and `Q` sit on opposite sides
  of `1/r12`, so what is wanted is two *one-centre* expansions -- `E2<A,0,0>` at
  `(a, 0)` and `E2<B,0,0>` at `(b, 0)`, which is exactly what the reference gets
  by putting a `Unit` shell in each ket slot -- and no `K_ab` at all. Recovering
  `C_a*C_b` by dividing the skeleton's `C` back out is not an option: for a well
  separated, sharply contracted pair `K_ab` underflows to zero and the division
  is `0/0`. So `coulomb2.cu` carries its own block driver and shares what is
  about the *engine* rather than the operator -- `GaussianPairs`,
  `block_size<A,B,DB>()`, `uindex<A,B>` and the output layout. The ~15
  duplicated lines are the batch load and the cartesian-to-pure store.
- **The two one-centre `E2`s are cheaper in shared memory than the one product
  `E2` would have been** -- `2*3*(A+1)^2` doubles against `3*(A+1)(B+1)(A+B+1)`,
  150 against 2025 at `(4|4)`.
- **`gpu::boys()` covers it, but only just.** The table is sized
  `max(4*LMAX, 2*LMAX+XMAX)`; `(P|Q)` needs order `2*max(LMAX,XMAX)`, which
  holds iff `XMAX <= 2*LMAX` -- true for the default `XMAX = LMAX+1` at every
  `LMAX >= 1`. A `static_assert` in the kernel says so rather than letting an
  exotic `LIBINTX_MAX_X` read off the end of the table.
- **The Boys evaluation and `r1::visit` run on thread 0.** Unlike the nuclear
  kernel there is no third axis to spread them over: a nuclear potential has
  O(10-100) nuclei per primitive pair, a metric element has exactly one
  `(PQ, alpha)`, and every thread computing it redundantly would take the same
  wall clock. The Hermite recursion and the Cartesian contraction are
  parallel. **Nothing here has been benchmarked** -- there is no device in the
  environment it was written in.
- **`(P|Q)` grows with the coefficients on both sides of `1/r12` and has no
  overlap factor to damp it**, so with `test::gaussian`'s raw `C = 1/a` the
  entries reach 1e6 and the tree's absolute-ish `max(|a|,|b|,1)*epsilon`
  comparison sits below the double-precision noise floor. The test uses a
  primitive-normalized auxiliary basis, which is what a real one is. Measured
  on the host shim harness at `LMAX=3, XMAX=4`: normalized, worst error 4e-12
  over the whole sweep; unnormalized, 2.4e-8 absolute, which is 1.5e-11
  relative to the block maximum.

**The derivative half of issue #28 is not here.** `d(P|Q)/dX` is deferred
pending #27, which decides the route for derivative ERI batches; there is
deliberately no second derivative mechanism in the tree.

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

## Gradients: the derivative integrals, and where the scope line is

The **integral** layer has its first derivatives on the device now -- `dS/dX`
on `gpu::md::IntegralEngine<2>` and `d(ab|cd)/dX`, `d(P|cd)/dX` on
`<4>`/`<3>`, all through `compute1`. Nothing above that layer exists: no
derivative J or K, no gradient driver, no host derivative kernel of any kind.
Analytic gradients are tracked as a DAG of issues; what matters for anyone
starting on them is that libintx ships **per-term derivatives** and never a
total force:

```
dE/dX = sum D_uv d(T+V)_uv/dX + [J and K] - sum W_uv dS_uv/dX + dE_nn/dX
```

**Nuclear repulsion and gradient assembly are the caller's**, deliberately.
`E_nn` involves no integral, and assembly needs `D` and the energy-weighted
density `W` that only an SCF has. Neither is missing by accident; see README's
"Gradients: what libintx owns, and what the caller does".

Two structural facts, both verified against the code, that shape all of it:

- **The one-electron skeleton has no pure-only problem**, and all three of
  `dS/dX`, `dT/dX` and `dV/dX` are done -- `gpu::md::IntegralEngine<2>::compute1`,
  see "The overlap gradient kernel", "The kinetic gradient kernel" and "The
  electron-nuclear gradient kernels" above. **The one-electron half of a total
  SCF gradient is therefore complete**, and none of it needed the Cartesian
  output branch the scaffolding was expected to instantiate: the extra unit of
  bra angular momentum is spent inside `E` (`compute2`'s `DA`), never on the
  shell, so the output stays pure and the pure transform commutes with the
  derivative.
- **The device Coulomb kernels *are* pure-only** -- `gpu/md/basis.cu` bakes
  the solid-harmonic transform into the batch and `gpu/md/md.kernel.h`
  contracts against an `npure(A)*npure(B)`-wide transform fixed at compile
  time, so `T_pure(L)*(a+1_x)` has nowhere to live -- **and the derivative
  batch is how that was got around**, see "The derivative ERI batches" below.
  The kernel is generic in that transform (it does not know it is applying a
  pure one), so the derivative is baked into that slot instead: no Cartesian
  output path was added and no shell is shifted. Which is the same answer the
  one-electron side reached independently, by a different route.

Three smaller things that will otherwise be rediscovered:

- **`Basis<Shell>` carries no atom index.** `make_basis` builds from `(Z,r)`
  atoms and drops the mapping. A gradient needs a shell-to-atom map, and two
  shells on one centre must accumulate into the same slot -- a bug the
  translational-invariance check cannot see. (The *nucleus* side of this is
  settled: `Nuclear::Operator::Parameters::centers[i]` is atom `i`, and
  `compute1`'s `dVC` is indexed by that same `i`. See "The device one-electron
  engine" above.)
- **A derivative scatter indexed by atom must accumulate, not assign.** A
  quartet with two shells on the same atom hits one accumulator twice, which is
  the normal case in a molecule. This is a second source of the
  wrong-by-a-small-integer-factor symptom, independent of the orbit-weighting
  one described above.
- **`dV/dX` has a Hellmann-Feynman term** -- the operator moves, not just the
  basis functions -- and a finite-difference test that displaces only shell
  centres passes with that term entirely absent. It costs one more Boys order
  (`A+B+1`, reaching `2*LMAX+1` at the top), which `gpu::boys()` already
  carried before the derivative ERI batches widened the table further; it was
  **not** extended for `dV/dX`, contrary to what this file used to say.

### The derivative ERI batches

`gpu::md::IntegralEngine<4>::compute1(Operator, centre, bra, ket, norms, V,
dims)` is `d(ab|cd)/dX` for `centre` 0..3 (a, b, c, d);
`IntegralEngine<3>::compute1` is `d(P|cd)/dX` for `centre` 1 (c) or 2 (d).
Layout is `compute`'s with the Cartesian component as one more, slowest index
-- three back-to-back blocks of exactly the shape and stride `compute` writes,
so component `x` starts at `V + x*dims[0]*dims[1]` -- which is what
`ao::IntegralEngine<2>::compute1` does for the one-electron case, one index
shorter. **The host overrides throw**; they are declared so that one interface
still drives both engines, and they throw rather than return zeros because a
zero derivative reads as a converged gradient.

**Route A: the derivative goes in the batch, not in the shell.** The
identity is the same one the overlap gradient uses,
`d/dA_x G_a = 2a G_{a+1_x} - i_x G_{a-1_x}`, but the McMurchie-Davidson
expansion lets it be applied *once per shell pair* rather than per quartet:
the Hermite coefficient block of the pair carries

```
D^{ab,x}_t = 2a E^{(a+1_x)b}_t - i_x E^{(a-1_x)b}_t        (centre a)
D^{ab,x}_t = 2b E^{a(b+1_x)}_t - j_x E^{a(b-1_x)}_t        (centre b)
```

in place of `E^{ab}_t`, and *nothing else about the batch changes* -- same
`Hermite` header, so the same `p`, `P`, `C` and `K_ab`; same ket coefficients;
same `R` tensor; same `(-1)^|u|` phase; same `npure(A)*npure(B)` output. The
solid-harmonic transform is applied to `D` exactly as it is to `E`, because it
is linear with constant coefficients and so commutes with `d/dX`. That is
`gpu::md::make_basis1` (`gpu/md/basis.cu`), which is the value `make_basis`
kernel with one extra template parameter; the two share the E2 recursion, the
primitive loop, the two `cartesian_to_pure` passes and the `[ab,p]` layout.

Route B -- a Cartesian output path through `md.kernel.h`, `md3.kernel.h`,
`md4.kernel.h` and `basis.cu` plus shifted `L+1` shells -- was not taken. It is
strictly more code and it re-opens the re-normalization trap (`gto::normalized`
carries an `L`-dependent factor, so an `L+1` copy of a shell is a silently
different basis), and the only thing it buys is a Cartesian ERI path that
nothing in the tree wants.

**What Route A costs is one unit of L-*sum*.** Differentiating raises the
Hermite index by one while the output stays shaped for the pair, so the two
must be decoupled:

- `Basis2` gained `dL` (`gpu/md/basis.h`), the extra Hermite degree its data
  carries; `Hermite::extent` takes it. `kernel::Basis2<L>` already had `L` and
  `nbf` as independent parameters, which is why the kernels needed no change
  of shape -- only instantiating at `A+B+1`.
- `compute1` has **its own dispatch tables**, indexed by the batches' Hermite
  L-sums rather than the shell pairs'. `compute`'s single
  `ab_cd_kernels[bra.L][ket.L]` cannot express the decoupling; that is the
  coupling the issue said had to be broken, and this is where.
- `hermite::orbitals2<2*LMAX>` became `<2*LMAX+1>` in `gpu/md/basis.cu` and
  `gpu/md/md.kernel.h`, and `gpu::boys()`'s table gained one order.
- The kernel translation-unit grid runs to `2*LMAX+1` on each side (md4) and on
  the ket side (md3). At `LIBINTX_MAX_L=3` that is **63 md4 TUs, up from 49**
  (the `(2L+1, 2L+1)` corner carries nothing and is skipped) and **40 md3 TUs,
  up from 35**; at `MAX_L=2`, 35 from 25 and 24 from 20. Raising
  `LIBINTX_MAX_L` instead would have taken md4 to 81 *and* recompiled every
  existing kernel at higher L; this touches no existing instantiation.
- **The derivative path uses only the generic v2 route** -- one kernel over the
  Hermite indices plus two GEMMs against the coefficient blocks. v0, v1 and
  v2's `p_cd` branch all reconstruct a top Hermite block in closed form from
  `inv_2_exp` and `pure_transform`, which is the value `E`'s identity and not
  the derivative's; a derivative batch therefore leaves `pure_transform` null
  and takes the path that reads every degree out of the batch. Making the
  faster kernels available to a bra-derivative batch is a performance question,
  not a correctness one, and nobody has measured it.

Two things it does not do. **`d/dP` for the three-centre bra is not
computed** -- the auxiliary shell is a single Hermite bra whose transform is
the analytic `hermite_to_cartesian` + `cartesian_to_pure` written into
`md3.kernel.h`, not a per-batch coefficient block, so there is no slot to bake
`dE/dP` into. It is also unnecessary: `(P|cd)` depends on the three centres
only through their differences, so `d/dP = -(d/dC + d/dD)`, which
`libintx.gpu.deriv.test` checks against a finite difference rather than
assuming. And **the four-centre engine computes all four centres** rather than
taking the fourth from `sum_centres d/dX = 0`, so that identity stays available
as a test.

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
  host engine pins the output layout. All three are turned on, so what the
  `scaffolding` case still covers is the engine around them: the factory, the
  one-bin batching invariant, the point-charge upload through `set()`, and —
  now that `Operator::Coulomb` has a kernel too and **nothing reaches the
  `(A|B)` fallback table any more** — that every `Operator` is dispatched to
  one and none of them returns the buffer untouched. **That subcase is a
  semantic conflict between the operator PRs**: git merges patches each
  deleting a different `CHECK_THROWS` without complaint, leaving a stale line
  asserting that an implemented operator still throws. Read it by hand after
  any merge. Operator-specific cases sit alongside the sweep:
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
  **`libintx.gpu.md2.Coulomb` is not a `LIBINTX_GPU_MD2_TEST_CASE` line and
  cannot be**: `libintx::md::reference::Integral<Op>` is specialized for
  Overlap, Kinetic and Nuclear only, so `reference::compute2<Coulomb>` does not
  compile, and the host `md::IntegralEngine<2>::compute` has no Coulomb branch
  at all — it leaves the buffer untouched, so a comparison against it would be
  a comparison against zeros. It carries its own oracle,
  `reference::compute(P, Unit, Q, Unit, ...)`, the same four-centre-with-unit-
  kets construction `tests/kengine.test.h`'s `reference_metric` already uses;
  it sweeps to `max(LMAX,XMAX)` rather than `LMAX`, because
  `test::enabled(A,B)` is the wrong guard for an auxiliary-basis operator; and
  its shells are primitive-normalized (see "The two-centre Coulomb kernel").
  `libintx.gpu.md2.Coulomb.metric` is what the sweep cannot see and what a DF
  caller depends on: over a whole auxiliary basis assembled bin by bin,
  `V == V^T` (which is also the only layout check available, there being no
  host Coulomb to compare a layout against) and `V` positive definite by
  Cholesky — a symmetric-but-wrong metric is exactly what a DF engine cannot
  detect.
  Four cases cover the derivative path, and they are **two shapes written once
  over the operator** -- `gradient_sweep(Op, ...)` and
  `gradient_translation(Op, ...)` in the file's anonymous namespace, each
  driven by an `Overlap` and a `Kinetic` `TEST_CASE`. Only the kernels differ
  between the two operators; what a derivative has to satisfy does not, and a
  second copy of it would be a second thing to keep in step.
  `libintx.gpu.md2.{Overlap,Kinetic}.gradient` is the `(A|B)` sweep against
  **central finite differences** of `libintx::md::reference::compute2<Op>` -- a
  five-point stencil at `h = 0.0025`, whose own error is 1.4e-9 relative for
  both operators against the case's 1e-7 tolerance, so what the test measures
  is the kernel and not the oracle.
  `libintx.gpu.md2.{Overlap,Kinetic}.gradient.translation` references
  nothing: only the bra derivative is computed, and both operators are
  symmetric, so the bra derivative of the swapped `(B|A)` bin is the *ket*
  derivative of this one and
  `G_(A|B)[ij,na,nb,x] == -G_(B|A)[ji,nb,na,x]` states `dO/dA + dO/dB = 0`
  elementwise while walking both index orders and both contraction depths. The
  kinetic one is also the derivative analogue of `Kinetic.transpose` above, and
  is what would catch the host's swapped `kinetic<B,A>` branch appearing in the
  device path. Their
  one-centre subcase is a statement about the caller's scatter, not the kernel:
  `dO/dA` for such a pair is generally *not* zero above `L = 0`, only the sum
  over the two centres is.
  `libintx.gpu.md2.Nuclear.parameters` is everything
  about the point-charge set the `(A|B)` sweep cannot see: `V == V^T`, a second
  `set()` with different centres actually changing the answer (and setting the
  first back reproducing it), a `Z = 0` nucleus contributing nothing, and a
  nucleus placed exactly on a shell's centre — the `T = 0` limit of the Boys
  function, where a naive `1/sqrt(T)` asymptotic form blows up.
  Three more cover `dV/dX`, and the thing they are built around is that
  **the finite differences displace the nuclei as well as the bra centre** — a
  sweep over shell centres alone passes with the Hellmann-Feynman term entirely
  absent. `libintx.gpu.md2.Nuclear.gradient` is the `(A|B)` × `{1,1}/{1,5}/{3,5}`
  sweep against the five-point stencil, per Cartesian component *and per
  nucleus*. `libintx.gpu.md2.Nuclear.gradient.translation` is the three-way sum
  `dV/dA + dV/dB + sum_C dV/dR_C = 0` elementwise, referencing nothing, with
  `dV/dB` taken from the transposed bin the way the overlap case takes it — it
  is the check that pins the two contributions against each other, and the
  tolerance is relative to what cancels rather than absolute.
  `libintx.gpu.md2.Nuclear.gradient.parameters` is the rest: a `Z = 0` nucleus
  with an identically zero block (and no effect on the others), a charged
  nucleus exactly on a shell centre against the stencil, and shells *and*
  nucleus all on one centre where every set is separately zero — which, unlike
  the translational sum, does not let a sign error between the two
  contributions hide. The helpers all three share (`displaced`,
  `value_reference`, `bra_gradient_reference`, `gradient`) are templated on the
  operator; `nuclear_gradient_reference` and `nuclear_gradient` are the two that
  are not, because only this operator moves with its operator.
- `tests/libintx.gpu.kengine.test.cc` and
  `tests/libintx.gpu.jengine.direct.test.cc` — the device engines against the
  host ones, plus the two Schwarz passes against each other. What they pin down
  is the device engines and the pinned-memory path; the driver itself is pinned
  down by the host tests above. (`tests/libintx.gpu.jengine.test.cc`, without
  `.direct`, is upstream's **DF** J engine test — a different engine.) The
  device DF K engine is checked in the same `gpu.kengine` test, against the
  host DF one.
- `tests/libintx.gpu.deriv.test.cc` — the derivative ERI batches,
  `IntegralEngine<4>::compute1` and `<3>::compute1`, **against the engines'
  own value path**. It references nothing else: the oracle is `compute` re-run
  with one shell centre displaced, differenced with the same five-point
  stencil at `h = 0.0025` `libintx.gpu.md2.test` uses (a two-point difference
  leaves ~1e-6 and cannot see a wrong `2*alpha`). Three cases beyond the
  finite differences: elementwise translational invariance
  (`sum_centres d(ab|cd)/dX = 0`), which references nothing at all and catches
  component mixing far more cheaply than a finite difference; `d/dP` for the
  three-centre bra checked as `-(d/dC + d/dD)` against a finite difference in
  `P`, which is what makes not computing it legitimate; and an `interface`
  case pinning what throws -- an out-of-range centre, md3's auxiliary centre,
  and both host engines. The four-index sweep is `(LMAX+1)^4` bins and each
  runs 12 analytic plus 48 value batches, so the contraction sweep
  (`{1,5}`/`{3,5}`) is a separate, smaller case rather than crossed with it.
  Mutation coverage is not expressible in a test that links the kernel; the
  derivative coefficient is mutation-tested in the host shim harness (see the
  end of this file).
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
`src/libintx/gpu/potential_en/{potential_en.h,potential_en.cu}`,
`src/libintx/gpu/coulomb2/{coulomb2.h,coulomb2.cu}` (none of the four has a
`CMakeLists.txt` of its own -- the sources join `libintx.gpu.md2`, which is
declared in `gpu/onebody/CMakeLists.txt`),
`tests/libintx.{,df.,gpu.}kengine.test.cc`, `tests/kengine.test.h`,
`tests/libintx.jengine.test.cc`,
`tests/libintx.gpu.jengine.direct.test.cc`,
`tests/libintx.gpu.e2.test.cu`, `tests/libintx.gpu.md2.test.cc`
(which now also carries the gradient cases: `gradient_sweep` and
`gradient_translation` written once over the operator and driven by
`{Overlap,Kinetic}.gradient{,.translation}`, plus the three `Nuclear.gradient`
ones, which need their own because that operator moves too),
`tests/libintx.gpu.deriv.test.cc`,
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
  geometric derivative, plus the `overlap1` convenience wrapper; and a second
  overload `compute1(op, ijs, dV, dVC)` for the operator that moves with the
  nuclei, `Operator::Nuclear`. Two new pure virtuals, so both engines implement
  both: the device one for `Overlap`/`Kinetic` (3-arg) and `Nuclear` (4-arg),
  the host one by throwing (`src/libintx/ao/md/{engine.h,md2.cc}`). The
  `Nuclear::Operator::Parameters` doc comment now states the `centers[i]` =
  atom `i` convention the Hellmann-Feynman buffer's nucleus index is defined
  against. Then `compute1` on `ao::IntegralEngine<3>` and `<4>` too, with a
  centre selector instead -- above two centres there is no `d/dA = -d/dB` to
  collapse them; the host overrides throw
  (`src/libintx/ao/md/{engine.h,md3.cc,md4.cc}`).
- `src/libintx/gpu/md/{basis.h,basis.cu}` (again), `md.kernel.h`,
  `{md3,md4}.cc`, `{md3,md4}.kernel.cu`, `engine.h`, `CMakeLists.txt` and
  `src/libintx/gpu/boys.h` — the derivative ERI batches. `make_basis` gained a
  `Deriv` template parameter and `make_basis1` its host launcher; `Basis2`
  gained `dL`; `compute_v2` gained `DA`/`DC`; `compute1` and its own dispatch
  tables are new; `orbitals2` and the Boys table each gained one order; the
  kernel TU grid runs one unit of L-sum further. See "The derivative ERI
  batches".
- `src/libintx/gpu/onebody/kernel.h` — `compute2` gained `DA` (extra bra
  degree; `E2<A+DA,B,DB>`) and `NC` (output components), both defaulted so the
  three value kernels are unchanged, and `block_size` gained `DA`. See "The
  overlap gradient kernel" above.
- `src/libintx/gpu/overlap/{overlap.h,overlap.cu}` — the `overlap1` launcher
  and its `OverlapD1` kernel body, in the same translation unit as the value
  kernel and the same `(LMAX+1)^2` table.
- `src/libintx/gpu/kinetic/{kinetic.h,kinetic.cu}` — the `kinetic1` launcher
  and its `KineticD1` kernel body, in the same translation unit as the value
  kernel and the same `(LMAX+1)^2` table. See "The kinetic gradient kernel"
  above.
- `src/libintx/gpu/potential_en/{potential_en.h,potential_en.cu}` — the
  `potential_en1` launcher and its two kernel bodies, `NuclearD1` (bra) and
  `NuclearD1C` (Hellmann-Feynman), in the same translation unit as the value
  kernel. The value body's Hermite contraction moved into a shared
  `contract(E, m, shift, R)` the three now use; `shift` is what lets the
  Hellmann-Feynman half read `R_{t+1_x}` without a second tensor.
- `src/libintx/boys/gpu/chebyshev.h` — `static constexpr int orders = M`, so a
  kernel can `static_assert` its Boys order against the table instead of reading
  past the end of it. No table was resized.
- `src/libintx/gpu/onebody/{engine.h,md2.cc}` — `compute1` on the device
  engine: `Operator::Overlap` to `onebody::overlap1` and `Operator::Nuclear` to
  `onebody::potential_en1` on the 4-argument overload, everything else
  throwing — including `Nuclear` on the 3-argument overload, with an error that
  names the one to use instead.
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
  "Using"; and "Gradients: what libintx owns, and what the caller does", which
  names the two terms that exist (`dS/dX`, `dV/dX`) and the two pieces that are
  deliberately outside the library.

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

**All four two-centre operator kernels, all three of the one-electron
gradients, `gpu/md/basis.cu` (value and derivative) and `gpu/eri` are the
places with more than that behind them**, and in every case only for the part
that is hardware-independent.

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
`kinetic.cu` and `potential_en.cu` go through the same shim too -- the latter
needing two stand-ins the overlap harness does not, `atomicAdd` over
`std::atomic_ref` and a host `gpu::boys()` -- so the skeleton change is checked
against every kernel built on it.

`kinetic1` went through the same shim, in the `std::thread` form, with
`kinetic.cu` compiled verbatim but for its two `<<<...>>>` lines. Against
`libintx::md::reference::compute2<Kinetic>` differentiated by the same
five-point central stencil it reproduces `dT/dA_x` for every `(A|B)` up to
`LMAX = 3` over the `{1,1}/{1,5}/{3,5}` contraction sweep to 1.46e-9 relative --
the stencil's own error -- with `dT/dA + dT/dB = 0` elementwise to 1e-11 and
the one-centre pair checked against the finite differences rather than assumed
zero. The value kernel ran in the same harness and still agrees to 8.4e-13, so
the shared skeleton is re-checked against the kernel that predates the
derivative. `sizeof` in that harness is where the shared-memory figures above
come from: `E2<3,3,2>` is 5,184 B and `E2<4,3,2>` 7,200 B.

Ten mutations each break it by O(1): the four the issue names -- `2*B+3`
replaced by `2*A+3` and the ket exponent replaced by the bra's, applied to the
value body *and* to the derivative body -- plus dropping the `2` in `2*alpha`,
dropping the `-i_x` lowering term, transposing the `E` lookup, writing a
component into the wrong slot, differentiating every axis rather than one, and
differentiating only `t0` and leaving the `+2`/`-2` terms at their values. Two
of them are the reason the sweep walks both index orders: `2*A+3` fails on
exactly the 36 off-diagonal `(A|B)` combinations and passes all 12 diagonal
ones, on the value and derivative bodies alike. Dropping the lowering term
passes only at `(0|0)`, where there is no bra index to lower.

In that first, value-only run, kinetic's contexts are ucontext coroutines,
round-robin, so every one reaches
its next `sync()` before any runs past it — which *is* the barrier, and is what
makes the full sweep affordable where 128 OS threads on 4 cores is not. That
serialises the code between syncs, so it cannot see a data race a missing
`sync()` would allow; the same harness has a genuinely concurrent `std::thread`
mode (the form the other two kernels were run under), and kinetic was run under
it too, over every `(A|B)` and every `K` at a reduced batch size.

`potential_en.cu` went through its own instance of the same shim, extended with
an `atomicAdd` over `std::atomic_ref`, a host stand-in for `gpu::boys()` (the device
Chebyshev table's `compute` is behind `#if __CUDACC__` and its constructor
uploads to a device; the stand-in forwards to `libintx::boys::Chebyshev` at the
same `Order`/`MaxT`/`Segments`, so the interpolation coefficients are literally
the same numbers, while reporting the *device* table's `orders` so the kernels'
`static_assert`s test the real sizing) and a `grid.y` loop, since the
Hellmann-Feynman kernel puts the nucleus there. Against
`libintx::md::reference::compute2<Nuclear>` the **value** kernel reproduces every
`(A|B)` up to `LMAX = 3` over the `{1,1}/{1,5}/{3,5}` contraction sweep to
7.2e-13 relative, including a `Z = 0` nucleus and a nucleus placed exactly on
the pair's centre of charge.

Both **gradient** kernels went through the same harness in the same run, against
five-point central differences of that same reference taken **on the nuclei as
well as on the bra centre**: `dV/dA` to 3.6e-9 and `dV/dR_C` to 3.2e-8 relative
-- the stencil's own error, against a 1e-7 tolerance -- over the same sweep at
`LMAX = 3`, plus the three-way `dV/dA + dV/dB + sum_C dV/dR_C = 0` to 8.1e-14
relative to what cancels, a `Z = 0` nucleus contributing exactly zero, a nucleus
sitting on a shell centre, and shells-and-nucleus on one centre where every set
is separately zero. Nine mutations each break it and none of them touches
anything the others do not: dropping the Hellmann-Feynman term outright,
flipping its sign, shifting its Hermite index on the wrong axis, dropping the
`(-2p)^m` Boys scaling, transposing the `E` lookup, dropping the `2` in
`2*alpha`, dropping the `-i_x` lowering term, using the ket exponent for the
bra's, and building `R` at the value kernel's Hermite degree instead of one
higher. The first three leave `dV/dA` and `V` exact and break only `dV/dR_C` and
the sum -- which is the point: **a sweep that displaced only the shell centres
would pass all three.** As with the other kernels the `<<<...>>>` lines are what
this does not cover, and `atomicAdd` on shared doubles is a *device* instruction
the host stand-in only models: it needs compute capability 6.0 or later, which
every architecture the presets target is.

`coulomb2.cu` went through the same shim again -- the `atomicAdd` stand-in and
the same `gpu::boys()` forwarding to `libintx::boys::chebyshev`, with a
diagnostic switch to `boys::Reference` to separate interpolation error from the
kernel's own. Against `libintx::md::reference::compute(P, Unit, Q, Unit, ...)`
it reproduces every `(A|B)` up to `max(LMAX,XMAX) = 4` over the
`{1,1}/{1,5}/{3,5}` contraction sweep to 4e-12 on a primitive-normalized
auxiliary basis (2.4e-8 absolute on `test::gaussian`'s raw coefficients, which
is 1.5e-11 relative to the block maximum -- see "The two-centre Coulomb
kernel"), including a pair with both shells on one centre, which is the `T = 0`
limit of the Boys function and the diagonal of the metric. `(P|Q) == (Q|P)`
elementwise across the two bins, and the metric of a 75-function auxiliary
basis assembled bin by bin is symmetric to 3.5e-12 with eigenvalues in
[8.4e-2, 5.7e3] -- Cholesky succeeds. Nine mutations each break it: dropping
the `(-2*alpha)^m` Boys scaling, transposing the bra or the ket `E` lookup,
dropping the `(-1)^|u|` ket phase, using the product exponent for the reduced
one, building the ket expansion on the bra's exponent, mis-indexing `R` by
axis, swapping the pre-factor's `sqrt(a+b)`, and transposing the Cartesian
accumulator. As with the other three, the `<<<...>>>` line is what it does not
cover.

`gpu/md/basis.cu`'s `make_basis` kernel -- the value batch and both derivative
centres -- went through the same shim, extended with `threadIdx`/`blockDim`
(basis.cu's pure transform is written over a 2-D block, not over `thread_rank`
alone) and a launcher that maps rank to `(x,y,z)` the way CUDA does. Against
`libintx::md::reference::E` plus `libintx::pure::reference::transform` it
reproduces every coefficient of every batch -- all `(A|B)` up to `LMAX = 3`,
the `{1,1}/{1,5}/{3,5}` contraction sweep, both centres, all three components,
every Hermite index to `nherm2(A+B+1)` -- to 8.7e-15 relative, and the value
batch (which predates the `Deriv` generalisation) to 5.3e-15. Six mutations of
the kernel's derivative coefficient each break it by O(1) and none of them
moves the value batch: dropping the `2` in `2*alpha`, dropping the `-i_x`
lowering term, differentiating the wrong Cartesian component, using the ket
exponent for the bra's, transposing the raised `E` lookup, and reading the
unraised index. **Separately**, Route A itself -- that
`D^{ab,x}_t` contracted the ordinary way *is* `d(ab|cd)/dX` -- was checked as
mathematics on the host against a five-point central difference of
`libintx::md::reference::compute`, over every `(A,B|C,D)` up to `LMAX = 2`, all
four centres, both `{1,1}` and `{1,3}`: agreement to the stencil's own
truncation error (the one element outside 2e-7 grows as `h^4`, confirmed by
halving and doubling `h`), and elementwise translational invariance to 1e-11.

**What is *not* checked anywhere for the derivative ERI batches** is the
contraction itself -- `IntegralEngine<4>::compute_v2`'s and
`IntegralEngine<3>::compute_v2`'s kernel-plus-two-GEMMs -- beyond the fact that
it is the existing value code with one template argument changed. It cannot be
run under the shim and it cannot be cleanly type-checked either: with the
launch configuration stripped, `__shared__` mapped to a plain local and the
Boys table stubbed, an *unmodified* `md4.kernel.cu` already fails in
`hermite_to_pure<0,B>` and an unmodified `md3.kernel.cu` in
`md3_x_cd_kernel`'s `foreach` lambda, both places where nvcc's overload
resolution differs from g++'s. What that check does establish is a
*difference*: modified and unmodified produce the identical error set, 12 for
md4 and 21 for md3, all of those two pre-existing kinds -- so everything the
derivative work added to those two files type-checks, including the new
explicit instantiations, `compute1`, and `compute_v2`'s `DA`/`DC`.
`gpu/md/{md3,md4}.cc` (the dispatch, plain C++) and
`tests/libintx.gpu.deriv.test.cc` go through `-fsyntax-only` clean.

`gpu/eri`'s *combinatorics* — the eight-fold orbit, the
class-pair batching, the same-class triangle rule and the two index layouts —
were checked by running `eri/format.h` itself on the host (the same shim as
above, extended with `__global__`/`dim3` stubs and a launch-grid loop) against a
synthetic integral carrying exactly the ERI's symmetry group: both formats
reproduce a brute-force reference at four batch bounds, and accumulating instead
of assigning gives the same answer, which is what pins "every element is written
exactly once". That still says nothing about the CUDA half — launch
configuration, occupancy, or whether the kernel runs at all.
