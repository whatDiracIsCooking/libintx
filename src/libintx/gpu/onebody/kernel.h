// -*-c++-*-
#ifndef LIBINTX_GPU_ONEBODY_KERNEL_H
#define LIBINTX_GPU_ONEBODY_KERNEL_H

#include "libintx/gpu/onebody/basis.h"
#include "libintx/gpu/md/e2.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/pure.transform.h"
#include "libintx/array.h"
#include "libintx/math.h"
#include "libintx/shell.h"

#include <cmath>

// The block-level skeleton the three one-electron operator kernels share.
//
// This header is device code -- include it from a .cu only.

namespace libintx::gpu::md::onebody {

  /// One primitive pair, as an operator body wants it.
  struct PrimitivePair {
    /// bra and ket exponents.
    double a, b;
    /// `C_a*C_b*exp(-a*b/(a+b)*|r_a-r_b|^2)`, the contraction coefficients
    /// times the pair pre-factor.
    double C;
    /// bra and ket centres, and `r_a - r_b`.
    array<double,3> ra, rb, AB;
    /// Centre of charge, `(a*r_a + b*r_b)/(a+b)`.
    array<double,3> P;
  };

  /// Threads per block for a one-electron `(A|B)` kernel.
  ///
  /// Two hard constraints and one preference:
  ///
  ///  - `E2::init` parallelises the Hermite recursion over `t = 0..A+B+DB`,
  ///    one thread per `t`, so a block narrower than `A+B+DB+1` silently drops
  ///    coefficients. `compute2` static_asserts it; a warp covers every
  ///    `A+B+DB` this tree can be configured for.
  ///  - the accumulation wants one thread per Cartesian component pair, of
  ///    which there are `ncart(A)*ncart(B)` -- 100 at (3|3).
  ///  - a whole number of warps, capped at 128. Past that a block is mostly
  ///    threads idling in `E2`'s syncs, and `E2` is the serial part.
  ///
  /// The one-block-per-shell-pair shape this implies is what `compute2` is
  /// written against. The alternative -- pairs along threadIdx.x, components
  /// along threadIdx.y, the way md4's `md_v0_kernel_base` bins them -- should
  /// win for small `(A|B)` with `K = 1`, where a whole block per pair has
  /// almost nothing to do. That is a measurement nobody has made yet; make it
  /// before rewriting this.
  ///
  /// @tparam DB the operator's extra ket degree, as passed to `compute2`.
  template<int A, int B, int DB>
  constexpr int block_size() {
    constexpr int warp = 32;
    constexpr int n = warp*((ncart(A)*ncart(B) + warp - 1)/warp);
    static_assert(A + B + DB + 1 <= warp);
    return (n > 128 ? 128 : n);
  }

  /// Cartesian accumulator layout: `U[ia + ib*ncart(A)]`, column-major in the
  /// bra index so the pure transform below and the engine's output layout
  /// agree without a shuffle.
  template<int A, int B>
  LIBINTX_GPU_DEVICE
  constexpr int uindex(int ia, int ib) { return ia + ib*ncart(A); }

  /// Block-level driver for a 2-centre operator.
  ///
  /// One thread block per shell pair, `blockIdx.x` the pair index. For each of
  /// the batch's `K = K_a*K_b` primitive pairs this builds E in shared memory
  /// and hands `(pair, E, U, block)` to `op`, which accumulates its operator's
  /// contribution into the shared Cartesian buffer `U`. Once the primitive
  /// loop is done `U` is transformed to solid harmonics (when the shells are
  /// pure) and written to `V` in the host engine's layout,
  ///
  ///     V[ij + (na + nb*npure(A))*ldV]
  ///
  /// so `test::check2` and any caller are drop-in against the host engine.
  ///
  /// Thread-block contract, inherited from `E2::init` and asserted here:
  /// `Block::size() >= A + B + DB + 1`.
  ///
  /// @tparam DB extra ket degree the operator needs from E -- 0 for overlap
  ///         and the nuclear potential, 2 for kinetic.
  /// @tparam Pure whether the two shells are solid-harmonic. The host engine
  ///         only supports the pure case; the Cartesian branch is here so the
  ///         layout stays `nbf`-consistent if that ever changes.
  template<typename Block, int A, int B, int DB, bool Pure, typename Op>
  __device__
  void compute2(const GaussianPairs &basis, Op &&op, double *V, size_t ldV) {

    static_assert(Block::size() >= (A+B+DB+1));

    constexpr int NA = (Pure ? npure(A) : ncart(A));
    constexpr int NB = (Pure ? npure(B) : ncart(B));

    auto block = Block();

    // Gaussian carries default member initializers, so a bare
    // `__shared__ Gaussian2` has a non-empty constructor and nvcc rejects it.
    // Same union dodge gpu/md/basis.cu uses.
    __shared__
    union shmem {
      Gaussian2 ab;
      __device__ shmem() {}
    } shmem;
    auto &ab = shmem.ab;

    __shared__ array<double,3> AB;
    __shared__ double U[ncart(A)*ncart(B)];
    __shared__ E2<A,B,DB> E;

    memcpy1(&basis.data[blockIdx.x], &ab, block);
    fill(ncart(A)*ncart(B), U, 0.0, block);
    block.sync();

    if (block.thread_rank() == 0) {
      AB = ab.r.first - ab.r.second;
    }
    block.sync();

    for (int ki = 0; ki < ab.first.K; ++ki) {
      for (int kj = 0; kj < ab.second.K; ++kj) {

        __shared__ PrimitivePair pair;

        if (block.thread_rank() == 0) {
          auto& [ai,Ci] = ab.first.prims[ki];
          auto& [aj,Cj] = ab.second.prims[kj];
          pair.a = ai;
          pair.b = aj;
          pair.C = Ci*Cj*std::exp(-(ai*aj)/(ai+aj)*norm(AB));
          pair.ra = ab.r.first;
          pair.rb = ab.r.second;
          pair.AB = AB;
          pair.P = center_of_charge(ai, ab.r.first, aj, ab.r.second);
        }
        block.sync();

        // Every thread takes part: init syncs internally.
        E.init(pair.a, pair.b, AB, block);
        block.sync();

        op(pair, E, U, block);
        block.sync();

      }
    }

    if constexpr (!Pure) {
      for (int i = block.thread_rank(); i < NA*NB; i += Block::size()) {
        V[blockIdx.x + i*ldV] = U[i];
      }
      return;
    }
    else {
      // [ia,ib] -> [ib,pa] -> [pa,pb]. The intermediate is transposed so the
      // ket slice each cartesian_to_pure<B> reads is contiguous.
      __shared__ double S[ncart(B)*npure(A)];
      for (int ib = block.thread_rank(); ib < ncart(B); ib += Block::size()) {
        pure::cartesian_to_pure<A>(
          [&](auto ia) { return U[uindex<A,B>(index(ia),ib)]; },
          [&](auto ia, auto v) { S[ib + index(ia)*ncart(B)] = v; }
        );
      }
      block.sync();
      for (int pa = block.thread_rank(); pa < npure(A); pa += Block::size()) {
        pure::cartesian_to_pure<B>(
          [&](auto ib) { return S[index(ib) + pa*ncart(B)]; },
          [&](auto ib, auto v) {
            V[blockIdx.x + (pa + index(ib)*NA)*ldV] = v;
          }
        );
      }
    }

  }

}

#endif /* LIBINTX_GPU_ONEBODY_KERNEL_H */
