// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/kinetic/kinetic.h"
#include "libintx/gpu/onebody/kernel.h"
#include "libintx/gpu/api/thread_group.h"

#include "libintx/array.h"
#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/orbital.h"
#include "libintx/utility.h"

#include <cmath>
#include <functional>

// The AO kinetic-energy matrix on the device.
//
// Differentiating the ket Gaussian twice turns
//
//     T_ab = -1/2 <a| nabla^2 |b>
//
// into a fixed combination of OVERLAP integrals with the ket angular momentum
// shifted by +2, 0 and -2 along each axis:
//
//     T_ab = (pi/p)^(3/2) * [ b*(2*B+3)*t0 - 2*b^2*t1 - (1/2)*t2 ]
//
//     t0 = E^{l1 l2}_0 E^{m1 m2}_0 E^{n1 n2}_0
//     t1 = sum over the three axes of t0 with that axis' ket degree +2
//     t2 = sum over the three axes of j*(j-1) times t0 with that axis' -2
//
// which is `libintx::md::kinetic` (src/libintx/ao/md/md2.cc:94-121) with the
// loop over the Cartesian components spread across a thread block. Everything
// around it -- the per-pair block, the primitive loop, E in shared memory, the
// cartesian-to-pure pass and the output layout -- is the shared skeleton in
// gpu/onebody/kernel.h, the same one gpu/overlap/overlap.cu uses. The only
// structural difference from overlap is E2's extra ket degree (`DB = 2`) and
// the three-term contraction.
//
// Two things carried over from the host exactly:
//
//  - `E2<A,B,2>`. The `+2` is on the KET index -- what the host spells
//    `libintx::md::E2<T,A,B+2,0>`. gpu/md/e2.h already carries it as its `DB`
//    parameter, so nothing here hardcodes an enlarged `B`.
//  - `2*B+3` is the KET shell's angular momentum. Every Cartesian orbital of
//    that shell has `l2+m2+n2 == B`, so spelling it per-orbital is the same
//    number -- but reaching for the BRA's `A` instead is a real mistake, and
//    one that is invisible on every diagonal block `(L|L)`. Only the
//    off-diagonal (A|B) of the reference sweep catches it.
//
// The host's transpose branch is NOT replicated -- see `Kinetic` below.

namespace libintx::gpu::md::onebody {

  namespace {

    namespace cart = cartesian;

    // Same shape as gpu/overlap/overlap.cu's copy: the Cartesian orbital list,
    // as device-resident data indexed by cart::index(L) + i.
    __device__
    constexpr auto orbitals = hermite::orbitals2<LMAX>;

    /// The kinetic operator body: one primitive pair's contribution to the
    /// Cartesian accumulator `U`.
    ///
    /// One thread per Cartesian component pair `(ia,ib)`, each thread owning
    /// its own element of `U`, so the accumulation needs no atomics and no
    /// reduction. `pair.C` already carries `C_a*C_b*Kab` from the skeleton;
    /// what is left is the `(pi/p)^(3/2)` pre-factor and the three E terms.
    ///
    /// On the host's transpose branch. `libintx::md::compute2`
    /// (src/libintx/ao/md/md2.cc:246-257) evaluates `(A|B)` as `kinetic<B,A>`
    /// on the swapped, sign-flipped pair whenever `B > A`, writing through a
    /// transposing accessor, on the grounds that the `+2` is cheaper on the
    /// smaller index. On the device that trade goes the other way and the
    /// swap is a pessimisation, so it is deliberately not replicated:
    ///
    ///   - E2<A,B,2> is `(A+1)*(B+3)*(A+B+3)*3` doubles and the swap makes it
    ///     `(B+1)*(A+3)*(...)`. The difference is `2*(A-B)` per `(A+B+3)*3`,
    ///     i.e. the swapped table is strictly LARGER exactly when `B > A` --
    ///     which is precisely when the host swaps. At `(s|f)`: 6 doubles per
    ///     axis unswapped against 12 swapped.
    ///   - `E2::init` syncs `A + (B+2)*(A+1)` times, against
    ///     `B + (A+2)*(B+1)` swapped -- 5 against 11 at `(s|f)`.
    ///   - `compute2` fixes the accumulator as `U[ia + ib*ncart(A)]` and the
    ///     pure transform and output stride with it, so a swap would have to
    ///     be undone in every write of this functor: the one line the issue
    ///     calls the most bug-prone in the host path, bought for nothing.
    ///
    /// (Swapping the OTHER way -- when `A > B` -- would be a real saving by
    /// both counts above, but it needs the skeleton to build E on the swapped
    /// pair, not just this functor to transpose its writes. Nobody has
    /// measured whether it is worth that; the reference sweep covers both
    /// index orders either way.)
    template<int A, int B>
    struct Kinetic {

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A,B,2> &E,
        double *U,
        const Block &block) const
      {
        // The ket exponent: the differentiation is of |b>, so every term
        // below carries a power of it and none carries pair.a.
        const double b = pair.b;
        const double C = pair.C*std::sqrt(math::pow<3>(math::pi/(pair.a + b)));
        // E^{ij}_0 along one axis. The Hermite index is always 0 -- kinetic,
        // like overlap, is a pure product of one-dimensional overlaps.
        auto Ex = [&](int i, int j) { return E.value(i,j,0,0); };
        auto Ey = [&](int i, int j) { return E.value(i,j,0,1); };
        auto Ez = [&](int i, int j) { return E.value(i,j,0,2); };
        constexpr int N = ncart(A)*ncart(B);
        for (int i = block.thread_rank(); i < N; i += Block::size()) {
          int ia = i%ncart(A);
          int ib = i/ncart(A);
          auto a1 = orbitals[cart::index(A)+ia];
          auto a2 = orbitals[cart::index(B)+ib];
          int l1 = a1[0], m1 = a1[1], n1 = a1[2];
          int l2 = a2[0], m2 = a2[1], n2 = a2[2];
          double t0 = Ex(l1,l2)*Ey(m1,m2)*Ez(n1,n2);
          double t1 = (
            Ex(l1,l2+2)*Ey(m1,m2)*Ez(n1,n2) +
            Ex(l1,l2)*Ey(m1,m2+2)*Ez(n1,n2) +
            Ex(l1,l2)*Ey(m1,m2)*Ez(n1,n2+2)
          );
          double t2 = 0;
          // j*(j-1) is 0 for j < 2, which is also where the `j-2` index below
          // would be out of range -- so the guard is not an optimisation.
          int l = l2*(l2-1);
          int m = m2*(m2-1);
          int n = n2*(n2-1);
          if (l) t2 += l*Ex(l1,l2-2)*Ey(m1,m2)*Ez(n1,n2);
          if (m) t2 += m*Ex(l1,l2)*Ey(m1,m2-2)*Ez(n1,n2);
          if (n) t2 += n*Ex(l1,l2)*Ey(m1,m2)*Ez(n1,n2-2);
          U[uindex<A,B>(ia,ib)] += C*(b*(2*B+3)*t0 - 2*b*b*t1 - 0.5*t2);
        }
      }

    };

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void kinetic_kernel(const GaussianPairs basis, double *V, size_t ldV) {
      // DB = 2: the ket degree runs to B+2, so E is 648 doubles (5.2 KB) at
      // (3|3) against overlap's 336 -- still far inside LIBINTX_GPU_MAX_SHMEM.
      compute2<Block,A,B,2,true>(basis, Kinetic<A,B>{}, V, ldV);
    }

    template<int A, int B>
    void launch(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {
      using Block = thread_block< block_size<A,B,2>() >;
      dim3 grid = { (unsigned int)ab.N };
      kinetic_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, V, ldV);
    }

  }

  void kinetic(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    // The host engine writes the pure layout unconditionally, so a Cartesian
    // batch has no agreed answer to be checked against.
    libintx_assert(ab.first.pure && ab.second.pure);

    using Kernel = std::function<void(
      const GaussianPairs&, double*, size_t, gpuStream_t
    )>;

    // (LMAX+1)^2 = 16 kernels at LMAX=3, the whole two-centre table in one
    // translation unit.
    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, V, ldV, stream);

  }

}
