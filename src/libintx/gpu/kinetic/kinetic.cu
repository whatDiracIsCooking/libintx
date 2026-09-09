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

    /// The kinetic gradient body: one primitive pair's contribution to
    /// `dT/dA_x`, three components into the same Cartesian accumulator.
    ///
    /// Differentiating with respect to the BRA centre does not change the
    /// three-term structure above. `T_ab` is a fixed linear combination of
    /// one-dimensional overlap products, and
    ///
    ///     d/dA_x (a|  =  2*alpha*(a+1_x|  -  i_x*(a-1_x|
    ///
    /// acts on the bra Cartesian index alone, axis by axis. So `dT/dA_x` is
    /// the same `b*(2*B+3)*t0 - 2*b^2*t1 - (1/2)*t2` with every factor on axis
    /// `x` replaced by its derivative,
    ///
    ///     D^{i j}_x = 2*alpha*E^{i+1,j}_0 - i*E^{i-1,j}_0
    ///
    /// and the other two axes left alone. Three things follow, all of them
    /// easy to get wrong:
    ///
    ///  - **`2*B+3` is untouched.** It is the KET shell's angular momentum,
    ///    which raising the bra index does not move. (It is also the mutation
    ///    that is invisible on every diagonal `(L|L)` -- see the file header.)
    ///  - **Every exponent in the body is still the ket's.** `pair.a` appears
    ///    exactly once, in the raising relation; `b`, `2*b^2` and the `j(j-1)`
    ///    lowering coefficients are the ket's as before.
    ///  - **Nothing shifts a shell.** The raised coefficient is read out of
    ///    `E2<A+1,B,2>`, which the skeleton's `DA = 1` allocates, so the
    ///    batch's primitive coefficients stay the parent's and there is no
    ///    `gto::normalized` factor to get wrong. And the ket derivative is not
    ///    computed: `dT/dB = -dT/dA` exactly, as `compute1`'s contract says.
    ///
    /// As in `gpu/overlap/overlap.cu`, `pair.C` carrying `K_ab` -- which
    /// depends on the bra centre -- is not a missing term: the raising
    /// relation is an identity on the *primitive function*, so `C*E(a+1_x,..)`
    /// already carries the whole `A`-dependence through `E`'s own recursion.
    ///
    /// The host's transpose branch is not replicated here either, and a
    /// derivative makes the case against it worse: the raised index is no
    /// longer symmetric between bra and ket, so a transposing accessor would
    /// have to know which centre was differentiated.
    template<int A, int B>
    struct KineticD1 {

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A+1,B,2> &E,
        double *U,
        const Block &block) const
      {
        // The ket exponent, exactly as in the value kernel above.
        const double b = pair.b;
        const double C = pair.C*std::sqrt(math::pow<3>(math::pi/(pair.a + b)));
        constexpr int N = ncart(A)*ncart(B);
        for (int i = block.thread_rank(); i < N; i += Block::size()) {
          int ia = i%ncart(A);
          int ib = i/ncart(A);
          auto a1 = orbitals[cart::index(A)+ia];
          auto a2 = orbitals[cart::index(B)+ib];
          // Per axis, the three ket degrees the three-term body reads --
          // `j`, `j+2` and `j-2` -- as the plain coefficient `e` and as its
          // bra derivative `d`. Eighteen lookups, shared by all three output
          // components; every thread owns its own `(ia,ib)`, so as in the
          // value kernel there are no atomics and no reduction.
          double e[3][3] = {};
          double d[3][3] = {};
#pragma unroll
          for (int y = 0; y < 3; ++y) {
            const int i1 = a1[y];
            const int j2 = a2[y];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
              // k = 0,1,2 -> ket degree j, j+2, j-2.
              const int j = (k == 0 ? j2 : (k == 1 ? j2+2 : j2-2));
              // j < 0 only where `j*(j-1)` below is 0, so leaving the slot at
              // zero is not an approximation -- it is the same guard the value
              // kernel spells as `if (l)`.
              if (j < 0) continue;
              e[y][k] = E.value(i1, j, 0, y);
              double v = 2*pair.a*E.value(i1+1, j, 0, y);
              if (i1) v -= i1*E.value(i1-1, j, 0, y);
              d[y][k] = v;
            }
          }
          const int c2[3] = {
            a2[0]*(a2[0]-1), a2[1]*(a2[1]-1), a2[2]*(a2[2]-1)
          };
#pragma unroll
          for (int x = 0; x < 3; ++x) {
            // The differentiated axis takes `d`, the other two take `e`.
            auto g = [&](int y, int k) { return (y == x ? d[y][k] : e[y][k]); };
            double t0 = g(0,0)*g(1,0)*g(2,0);
            double t1 = (
              g(0,1)*g(1,0)*g(2,0) +
              g(0,0)*g(1,1)*g(2,0) +
              g(0,0)*g(1,0)*g(2,1)
            );
            double t2 = (
              c2[0]*g(0,2)*g(1,0)*g(2,0) +
              c2[1]*g(0,0)*g(1,2)*g(2,0) +
              c2[2]*g(0,0)*g(1,0)*g(2,2)
            );
            U[uindex<A,B>(ia,ib,x)] += C*(b*(2*B+3)*t0 - 2*b*b*t1 - 0.5*t2);
          }
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

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void kinetic1_kernel(const GaussianPairs basis, double *V, size_t ldV) {
      // DA = 1 on top of the value kernel's DB = 2: the ket degree still runs
      // to B+2 and the raising relation adds one unit of BRA degree, so E is
      // E2<A+1,B,2> -- 900 doubles (7.0 KiB) at (3|3) against the value
      // kernel's 648 (5.2 KiB). With the NC = 3 accumulator (300 doubles, 2.3
      // KiB) and the pure-transform scratch, one block's shared memory at
      // (3|3) is ~10 KiB, comfortably inside LIBINTX_GPU_MAX_SHMEM (49152).
      // The block-size contract moves with it -- block_size<A,B,2,1>()
      // static_asserts A+1+B+2+1 <= 32, which is 10 at (3|3) -- so the
      // existing 128-thread shape carries over unchanged.
      compute2<Block,A,B,2,true,1,3>(basis, KineticD1<A,B>{}, V, ldV);
    }

    template<int A, int B>
    void launch1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {
      using Block = thread_block< block_size<A,B,2,1>() >;
      dim3 grid = { (unsigned int)ab.N };
      kinetic1_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, V, ldV);
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

  void kinetic1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    // Same guard as the value kernel, and for the same reason -- the extra
    // unit of bra angular momentum the derivative needs is spent inside E,
    // not on the shell, so this table is (LMAX+1)^2 like the other one.
    libintx_assert(ab.first.pure && ab.second.pure);

    using Kernel = std::function<void(
      const GaussianPairs&, double*, size_t, gpuStream_t
    )>;

    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch1<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, V, ldV, stream);

  }

}
