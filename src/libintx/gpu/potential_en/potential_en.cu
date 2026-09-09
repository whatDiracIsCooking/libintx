// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/potential_en/potential_en.h"
#include "libintx/gpu/onebody/kernel.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/gpu/boys.h"

#include "libintx/ao/md/r1.h"
#include "libintx/array.h"
#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/orbital.h"
#include "libintx/utility.h"

#include <cmath>
#include <functional>

// The electron-nuclear attraction matrix on the device.
//
// The substantial one of the three one-electron operators. Overlap and kinetic
// are products of one-dimensional E coefficients; the 1/r operator is not, so
// this kernel carries the two pieces the Coulomb path carries -- the Boys
// function and the Hermite R tensor -- and a per-molecule parameter set that
// lives on the device across compute calls.
//
// Per primitive pair, following libintx::md::nuclear (src/libintx/ao/md/md2.cc):
//
//   1. p = a + b, P = center_of_charge(a, r_a, b, r_b)   [the skeleton's]
//   2. for each nucleus C:  PC = P - R_C,
//        s[m] = F_m(p*|PC|^2) * (-2p)^m,  m = 0..A+B
//        R[t] += -Z_C * r1(PC, s)[t]
//   3. V_ab = (2*pi/p) * sum_{t <= a+b} R[t] * E^{a b}_t
//
// Neither of the two hard pieces is reimplemented here. `gpu::boys()` is the
// tree's one device Chebyshev table -- an order A+B <= 2*LMAX evaluation is
// already inside the table it was sized for, and standing up a second one would
// be a pure duplicate upload. `libintx::md::r1::visit` is LIBINTX_GPU_ENABLED
// and is literally what the device Coulomb kernels call
// (src/libintx/gpu/md/md.kernel.h), so bra/ket conventions cannot drift between
// the two- and four-centre paths.
//
// Everything around that -- the per-pair block, the primitive loop, E in shared
// memory, the cartesian-to-pure pass and the output layout -- is the shared
// skeleton in gpu/onebody/kernel.h, the same as overlap.

namespace libintx::gpu::md::onebody {

  namespace {

    namespace cart = cartesian;
    namespace r1 = libintx::md::r1;

    // Same shape as gpu/overlap/overlap.cu's copy: the Cartesian orbital list
    // as device-resident data, indexed by cart::index(L) + i. Only single
    // shells are indexed, so LMAX spans it.
    __device__
    constexpr auto orbitals = hermite::orbitals2<LMAX>;

    /// Threads per block for the (A|B) kernel.
    ///
    /// The same shape as the overlap kernel's -- one block per shell pair, a
    /// whole number of warps, capped at 128 -- and for the same two hard
    /// reasons: `E2::init` parallelises its recursion over `t = 0..A+B` and so
    /// needs `A+B+1` threads, and the final contraction wants one thread per
    /// Cartesian component pair, of which there are `ncart(A)*ncart(B)`.
    ///
    /// The one difference is what fills the block in between: here it is the
    /// loop over nuclei (see `Nuclear::operator()`), which has O(10-100)
    /// iterations regardless of `(A|B)`, so even (0|0) has real work for 32
    /// threads.
    template<int A, int B>
    constexpr int block_size() {
      constexpr int warp = 32;
      constexpr int n = warp*((ncart(A)*ncart(B) + warp - 1)/warp);
      static_assert(A + B + 1 <= warp);
      return (n > 128 ? 128 : n);
    }

    /// The electron-nuclear potential operator body: one primitive pair's
    /// contribution to the Cartesian accumulator `U`.
    ///
    /// Parallelisation. Three axes are available -- shell pairs, the Hermite
    /// index and the nuclei -- and the issue's recommended starting point is a
    /// block per shell pair with the nuclei looped serially. That is what this
    /// is, with one deviation: the nuclei are spread across the block rather
    /// than run on thread 0, because the alternative leaves 31 of 32 threads
    /// (127 of 128 at (3|3)) idle through the part of the kernel that dominates
    /// it -- `r1::visit<A+B>` per nucleus, O(10-100) times per primitive pair,
    /// against a contraction that is one pass over `ncart(A)*ncart(B)`.
    ///
    /// The R accumulator is therefore a cross-thread reduction, done with
    /// shared-memory `atomicAdd` over `nherm2(A+B)` slots (84 at (3|3), 672
    /// bytes). Summation order is not reproducible run to run, which is
    /// immaterial at the 1e-10 the tests hold this to, and it is not a
    /// correctness question: every term is added exactly once. Reverting to the
    /// strictly serial form is the two-line change of dropping the stride on
    /// the `ic` loop and the atomic.
    ///
    /// Nothing here has been benchmarked -- see the PR -- so read this as the
    /// starting point the issue asked for, not as a measured optimum. If pair
    /// batches turn out to be small, the nuclei axis is the only one wide
    /// enough to fill a device and the right shape is probably a block spanning
    /// nuclei *and* pairs.
    template<int A, int B>
    struct Nuclear {

      static constexpr int L = A + B;
      static constexpr int NP = nherm2(L);

      /// Device pointer to the point charges from the engine's `set()`.
      const NuclearCenter *centers;
      int ncenters;
      /// The tree's one device Chebyshev table, by value -- the same way
      /// md3/md4 hand `gpu::boys()` to their kernels. The copy is a pointer to
      /// a table the `Boys` singleton owns, not a second table.
      Boys boys;

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A,B,0> &E,
        double *U,
        const Block &block) const
      {

        // R[t] = sum_C -Z_C * R^C_t, the whole nuclei sum for this primitive
        // pair. Reused across the skeleton's primitive loop; `compute2` syncs
        // after each call, so the zero-fill below cannot race the previous
        // iteration's readers.
        __shared__ double R[NP];
        fill(NP, R, 0.0, block);
        block.sync();

        const double p = pair.a + pair.b;

        for (int ic = block.thread_rank(); ic < ncenters; ic += Block::size()) {
          const double Z = centers[ic].Z;
          // A zero charge contributes nothing; the host skips it explicitly
          // (md2.cc) and so does this, which also keeps a Z = 0 padding entry
          // from paying for a Boys evaluation.
          if (!Z) continue;
          auto PC = pair.P - centers[ic].r;
          double s[L+1] = {};
          // T = 0 (a nucleus sitting exactly on the centre of charge) is the
          // interpolated branch of the Chebyshev table, not the asymptotic
          // one -- finite, and the classic way this integral goes wrong.
          boys.template compute<L>(p*norm(PC), 0, s);
          double f = 1;
          for (int m = 0; m <= L; ++m) {
            s[m] *= f;
            f *= -2*p;
          }
          r1::visit<L>(
            [&](auto &&r) {
              atomicAdd(&R[r.index], -Z*r.value);
            },
            PC, s
          );
        }
        block.sync();

        const double C = pair.C*(2*math::pi/p);

        constexpr int N = ncart(A)*ncart(B);
        for (int i = block.thread_rank(); i < N; i += Block::size()) {
          int ia = i%ncart(A);
          int ib = i/ncart(A);
          auto a = orbitals[cart::index(A)+ia];
          auto b = orbitals[cart::index(B)+ib];
          // The Hermite expansion of the product a*b terminates at a+b in each
          // axis. Unlike overlap and kinetic this reads E over the full
          // nherm2(A+B) index rather than only degree 0 -- which is why the
          // E2 the skeleton builds is DB = 0 but the whole `t` triangle of it
          // is live here.
          const double *Ex = &E.value(a[0], b[0], 0, 0);
          const double *Ey = &E.value(a[1], b[1], 0, 1);
          const double *Ez = &E.value(a[2], b[2], 0, 2);
          double u = 0;
          for (int iz = 0; iz <= a[2]+b[2]; ++iz) {
            for (int iy = 0; iy <= a[1]+b[1]; ++iy) {
              double Eyz = Ey[iy]*Ez[iz];
              for (int ix = 0; ix <= a[0]+b[0]; ++ix) {
                auto t = Orbital{{(uint8_t)ix,(uint8_t)iy,(uint8_t)iz}};
                u += R[hermite::index2(t)]*Ex[ix]*Eyz;
              }
            }
          }
          U[uindex<A,B>(ia,ib)] += C*u;
        }

      }

    };

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void potential_en_kernel(
      const GaussianPairs basis,
      const Nuclear<A,B> op,
      double *V,
      size_t ldV)
    {
      // DB = 0: the nuclear contraction differentiates neither shell, so E is
      // wanted at (A,B) -- but over every Hermite degree t <= A+B, which is the
      // whole triangle E2 stores anyway.
      compute2<Block,A,B,0,true>(basis, op, V, ldV);
    }

    template<int A, int B>
    void launch(
      const GaussianPairs &ab,
      const NuclearCenter *centers,
      int ncenters,
      double *V,
      size_t ldV,
      gpuStream_t stream)
    {
      using Block = thread_block< block_size<A,B>() >;
      dim3 grid = { (unsigned int)ab.N };
      Nuclear<A,B> op = { centers, ncenters, gpu::boys() };
      potential_en_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, op, V, ldV);
    }

  }

  void potential_en(
    const GaussianPairs &ab,
    const NuclearCenter *centers,
    int ncenters,
    double *V,
    size_t ldV,
    gpuStream_t stream)
  {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    // The host engine writes the pure layout unconditionally, so a Cartesian
    // batch has no agreed answer to be checked against.
    libintx_assert(ab.first.pure && ab.second.pure);
    // compute(Nuclear, ...) before any set(). The host asserts the same thing
    // (`assert(!Cs.empty())`, md2.cc); saying so beats reading an empty
    // device::vector and returning zeros.
    libintx_assert(centers && ncenters > 0);

    using Kernel = std::function<void(
      const GaussianPairs&, const NuclearCenter*, int, double*, size_t, gpuStream_t
    )>;

    // (LMAX+1)^2 = 16 kernels at LMAX=3. This is the operator most likely to
    // need the per-pair OBJECT-library split gpu/md/CMakeLists.txt uses for
    // md3/md4 -- it instantiates the Boys evaluation and the r1 recursion per
    // (A,B) -- but one translation unit is where to start; split it here if
    // nvcc compile time gets unreasonable.
    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, centers, ncenters, V, ldV, stream);

  }

}
