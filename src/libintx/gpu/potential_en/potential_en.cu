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

    /// `sum_t Ex[t_x]*Ey[t_y]*Ez[t_z] * R[index2(t + shift)]`, the Hermite
    /// contraction the value body and both derivative bodies come down to.
    ///
    /// `m[]` bounds each axis (`t_x <= m[0]`, ...) and `shift[]` is added to
    /// the Hermite index R is looked up at, which is how the Hellmann-Feynman
    /// term reads `R_{t+1_x}` without a second tensor. The `E` pointers are
    /// rows of the shared `E2` -- `&E.value(i,j,0,x)` is contiguous in the
    /// Hermite index -- except where a caller substitutes a local array, which
    /// is how the bra derivative folds `2*alpha*E^{i+1,j} - i_x*E^{i-1,j}` in
    /// on one axis.
    LIBINTX_GPU_DEVICE LIBINTX_ALWAYS_INLINE
    double contract(
      const double* const (&E)[3], const int (&m)[3], const int (&shift)[3],
      const double *R)
    {
      double u = 0;
      for (int iz = 0; iz <= m[2]; ++iz) {
        for (int iy = 0; iy <= m[1]; ++iy) {
          double Eyz = E[1][iy]*E[2][iz];
          for (int ix = 0; ix <= m[0]; ++ix) {
            auto t = Orbital{{
              (uint8_t)(ix+shift[0]), (uint8_t)(iy+shift[1]), (uint8_t)(iz+shift[2])
            }};
            u += R[hermite::index2(t)]*E[0][ix]*Eyz;
          }
        }
      }
      return u;
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
          const double *e[3] = {
            &E.value(a[0], b[0], 0, 0),
            &E.value(a[1], b[1], 0, 1),
            &E.value(a[2], b[2], 0, 2)
          };
          const int m[3] = { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
          constexpr int shift[3] = { 0, 0, 0 };
          U[uindex<A,B>(ia,ib)] += C*contract(e, m, shift, R);
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
      // `gpu/onebody/kernel.h`'s shared block shape, at this operator's
      // DB = 0: one block per shell pair, a whole number of warps capped at
      // 128. What fills the block here that does not fill it for overlap is
      // the loop over nuclei (see `Nuclear::operator()`), which has O(10-100)
      // iterations regardless of `(A|B)` -- so even (0|0) has real work for
      // 32 threads.
      using Block = thread_block< block_size<A,B,0>() >;
      dim3 grid = { (unsigned int)ab.N };
      Nuclear<A,B> op = { centers, ncenters, gpu::boys() };
      potential_en_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, op, V, ldV);
    }

    /// The bra-derivative half of `dV/dX`: `dV/dA_x`, three components.
    ///
    /// Differentiating a Cartesian primitive with respect to its own centre
    /// raises and lowers one Cartesian index,
    ///
    ///     d/dA_x (a|  =  2*alpha*(a+1_x|  -  i_x*(a-1_x|
    ///
    /// and the nuclear attraction integral is linear in the bra primitive, so
    /// the identity passes straight through it: the raised and lowered
    /// integrals are the same p, P, PC and R tensor with different Hermite
    /// expansion coefficients. So
    ///
    ///     dV_ab/dA_x = (2*pi/p) * sum_t [2a*E^{a+1_x,b}_t - a_x*E^{a-1_x,b}_t] * R_t
    ///
    /// with `R_t = sum_C -Z_C * r1(P-R_C, s)[t]` exactly as the value kernel
    /// builds it. Nothing shifts a shell: `E` is `E2<A+1,B,0>`, allocated by
    /// the skeleton's `DA = 1`, so the batch keeps the parent's primitive
    /// coefficients and there is no `gto::normalized` factor to get wrong.
    ///
    /// **This is where the extra Boys order comes from, and the
    /// Hellmann-Feynman term is not the only thing that needs it.** `a+1_x`
    /// reaches Hermite degree `A+B+1`, so `R` is `nherm2(A+B+1)` and the
    /// recursion is `r1::visit<A+B+1>` over Boys values through `m = A+B+1`.
    /// That is `2*LMAX+1` at the top, and `gpu::boys()` already carries it --
    /// the tree's one device Chebyshev table is sized
    /// `max(4*LMAX, 2*LMAX+XMAX)+1` orders for the four-centre Coulomb path,
    /// which is strictly more for every LMAX this tree configures. The
    /// static_assert is what keeps that true if the sizing ever changes; the
    /// table did NOT have to be extended, and extending it would have been a
    /// larger upload for nothing.
    template<int A, int B>
    struct NuclearD1 {

      static constexpr int L = A + B + 1;
      static constexpr int NP = nherm2(L);

      static_assert(
        L < Boys::orders,
        "gpu::boys() does not span the derivative's Boys order"
      );

      const NuclearCenter *centers;
      int ncenters;
      Boys boys;

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A+1,B,0> &E,
        double *U,
        const Block &block) const
      {

        // The whole nuclei sum, one Hermite degree wider than the value
        // kernel's. Same cross-thread atomicAdd reduction, for the same
        // reason -- see Nuclear::operator() above.
        __shared__ double R[NP];
        fill(NP, R, 0.0, block);
        block.sync();

        const double p = pair.a + pair.b;

        for (int ic = block.thread_rank(); ic < ncenters; ic += Block::size()) {
          const double Z = centers[ic].Z;
          if (!Z) continue;
          auto PC = pair.P - centers[ic].r;
          double s[L+1] = {};
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
          // Rows of E at the undifferentiated degree, one per axis; each
          // component replaces exactly one of them.
          const double *e[3] = {
            &E.value(a[0], b[0], 0, 0),
            &E.value(a[1], b[1], 0, 1),
            &E.value(a[2], b[2], 0, 2)
          };
          const int m[3] = { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
          constexpr int shift[3] = { 0, 0, 0 };
          for (int x = 0; x < 3; ++x) {
            // `2*alpha*E^{i+1,j} - i_x*E^{i-1,j}` along the differentiated
            // axis. `E2::init` zero-fills and the recursion only ever writes
            // `k <= i+j`, so reading the lowered row past its own bound gives
            // zeros rather than a neighbour's coefficients -- and the `i_x`
            // factor kills that whole term at `i_x = 0` anyway, which is why
            // the row index is clamped rather than the term branched on.
            double D[A+B+2];
            const double *ep = &E.value(a[x]+1, b[x], 0, x);
            const double *em = &E.value(a[x] ? a[x]-1 : 0, b[x], 0, x);
            const int md = a[x]+1+b[x];
            for (int t = 0; t <= md; ++t) {
              D[t] = 2*pair.a*ep[t] - a[x]*em[t];
            }
            const double *ex[3] = { e[0], e[1], e[2] };
            int mx[3] = { m[0], m[1], m[2] };
            ex[x] = D;
            mx[x] = md;
            U[uindex<A,B>(ia,ib,x)] += C*contract(ex, mx, shift, R);
          }
        }

      }

    };

    /// The Hellmann-Feynman half of `dV/dX`: `dV/dR_C,x` for ONE nucleus,
    /// three components. The nucleus is `blockIdx.y`.
    ///
    /// `1/|r - R_C|` depends on the nuclear position directly, so this term
    /// exists for no other one-electron operator in this tree -- and a
    /// gradient carrying only the bra/ket derivative is wrong by the dominant
    /// part of the force on any charged atom while still being smooth and
    /// entirely plausible.
    ///
    /// `R_t(PC)` is `(d/dPC_x)^{t_x} (d/dPC_y)^{t_y} (d/dPC_z)^{t_z}` of the
    /// Boys kernel, so differentiating it raises the Hermite index by one, and
    /// `PC = P - R_C` supplies the sign:
    ///
    ///     dV_ab/dR_C,x = -(2*pi/p) * sum_t E^{ab}_t * R^C_{t+1_x}
    ///
    /// with `R^C_t = -Z_C * r1(PC, s)[t]`, i.e. the value kernel's per-nucleus
    /// term kept per nucleus rather than summed into one accumulator. `DA = 0`:
    /// `E` is wanted at `(A,B)`, only `R` goes one degree wider.
    ///
    /// Shape. One block per (pair, nucleus), so the per-nucleus output never
    /// has to live in shared memory and the grid is `ncenters` times wider --
    /// which is the axis that fills a device when a batch is small. The price
    /// is that `E` is rebuilt once per nucleus rather than once per pair, and
    /// that `r1::visit` runs on thread 0 while the rest of the block waits
    /// (there is exactly one nucleus here, and the recursion does not divide).
    /// The alternative -- nuclei inside the primitive loop, the way the value
    /// kernel has them -- needs `ncenters*3*ncart(A)*ncart(B)` accumulators,
    /// which is not a shared-memory shape for any real molecule. **Nothing
    /// here has been benchmarked**, on this arrangement or any other.
    template<int A, int B>
    struct NuclearD1C {

      static constexpr int L = A + B + 1;
      static constexpr int NP = nherm2(L);

      static_assert(
        L < Boys::orders,
        "gpu::boys() does not span the derivative's Boys order"
      );

      const NuclearCenter *centers;
      int ncenters;
      Boys boys;

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A,B,0> &E,
        double *U,
        const Block &block) const
      {

        const int ic = blockIdx.y;
        const double Z = centers[ic].Z;
        // Uniform across the block -- every thread reads the same charge -- so
        // this is a block-wide early exit, not divergence. A zero charge feels
        // no force: `U` stays zero and the skeleton writes zeros, which is the
        // answer, not a skipped one.
        if (!Z) return;

        __shared__ double R[NP];
        fill(NP, R, 0.0, block);
        block.sync();

        const double p = pair.a + pair.b;
        auto PC = pair.P - centers[ic].r;

        if (block.thread_rank() == 0) {
          double s[L+1] = {};
          // T = 0 -- a nucleus sitting exactly on the centre of charge -- is
          // the interpolated branch of the Chebyshev table, not the asymptotic
          // one, and it stays finite for the derivative: the Hermite terms
          // that carry an odd power of PC vanish there rather than diverging.
          boys.template compute<L>(p*norm(PC), 0, s);
          double f = 1;
          for (int m = 0; m <= L; ++m) {
            s[m] *= f;
            f *= -2*p;
          }
          r1::visit<L>(
            [&](auto &&r) {
              R[r.index] = -Z*r.value;
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
          const double *e[3] = {
            &E.value(a[0], b[0], 0, 0),
            &E.value(a[1], b[1], 0, 1),
            &E.value(a[2], b[2], 0, 2)
          };
          const int m[3] = { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
          for (int x = 0; x < 3; ++x) {
            int shift[3] = { 0, 0, 0 };
            shift[x] = 1;
            U[uindex<A,B>(ia,ib,x)] -= C*contract(e, m, shift, R);
          }
        }

      }

    };

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void potential_en1_kernel(
      const GaussianPairs basis,
      const NuclearD1<A,B> op,
      double *V,
      size_t ldV)
    {
      // DA = 1: the raising relation reads E at bra degree A+1. NC = 3: one
      // accumulator block per Cartesian component.
      compute2<Block,A,B,0,true,1,3>(basis, op, V, ldV);
    }

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void potential_en1c_kernel(
      const GaussianPairs basis,
      const NuclearD1C<A,B> op,
      double *V,
      size_t ldV,
      int nab)
    {
      // blockIdx.y is the nucleus. Offsetting V here rather than inside the
      // operator leaves the skeleton's output layout untouched: each nucleus
      // writes a complete `(ij, na, nb, x)` block, and the nucleus becomes the
      // slowest index of the buffer by construction.
      compute2<Block,A,B,0,true,0,3>(
        basis, op, V + (size_t)blockIdx.y*3*nab*ldV, ldV
      );
    }

    template<int A, int B>
    void launch1(
      const GaussianPairs &ab,
      const NuclearCenter *centers,
      int ncenters,
      double *dV,
      double *dVC,
      size_t ldV,
      gpuStream_t stream)
    {
      {
        using Block = thread_block< block_size<A,B,0,1>() >;
        dim3 grid = { (unsigned int)ab.N };
        NuclearD1<A,B> op = { centers, ncenters, gpu::boys() };
        potential_en1_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, op, dV, ldV);
      }
      {
        using Block = thread_block< block_size<A,B,0>() >;
        dim3 grid = { (unsigned int)ab.N, (unsigned int)ncenters };
        NuclearD1C<A,B> op = { centers, ncenters, gpu::boys() };
        potential_en1c_kernel<Block,A,B><<<grid,Block(),0,stream>>>(
          ab, op, dVC, ldV, npure(A)*npure(B)
        );
      }
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

  void potential_en1(
    const GaussianPairs &ab,
    const NuclearCenter *centers,
    int ncenters,
    double *dV,
    double *dVC,
    size_t ldV,
    gpuStream_t stream)
  {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    libintx_assert(ab.first.pure && ab.second.pure);
    libintx_assert(centers && ncenters > 0);
    // Both halves are required and there is no "bra derivative only" mode, on
    // purpose: a caller that dropped the Hellmann-Feynman term would get a
    // smooth, plausible, wrong force rather than an error.
    libintx_assert(dV && dVC);

    using Kernel = std::function<void(
      const GaussianPairs&, const NuclearCenter*, int, double*, double*,
      size_t, gpuStream_t
    )>;

    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch1<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, centers, ncenters, dV, dVC, ldV, stream);

  }

}
