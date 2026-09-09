// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/overlap/overlap.h"
#include "libintx/gpu/onebody/kernel.h"
#include "libintx/gpu/api/thread_group.h"

#include "libintx/array.h"
#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/orbital.h"
#include "libintx/utility.h"

#include <cmath>
#include <functional>

// The AO overlap matrix on the device.
//
// The cheapest of the three one-electron operators: no Boys function, no
// Hermite R tensor, only the t=0 expansion coefficient. Over one primitive
// pair the McMurchie-Davidson overlap factorises completely,
//
//     S_ab = (pi/p)^(3/2) * prod_{x in {x,y,z}} E^{i_x j_x}_0,   p = a + b
//
// which is `libintx::md::overlap` (src/libintx/ao/md/md2.cc) with the loop over
// the Cartesian components spread across a thread block. Everything around that
// -- the per-pair block, the primitive loop, E in shared memory, the
// cartesian-to-pure pass and the output layout -- is the shared skeleton in
// gpu/onebody/kernel.h.

namespace libintx::gpu::md::onebody {

  namespace {

    namespace cart = cartesian;

    // Same shape as gpu/md/basis.cu's copy: the Cartesian orbital list, as
    // device-resident data indexed by cart::index(L) + i. Only single shells
    // are indexed here, so LMAX spans it -- the Coulomb path needs 2*LMAX
    // because it walks Hermite indices of a pair.
    __device__
    constexpr auto orbitals = hermite::orbitals2<LMAX>;

    /// The overlap operator body: one primitive pair's contribution to the
    /// Cartesian accumulator `U`.
    ///
    /// `pair.C` already carries `C_a*C_b*Kab` from the skeleton, so what is
    /// left is the `(pi/p)^(3/2)` pre-factor and `E(a,b,0)` -- one thread per
    /// `(a,b)`, each thread owning its own element of `U`, so the accumulation
    /// needs no atomics and no reduction.
    template<int A, int B>
    struct Overlap {

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A,B,0> &E,
        double *U,
        const Block &block) const
      {
        double C = pair.C*std::sqrt(math::pow<3>(math::pi/(pair.a + pair.b)));
        // NB: Orbital's default member initializer is {0xFF,0xFF,0xFF}, not
        // zero -- the Hermite index has to be spelled out.
        Orbital p = Orbital{{0,0,0}};
        constexpr int N = ncart(A)*ncart(B);
        for (int i = block.thread_rank(); i < N; i += Block::size()) {
          int ia = i%ncart(A);
          int ib = i/ncart(A);
          auto a = orbitals[cart::index(A)+ia];
          auto b = orbitals[cart::index(B)+ib];
          U[uindex<A,B>(ia,ib)] += C*E(a,b,p);
        }
      }

    };

    /// The overlap gradient body: one primitive pair's contribution to
    /// `dS/dA_x`, three components into the same Cartesian accumulator.
    ///
    /// Differentiating a Cartesian primitive with respect to its own centre
    /// raises and lowers one Cartesian index,
    ///
    ///     d/dA_x (a|  =  2*alpha*(a+1_x|  -  i_x*(a-1_x|
    ///
    /// and because the overlap factorises axis by axis -- `S_ab` is
    /// `(pi/p)^{3/2}` times `prod_x E^{i_x j_x}_0` -- the derivative is the
    /// same product with one axis replaced:
    ///
    ///     dS/dA_x = (pi/p)^{3/2} * [ 2a*E^{i_x+1,j_x}_0 - i_x*E^{i_x-1,j_x}_0 ]
    ///                            * prod_{y != x} E^{i_y j_y}_0
    ///
    /// Two things this does *not* do, both of them deliberate. It does not
    /// build a shifted `L+1` shell: the raised coefficient is read straight
    /// out of `E2<A+1,B,0>`, which the skeleton's `DA = 1` allocates, so the
    /// batch's primitive coefficients are the parent's and there is no
    /// `gto::normalized` factor to get wrong. And it does not compute the ket
    /// derivative: `dS/dB = -dS/dA` exactly, and the engine's contract says so.
    ///
    /// Note `pair.C` carries `K_ab = exp(-a*b/p*|AB|^2)`, which depends on the
    /// bra centre too. That is not a missing term: the raising relation above
    /// is an identity on the *primitive function*, so `S_{a+1_x,b}` computed
    /// with the same `K_ab` -- which is what `C*E(a+1_x,b,0)` is -- already
    /// carries the whole `A`-dependence, `K_ab`'s included, through `E`'s own
    /// recursion.
    template<int A, int B>
    struct OverlapD1 {

      template<typename Block>
      __device__
      void operator()(
        const PrimitivePair &pair,
        E2<A+1,B,0> &E,
        double *U,
        const Block &block) const
      {
        double C = pair.C*std::sqrt(math::pow<3>(math::pi/(pair.a + pair.b)));
        constexpr int N = ncart(A)*ncart(B);
        for (int i = block.thread_rank(); i < N; i += Block::size()) {
          int ia = i%ncart(A);
          int ib = i/ncart(A);
          auto a = orbitals[cart::index(A)+ia];
          auto b = orbitals[cart::index(B)+ib];
          // E^{i_y j_y}_0 on each axis, then one axis at a time replaced by
          // its derivative. Every thread owns its own (ia,ib), so the three
          // components need no atomics either.
          double e[3];
#pragma unroll
          for (int x = 0; x < 3; ++x) {
            e[x] = E.value(a[x], b[x], 0, x);
          }
#pragma unroll
          for (int x = 0; x < 3; ++x) {
            double d = 2*pair.a*E.value(a[x]+1, b[x], 0, x);
            if (a[x]) d -= a[x]*E.value(a[x]-1, b[x], 0, x);
            U[uindex<A,B>(ia,ib,x)] += C*d*e[(x+1)%3]*e[(x+2)%3];
          }
        }
      }

    };

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void overlap_kernel(const GaussianPairs basis, double *V, size_t ldV) {
      // DB = 0: overlap reads E only at Hermite degree 0, so this is E2 at its
      // cheapest -- 336 doubles of shared memory at (3|3).
      compute2<Block,A,B,0,true>(basis, Overlap<A,B>{}, V, ldV);
    }

    template<int A, int B>
    void launch(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {
      using Block = thread_block< block_size<A,B,0>() >;
      dim3 grid = { (unsigned int)ab.N };
      overlap_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, V, ldV);
    }

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void overlap1_kernel(const GaussianPairs basis, double *V, size_t ldV) {
      // DA = 1: the raising relation reads E at bra degree A+1, so E2 is one
      // unit wider than the value kernel's -- 480 doubles (3.8 KB) at (3|3)
      // against 336. NC = 3: one accumulator block per Cartesian component,
      // 300 doubles at (3|3).
      compute2<Block,A,B,0,true,1,3>(basis, OverlapD1<A,B>{}, V, ldV);
    }

    template<int A, int B>
    void launch1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {
      using Block = thread_block< block_size<A,B,0,1>() >;
      dim3 grid = { (unsigned int)ab.N };
      overlap1_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, V, ldV);
    }

  }

  void overlap(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    libintx_assert(ab.first.L <= LMAX);
    libintx_assert(ab.second.L <= LMAX);
    // The host engine writes the pure layout unconditionally, so a Cartesian
    // batch has no agreed answer to be checked against; say so rather than
    // write npure-strided values for an ncart-strided caller.
    libintx_assert(ab.first.pure && ab.second.pure);

    using Kernel = std::function<void(
      const GaussianPairs&, double*, size_t, gpuStream_t
    )>;

    // (LMAX+1)^2 = 16 kernels at LMAX=3 -- the whole two-centre table in one
    // translation unit, which is why gpu/onebody/CMakeLists.txt does not need
    // the one-OBJECT-library-per-pair split the Coulomb path uses.
    static auto kernels = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, V, ldV, stream);

  }

  void overlap1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream) {

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
