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

}
