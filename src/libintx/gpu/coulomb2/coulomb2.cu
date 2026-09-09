// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/coulomb2/coulomb2.h"
#include "libintx/gpu/onebody/kernel.h"
#include "libintx/gpu/md/e2.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/gpu/boys.h"

#include "libintx/ao/md/r1.h"
#include "libintx/array.h"
#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/orbital.h"
#include "libintx/pure.transform.h"
#include "libintx/utility.h"

#include <cmath>
#include <functional>

// The two-centre Coulomb metric (P|Q) on the device.
//
// Per primitive pair (a on r_A, b on r_B), following
// libintx::md::reference::compute(P, Unit, Q, Unit, ...) -- the four-centre
// reference with a unit shell in each ket slot, which is what
// tests/kengine.test.h already builds the metric out of:
//
//   1. alpha = a*b/(a+b),  PQ = r_A - r_B
//   2. s[m] = F_m(alpha*|PQ|^2) * (-2*alpha)^m,  m = 0..A+B
//      R[t] = r1(PQ, s)[t]
//   3. E^A_t(a), E^B_u(b): the ONE-centre Hermite expansions of the two shells
//   4. (a|b) = C_a*C_b * 2*pi^(5/2)/(a*b*sqrt(a+b))
//              * sum_{t<=A} sum_{u<=B} E^A_t E^B_u (-1)^|u| R[t+u]
//
// Neither hard piece is reimplemented. `gpu::boys()` is the tree's one device
// Chebyshev table; `libintx::md::r1::visit` is LIBINTX_GPU_ENABLED and is
// literally what the device Coulomb kernels call, so the bra/ket conventions
// cannot drift between this and the three- and four-centre paths.
//
// WHY THIS DOES NOT USE gpu/onebody/kernel.h's compute2 SKELETON.
//
// That skeleton is written against a PRODUCT DENSITY: it hands the operator
// body one `E2<A+DA,B,DB>` built from (a, b, r_A-r_B) and a coefficient
// `C = C_a*C_b*exp(-a*b/(a+b)*|r_A-r_B|^2)`. Both are exactly wrong here.
// `P` and `Q` sit on opposite sides of 1/r12, so
//
//   - the expansion wanted is two independent one-centre ones, `E2<A,0,0>` at
//     (a, 0) and `E2<B,0,0>` at (b, 0) -- different objects, not one;
//   - there is no Gaussian product and so no `K_ab` pre-factor. Recovering
//     `C_a*C_b` by dividing the skeleton's `C` back out is not an option: for a
//     well separated, sharply contracted pair `K_ab` underflows to zero and the
//     division is 0/0, which is precisely the pair a screened build keeps.
//
// So the block driver below is this kernel's own. What it does share is
// everything that is genuinely about the ENGINE rather than the operator:
// `GaussianPairs`/`Gaussian2`, `block_size<A,B,DB>()`, `uindex<A,B>` and the
// output layout, all from gpu/onebody/kernel.h. The ~15 duplicated lines are
// the batch load and the cartesian-to-pure store; the alternative was a third
// mode on `compute2` whose two existing modes are both product densities.

namespace libintx::gpu::md::onebody {

  namespace {

    namespace cart = cartesian;
    namespace r1 = libintx::md::r1;

    /// The angular momentum this operator has to span.
    ///
    /// (P|Q) is the density-fitting metric, so its shells come from the
    /// AUXILIARY basis and reach XMAX (`LIBINTX_MAX_X`, default `LMAX+1`), not
    /// LMAX -- which is why the dispatch table below is (CMAX+1)^2 and not
    /// (LMAX+1)^2 like the other three operators'. `max` rather than plain
    /// XMAX because nothing stops a caller handing this engine an
    /// orbital-basis pair, and XMAX < LMAX is a legal (if odd) configuration.
    constexpr int CMAX = libintx::max(LMAX,XMAX);

    // The Cartesian orbital list as device-resident data, indexed by
    // cart::index(L) + i, the same as gpu/overlap/overlap.cu's copy. Only
    // single shells are indexed and this operator's reach CMAX, so CMAX spans
    // it -- LMAX would not.
    __device__
    constexpr auto orbitals = hermite::orbitals2<CMAX>;

    /// Block-level driver for one bin of (P|Q).
    ///
    /// One thread block per shell pair, `blockIdx.x` the pair index; see the
    /// file header for why this is not `compute2`.
    ///
    /// Parallelisation. Two axes are available -- the Hermite recursion (over
    /// the Hermite index, inside `E2::init`) and the Cartesian contraction
    /// (over `ncart(A)*ncart(B)` component pairs), and both are used. The one
    /// serial part is the Boys evaluation plus `r1::visit<A+B>`, which runs on
    /// thread 0: unlike the electron-nuclear kernel there is no third axis to
    /// spread it over -- a nuclear potential has O(10-100) nuclei per primitive
    /// pair, a metric element has exactly one `(PQ, alpha)`. Every thread
    /// computing it redundantly would take the same wall clock, so it is done
    /// once. NOTHING HERE HAS BEEN BENCHMARKED (there is no device in the
    /// environment it was written in); read the shape as a starting point.
    template<typename Block, int A, int B>
    __device__
    void compute2_coulomb(const GaussianPairs &basis, Boys boys, double *V, size_t ldV) {

      // E2<A,0,0>::init parallelises its recursion over t = 0..A, and
      // E2<B,0,0>::init over t = 0..B, so the block has to cover both.
      static_assert(Block::size() >= A+1);
      static_assert(Block::size() >= B+1);
      // gpu::boys() is sized for max(4*LMAX, 2*LMAX+XMAX); (P|Q) at the top of
      // this operator's range needs order 2*CMAX. With the default
      // XMAX = LMAX+1 that holds for every LMAX >= 1. It fails only for
      // XMAX > 2*LMAX, where the fix is to widen the ONE device Chebyshev
      // table in gpu/boys.h rather than stand up a second one.
      static_assert(
        2*CMAX <= libintx::max(4*LMAX, 2*LMAX+XMAX),
        "gpu::boys() is not sized for (P|Q) at this LIBINTX_MAX_X;"
        " widen libintx::gpu::Boys (LIBINTX_MAX_X <= 2*LIBINTX_MAX_L holds)"
      );

      constexpr int L = A + B;
      constexpr int NR = nherm2(L);
      constexpr int NU = ncart(A)*ncart(B);

      auto block = Block();

      // Gaussian carries default member initializers, so a bare
      // `__shared__ Gaussian2` has a non-empty constructor and nvcc rejects
      // it. Same union dodge gpu/md/basis.cu and compute2 use.
      __shared__
      union shmem {
        Gaussian2 ab;
        __device__ shmem() {}
      } shmem;
      auto &ab = shmem.ab;

      /// r_A - r_B, the separation the Boys argument and R are built on. Note
      /// this is P-Q, the two SHELL centres: with a unit ket in each slot the
      /// centre of charge of each side is the shell's own centre.
      __shared__ array<double,3> PQ;
      __shared__ double U[NU];
      __shared__ double R[NR];
      // The two one-centre expansions. E2<A,0,0> is (A+1)^2*3 doubles -- 75 at
      // A = 4, against E2<4,4,0>'s 2025 -- so the pair of them is cheaper in
      // shared memory than the single product expansion compute2 would build.
      __shared__ E2<A,0,0> Ea;
      __shared__ E2<B,0,0> Eb;

      memcpy1(&basis.data[blockIdx.x], &ab, block);
      fill(NU, U, 0.0, block);
      block.sync();

      if (block.thread_rank() == 0) {
        PQ = ab.r.first - ab.r.second;
      }
      block.sync();

      // The one-centre expansion is E^{i,0}_t at zero separation: a single
      // Gaussian is its own product, so the recursion's `r` is zero and its
      // ket degree is zero. That is exactly what the reference does by passing
      // a Unit shell -- `E_recurrence` rewrites R to 0 when either exponent
      // vanishes -- and it is why nothing here divides by the ket exponent.
      const array<double,3> zero = { 0, 0, 0 };

      for (int ki = 0; ki < ab.first.K; ++ki) {
        for (int kj = 0; kj < ab.second.K; ++kj) {

          // Both shells are already in shared memory, so every thread reads
          // its own copy of the primitive rather than one thread broadcasting
          // through shared memory: unlike compute2's `PrimitivePair` there is
          // nothing to compute here, no exp() and no centre of charge.
          const double a = ab.first.prims[ki].a;
          const double Ca = ab.first.prims[ki].C;
          const double b = ab.second.prims[kj].a;
          const double Cb = ab.second.prims[kj].C;

          Ea.init(a, 0.0, zero, block);
          Eb.init(b, 0.0, zero, block);

          fill(NR, R, 0.0, block);
          block.sync();

          if (block.thread_rank() == 0) {
            const double alpha = (a*b)/(a+b);
            double s[L+1] = {};
            // T = 0 -- two auxiliary shells on the same centre, the diagonal
            // of the metric -- is the interpolated branch of the Chebyshev
            // table, not the asymptotic one. Finite, and the classic way a
            // metric build goes wrong.
            boys.template compute<L>(alpha*norm(PQ), 0, s);
            double f = 1;
            for (int m = 0; m <= L; ++m) {
              s[m] *= f;
              f *= -2*alpha;
            }
            r1::visit<L>(
              [&](auto &&r) { R[r.index] = r.value; },
              PQ, s
            );
          }
          block.sync();

          // 2*pi^(5/2)/(p*q*sqrt(p+q)) with p = a, q = b -- the reference's
          // sqrt(4*pi^5)/sqrt(p^2*q^2*(p+q)), written without the square root
          // of a fifth power.
          const double C = (
            Ca*Cb*(2*math::pi*math::pi*std::sqrt(math::pi))/(a*b*std::sqrt(a+b))
          );

          for (int i = block.thread_rank(); i < NU; i += Block::size()) {
            int ia = i%ncart(A);
            int ib = i/ncart(A);
            auto ca = orbitals[cart::index(A)+ia];
            auto cb = orbitals[cart::index(B)+ib];
            // E^{i,0}_t is contiguous in t (E2's k stride is 1) and terminates
            // at t = i, one axis at a time.
            const double *Eax = &Ea.value(ca[0], 0, 0, 0);
            const double *Eay = &Ea.value(ca[1], 0, 0, 1);
            const double *Eaz = &Ea.value(ca[2], 0, 0, 2);
            const double *Ebx = &Eb.value(cb[0], 0, 0, 0);
            const double *Eby = &Eb.value(cb[1], 0, 0, 1);
            const double *Ebz = &Eb.value(cb[2], 0, 0, 2);
            double v = 0;
            for (int tz = 0; tz <= ca[2]; ++tz) {
              for (int ty = 0; ty <= ca[1]; ++ty) {
                double Eayz = Eay[ty]*Eaz[tz];
                for (int tx = 0; tx <= ca[0]; ++tx) {
                  double Ea3 = Eax[tx]*Eayz;
                  if (!Ea3) continue;
                  for (int uz = 0; uz <= cb[2]; ++uz) {
                    for (int uy = 0; uy <= cb[1]; ++uy) {
                      double Ebyz = Eby[uy]*Ebz[uz];
                      for (int ux = 0; ux <= cb[0]; ++ux) {
                        // The (-1)^|u| that puts the ket's Hermite derivative
                        // on the ket side of 1/r12 -- r1::visit's own `phase`,
                        // spelled out here because R is indexed by t+u.
                        double phase = ((ux+uy+uz)%2 ? -1.0 : +1.0);
                        auto t = Orbital{{
                          (uint8_t)(tx+ux), (uint8_t)(ty+uy), (uint8_t)(tz+uz)
                        }};
                        v += Ea3*Ebx[ux]*Ebyz*phase*R[hermite::index2(t)];
                      }
                    }
                  }
                }
              }
            }
            U[uindex<A,B>(ia,ib)] += C*v;
          }
          block.sync();

        }
      }

      // The cartesian-to-pure store, byte for byte compute2's `Pure` branch at
      // NC = 1: [ia,ib] -> [ib,pa] -> [pa,pb], the intermediate transposed so
      // each cartesian_to_pure<B> reads a contiguous ket slice.
      {
        constexpr int NA = npure(A);
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

    template<typename Block, int A, int B>
    __global__
    __launch_bounds__(Block::size())
    void coulomb2_kernel(
      const GaussianPairs basis,
      const Boys boys,
      double *V,
      size_t ldV)
    {
      compute2_coulomb<Block,A,B>(basis, boys, V, ldV);
    }

    template<int A, int B>
    void launch(
      const GaussianPairs &ab,
      double *V,
      size_t ldV,
      gpuStream_t stream)
    {
      // gpu/onebody/kernel.h's shared block shape at DB = 0: one block per
      // shell pair, a whole number of warps capped at 128. The block has to
      // cover both one-centre recursions (A+1 and B+1 threads); a warp covers
      // either at any CMAX this tree can be configured for.
      //
      // Shared memory at the top of the range, (4|4) with LMAX=3, XMAX=4:
      // Gaussian2 ~0.5 KB + U 225 + R 165 + Ea 75 + Eb 75 + S 135 doubles,
      // about 5.9 KB against LIBINTX_GPU_MAX_SHMEM's 49152.
      using Block = thread_block< block_size<A,B,0>() >;
      dim3 grid = { (unsigned int)ab.N };
      coulomb2_kernel<Block,A,B><<<grid,Block(),0,stream>>>(ab, gpu::boys(), V, ldV);
    }

  }

  void coulomb2(
    const GaussianPairs &ab,
    double *V,
    size_t ldV,
    gpuStream_t stream)
  {

    libintx_assert(ab.N > 0);
    libintx_assert(ab.K > 0);
    // CMAX, not LMAX: the auxiliary basis this operator exists for reaches
    // XMAX.
    libintx_assert(ab.first.L <= CMAX);
    libintx_assert(ab.second.L <= CMAX);
    // The host engine writes the pure layout unconditionally, so a Cartesian
    // batch has no agreed answer to be checked against -- and a DF metric is
    // solid-harmonic in every caller this exists for.
    libintx_assert(ab.first.pure && ab.second.pure);

    using Kernel = std::function<void(
      const GaussianPairs&, double*, size_t, gpuStream_t
    )>;

    // (CMAX+1)^2 = 25 kernels at LMAX=3, XMAX=4 -- one more unit of angular
    // momentum in each index than the other three operators' (LMAX+1)^2,
    // because this one is handed auxiliary shells.
    static auto kernels = make_array<Kernel,CMAX+1,CMAX+1>(
      [](auto a, auto b) {
        return Kernel(&launch<a,b>);
      }
    );

    kernels[ab.first.L][ab.second.L](ab, V, ldV, stream);

  }

}
