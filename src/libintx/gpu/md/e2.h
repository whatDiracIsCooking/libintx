#ifndef LIBINTX_GPU_MD_E2_H
#define LIBINTX_GPU_MD_E2_H

#include "libintx/array.h"
#include "libintx/gpu/api/thread_group.h"

#include <cassert>

namespace libintx::gpu::md {

  /// Device McMurchie-Davidson expansion coefficients for one primitive pair.
  ///
  /// `E(a,b,p)` is the product over the three Cartesian axes of `E^{a_x b_x}_{p_x}`,
  /// the coefficient of the Hermite Gaussian `p` in the expansion of the
  /// Cartesian primitive product `a*b`. The upward recursion needs the whole
  /// triangle `t = 0 .. i+j`, so the storage is always
  ///
  ///     (A+1)*(Bmax+1)*(A+Bmax+1)*3 doubles
  ///
  /// even for a caller that only ever reads `t = 0`. At `A = Bmax = 3` that is
  /// 336 doubles (2.7 KB); the kinetic operator's `DB = 2` variant at
  /// `A = B = 3` is 648 doubles (5.2 KB). Both are comfortably inside
  /// LIBINTX_GPU_MAX_SHMEM (49152 by default) -- this is meant to live in
  /// shared memory, one instance per thread block.
  ///
  /// @tparam A  bra angular momentum.
  /// @tparam B  ket angular momentum.
  /// @tparam DB extra ket degree. The Coulomb path wants `DB = 0`; the kinetic
  ///         operator differentiates the ket twice and so needs `E^{ij}_0` for
  ///         `j` up to `B+2`, which the host spells `libintx::md::E2<T,A,B+2,0>`
  ///         (`src/libintx/ao/md/md2.cc`). Carrying the extra degree as its own
  ///         template parameter keeps that requirement visible at the use site
  ///         instead of hiding it in a hardcoded `B`.
  template<int A, int B, int DB = 0>
  struct E2 {

    /// Highest ket degree this instance spans.
    static constexpr int Bmax = B + DB;
    /// Highest Hermite index this instance spans.
    static constexpr int Pmax = A + Bmax;

    __device__
    auto& value(int i, int j, int k, int x) {
      constexpr int strides[4] = {
        (Pmax+1)*(Bmax+1),
        (Pmax+1),
        1,
        (Pmax+1)*(Bmax+1)*(A+1)
      };
      return data[i*strides[0]+j*strides[1]+k*strides[2]+x*strides[3]];
    }

    template<typename T>
    __device__
    auto operator()(T &&a, T &&b, T &&p) {
      double v = 1;
      for (int i = 0; i < 3; ++i) {
        v *= value(a[i], b[i], p[i], i);
      }
      return v;
    }

    /// Build the coefficients for the primitive pair (a, r_a), (b, r_b).
    ///
    /// @param r  `r_a - r_b`, the bra-minus-ket separation.
    ///
    /// Thread-group contract. This is a real constraint on the block shape of
    /// every caller, not a formality:
    ///
    ///  - `G::size() >= A+Bmax+1`. The recursion is parallelised over the
    ///    Hermite index `t`, one thread per `t`, so a narrower group would
    ///    silently drop coefficients. Asserted at compile time.
    ///  - Every thread of `thread_group` must call `init`, including the ones
    ///    outside the `t <= i+j` range for a given step: the recursion syncs
    ///    between its `i` and `j` steps and all threads take part in the syncs.
    ///  - `init` syncs on entry and between steps but NOT on exit. The caller
    ///    must `sync()` before reading a coefficient written by another thread.
    template<typename G>
    __device__
    void init(double a, double b, const auto &r, const G &thread_group) {
      static_assert(G::size() >= (A+Bmax+1));
      auto p = a + b;
      assert(p);
      auto q = ((a ? a : 1)*(b ? b : 1))/p;
      assert(q);
      fill(3*(Pmax+1)*(Bmax+1)*(A+1), this->data, 0, thread_group);
      thread_group.sync();
      if (thread_group.thread_rank() == 0) {
        value(0,0,0,0) = 1;
        value(0,0,0,1) = 1;
        value(0,0,0,2) = 1;
      }
      thread_group.sync();
      auto k = thread_group.thread_rank();
      for (int i = 1; i <= A; ++i) {
        thread_group.sync();
        if (k > i) continue;
#pragma unroll
        for (int x = 0; x < 3; ++x) {
          double v0 = (k ? value(i-1,0,k-1,x) : 0);
          double v1 = value(i-1,0,k,x);
          double v2 = (k < i ? value(i-1,0,k+1,x) : 0);
          double v = (1/(2*p))*v0 - (q*r[x]/a)*v1 + (k+1)*v2;
          value(i,0,k,x) = v;
        }
      }
      // j
      for (int j = 1; j <= Bmax; ++j) {
        for (int i = 0; i <= A; ++i) {
          thread_group.sync();
          if (k > i+j) continue;
          for (int x = 0; x < 3; ++x) {
            double v0 = (k ? value(i,j-1,k-1,x) : 0);
            double v1 = value(i,j-1,k,x);
            double v2 = (k < i+j ? value(i,j-1,k+1,x) : 0);
            double v = (1/(2*p))*v0 + (q*r[x]/b)*v1 + (k+1)*v2;
            value(i,j,k,x) = v;
          }
        }
      }
    }

    double data[(A+1)*(Bmax+1)*(A+Bmax+1)*3];

  };

}

#endif /* LIBINTX_GPU_MD_E2_H */
