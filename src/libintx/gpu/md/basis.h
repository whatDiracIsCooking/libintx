#ifndef LIBINTX_GPU_MD_BASIS_H
#define LIBINTX_GPU_MD_BASIS_H

#include "libintx/shell.h"
#include "libintx/gpu/forward.h"
#include "libintx/gpu/api/api.h"

namespace libintx::gpu::md {

  /// A shell pair as raw primitives: the two shells (angular momentum, the
  /// solid-harmonic flag and the contraction) plus their centres.
  ///
  /// This is what a host-to-device upload of a 2-centre batch carries. The
  /// Coulomb path immediately bakes it into Hermite coefficients (see
  /// `Basis2` below); the one-electron kernels consume it as-is, because each
  /// of them wants a different slice of E and building it per primitive pair
  /// in shared memory is cheaper than materialising the union.
  struct Gaussian2 {
    Gaussian first, second;
    struct {
      array<double,3> first, second;
    } r;
  };

  struct alignas(8) Hermite {
    double exp;
    double C;
    array<double,3> r;
    double inv_2_exp;

    LIBINTX_GPU_ENABLED
    static auto* hdata(double *p) {
      return reinterpret_cast<Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* hdata(const double *p) {
      return reinterpret_cast<const Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(double *p) {
      return reinterpret_cast<double*>(hdata(p)+1);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(const double *p) {
      return reinterpret_cast<const double*>(hdata(p)+1);
    }

    /// Doubles per (shell pair, primitive pair): this header plus the
    /// `nbf(A)*nbf(B)` by `nherm2(A.L+B.L+dL)` coefficient block.
    ///
    /// `dL` is the *extra* Hermite degree the block carries beyond the pair's
    /// own angular momentum. It is 0 for a value batch and 1 for a derivative
    /// batch: differentiating with respect to a centre raises the Hermite
    /// index by one while the basis-function extent stays `nbf(A)*nbf(B)`.
    /// That decoupling of the Hermite range from the shell pair is the whole
    /// of Route A -- see `make_basis1` below.
    LIBINTX_GPU_ENABLED
    static constexpr size_t extent(const Shell &A, const Shell &B, int dL = 0) {
      return (sizeof(Hermite)/sizeof(double) + nbf(A)*nbf(B)*nherm2(A.L+B.L+dL));
    }

  };

  struct Basis1 {
    const int L, K, N;
    const Hermite *data;
  };

  struct Basis2 {
    static constexpr size_t alignment = 128;
    const Shell first, second;
    const int N, K;
    const double *data;
    const size_t k_stride;
    /// The `nbf(A)*nbf(B)` by `ncart(A.L+B.L)` cartesian-to-pure matrix, which
    /// the value kernels use as a closed form for the *top* Hermite block
    /// (`E^{ab}_t` at `|t| = A+B` is `(1/2p)^{A+B}` times this).
    ///
    /// **A derivative batch leaves this null**: its top block is not a
    /// multiple of the pure transform, so a kernel that takes the shortcut is
    /// wrong on one. `compute1` uses only the fully generic path, which reads
    /// every Hermite degree out of `data`.
    const double *pure_transform;
    /// Extra Hermite degree carried by `data`; see `Hermite::extent`. 0 for a
    /// value batch, 1 for a derivative batch. The consumer must instantiate
    /// its kernel at `first.L + second.L + dL`.
    const int dL = 0;
  };

  Basis1 make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    device::vector<Hermite> &H,
    gpuStream_t
  );

  Basis2 make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    device::vector<double> &H,
    gpuStream_t
  );

  /// First-derivative shell-pair batch: `make_basis` with the Hermite
  /// coefficients `E^{ab}_t` replaced by
  ///
  ///     D^{ab,x}_t = 2a E^{(a+1_x)b}_t - i_x E^{(a-1_x)b}_t     (centre = 0)
  ///     D^{ab,x}_t = 2b E^{a(b+1_x)}_t - j_x E^{a(b-1_x)}_t     (centre = 1)
  ///
  /// i.e. the McMurchie-Davidson expansion of `d/dA_x` (resp. `d/dB_x`) of the
  /// primitive product, from `d/dA_x G_a = 2a G_{a+1_x} - i_x G_{a-1_x}`.
  /// The solid-harmonic transform is applied to it exactly as it is to a
  /// value, because it is linear with constant coefficients and so commutes
  /// with `d/dX`; no shell is shifted, no `L+1` shell is built, and no
  /// re-normalization trap is entered.
  ///
  /// Everything else about the batch is byte-identical to the value one -- the
  /// same `Hermite` header, so the same `p`, `P`, `C` and `K_ab` -- and the
  /// only structural difference is that the coefficient block spans
  /// `nherm2(A+B+1)` rather than `nherm2(A+B)` Hermite indices, which is what
  /// `Basis2::dL` records.
  ///
  /// @param centre 0 for the pair's first shell, 1 for its second.
  /// @param x      Cartesian component, 0..2.
  Basis2 make_basis1(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    int centre,
    int x,
    device::vector<double> &H,
    gpuStream_t
  );

}


#endif /* LIBINTX_GPU_MD_BASIS_H */
