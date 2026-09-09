#ifndef LIBINTX_GPU_POTENTIAL_EN_POTENTIAL_EN_H
#define LIBINTX_GPU_POTENTIAL_EN_POTENTIAL_EN_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/onebody/basis.h"

// The device electron-nuclear attraction matrix,
//
//     V[mu,nu] = - sum_C Z_C * integral phi_mu(r) phi_nu(r) / |r - R_C| dr
//
// as Operator::Nuclear on gpu::md::IntegralEngine<2>.
//
// Same shape of interface as gpu/overlap/overlap.h -- ONE non-template launcher
// taking a bin the engine has already uploaded, with the (A|B) dispatch on the
// kernel side, because a function template crossing the nvcc/host boundary
// would have to be explicitly instantiated over a table whose size is a
// configure-time decision. The one addition is the point-charge array: the
// nuclear parameters arrive through the engine's `set()`, once per geometry,
// and live on the device across `compute` calls.

namespace libintx::gpu::md::onebody {

  /// Launch the electron-nuclear potential kernel for one bin of shell pairs.
  ///
  /// `ab` is a batch from `gpu::md::make_basis` -- one bin, so the angular
  /// momentum, the solid-harmonic flag and the contraction degree agree across
  /// every pair in it. `centers` is a *device* pointer to `ncenters` point
  /// charges, the array `gpu::md::make_centers` uploaded from the engine's
  /// `set()`. `V` is host memory the caller has registered, written in the host
  /// engine's layout,
  ///
  ///     V[ij + (na + nb*npure(A))*ldV]
  ///
  /// Asynchronous on `stream`; the caller synchronizes.
  ///
  /// Throws if the batch is empty, is above LMAX, is not solid-harmonic, or if
  /// there are no point charges -- the last is `compute(Nuclear, ...)` before
  /// any `set()`, which must say so rather than read an empty `device::vector`
  /// and hand back a buffer of zeros.
  void potential_en(
    const GaussianPairs &ab,
    const NuclearCenter *centers,
    int ncenters,
    double *V,
    size_t ldV,
    gpuStream_t stream
  );

}

#endif /* LIBINTX_GPU_POTENTIAL_EN_POTENTIAL_EN_H */
