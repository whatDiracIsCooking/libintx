#ifndef LIBINTX_GPU_KINETIC_KINETIC_H
#define LIBINTX_GPU_KINETIC_KINETIC_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/onebody/basis.h"

// The device electronic kinetic-energy matrix,
//
//     T[mu,nu] = -1/2 <mu| nabla^2 |nu>
//
// as Operator::Kinetic on gpu::md::IntegralEngine<2>.
//
// Same interface shape as gpu/overlap/overlap.h, and for the same reason: ONE
// non-template launcher between the engine (gpu/onebody/md2.cc, plain C++) and
// the kernel (gpu/kinetic/kinetic.cu, compiled by nvcc or hipcc), with the
// (A|B) dispatch and its (LMAX+1)^2 instantiations on the kernel side.

namespace libintx::gpu::md::onebody {

  /// Launch the kinetic-energy kernel for one bin of shell pairs.
  ///
  /// `ab` is a batch from `gpu::md::make_basis` -- one bin, so the angular
  /// momentum, the solid-harmonic flag and the contraction degree agree across
  /// every pair in it. `V` is host memory the caller has registered, written in
  /// the host engine's layout,
  ///
  ///     V[ij + (na + nb*npure(A))*ldV]
  ///
  /// Asynchronous on `stream`; the caller synchronizes.
  ///
  /// Throws if the batch is empty, is above LMAX, or is not solid-harmonic --
  /// the host `md::IntegralEngine<2>` only ever writes the pure layout.
  void kinetic(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream);

}

#endif /* LIBINTX_GPU_KINETIC_KINETIC_H */
