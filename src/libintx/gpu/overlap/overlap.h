#ifndef LIBINTX_GPU_OVERLAP_OVERLAP_H
#define LIBINTX_GPU_OVERLAP_OVERLAP_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/onebody/basis.h"

// The device overlap matrix,
//
//     S[mu,nu] = <mu|nu> = integral phi_mu(r) phi_nu(r) dr
//
// as Operator::Overlap on gpu::md::IntegralEngine<2>.
//
// This declaration is the whole interface between the engine
// (gpu/onebody/md2.cc, plain C++) and the kernel (gpu/overlap/overlap.cu,
// compiled by nvcc or hipcc): ONE non-template launcher, taking a bin the
// engine has already uploaded. The (A|B) dispatch sits on the kernel side,
// with the (LMAX+1)^2 instantiations it selects between, because a function
// template crossing this boundary would have to be explicitly instantiated
// over a table whose size is a configure-time decision.

namespace libintx::gpu::md::onebody {

  /// Launch the overlap kernel for one bin of shell pairs.
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
  /// the host `md::IntegralEngine<2>` only ever writes the pure layout, and a
  /// device engine answering a Cartesian batch in some other layout would not
  /// be the drop-in replacement it is meant to be.
  void overlap(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream);

}

#endif /* LIBINTX_GPU_OVERLAP_OVERLAP_H */
