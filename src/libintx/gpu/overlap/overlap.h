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

  /// Launch the overlap *gradient* kernel for one bin of shell pairs.
  ///
  /// `dS/dA_x` -- the derivative with respect to the **bra** centre, three
  /// components, written in the value layout with the component as the
  /// slowest index:
  ///
  ///     V[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV]
  ///
  /// The ket derivative is not computed and does not need to be: a 2-centre
  /// integral depends on the centres only through `r_a - r_b`, so
  /// `dS/dB = -dS/dA` elementwise. See `ao::IntegralEngine<2>::compute1`.
  ///
  /// Same batch contract, same guards and the same solid-harmonic requirement
  /// as `overlap` above. The requirement stands here for the same reason and
  /// costs nothing extra: `d/dA_x` reaches Cartesian bra components at
  /// `L+1`, but only *inside* the kernel, through the `E^{i+1,j}` coefficients
  /// -- the accumulator, the shell and the output all stay at `L`, so there is
  /// no Cartesian intermediate for a caller to see. Nothing here shifts a
  /// shell, so nothing here can re-normalize one.
  void overlap1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream);

}

#endif /* LIBINTX_GPU_OVERLAP_OVERLAP_H */
