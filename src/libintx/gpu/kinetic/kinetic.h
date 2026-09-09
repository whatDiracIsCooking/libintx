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

  /// Launch the kinetic-energy *gradient* kernel for one bin of shell pairs.
  ///
  /// `dT/dA_x` -- the derivative with respect to the **bra** centre, three
  /// components, written in the value layout with the component as the
  /// slowest index:
  ///
  ///     V[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV]
  ///
  /// The ket derivative is not computed and does not need to be: a 2-centre
  /// integral depends on the centres only through `r_a - r_b`, so
  /// `dT/dB = -dT/dA` elementwise. See `ao::IntegralEngine<2>::compute1`.
  ///
  /// Same batch contract, same guards and the same solid-harmonic requirement
  /// as `kinetic` above, and for the same reasons. `d/dA_x` reaches Cartesian
  /// bra components at `L+1`, but only *inside* the kernel, through the
  /// `E^{i+1,j}` coefficients of an `E2<A+1,B,2>` -- the accumulator, the
  /// shell and the output all stay at `L`. Nothing here shifts a shell, so
  /// nothing here can re-normalize one.
  void kinetic1(const GaussianPairs &ab, double *V, size_t ldV, gpuStream_t stream);

}

#endif /* LIBINTX_GPU_KINETIC_KINETIC_H */
