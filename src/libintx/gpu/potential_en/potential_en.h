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

  /// Launch the electron-nuclear potential *gradient* kernels for one bin.
  ///
  /// `V` has TWO derivative contributions and both are computed here, into two
  /// separate buffers, because they are indexed by different things.
  ///
  ///  1. **The basis functions move.** `dV/dA_x`, the derivative with respect
  ///     to the **bra** centre, through the same raising relation the overlap
  ///     gradient uses -- three components, written to `dV` in the value layout
  ///     with the component as the slowest index:
  ///
  ///         dV[ij + (na + nb*npure(A) + x*npure(A)*npure(B))*ldV]
  ///
  ///  2. **The operator moves.** `dV/dR_C,x`, the Hellmann-Feynman term:
  ///     `1/|r - R_C|` depends on the nuclear position directly, so a nucleus
  ///     contributes to the gradient even when it carries no basis function.
  ///     This one is indexed by *nucleus*, which does not fit (1)'s shape, so
  ///     it goes to `dVC` as `ncenters` consecutive copies of that block --
  ///     the nucleus is one more, slowest index:
  ///
  ///         dVC[ij + (na + nb*npure(A) + x*npure(A)*npure(B)
  ///                   + c*3*npure(A)*npure(B))*ldV]
  ///
  ///     `c` indexes `centers` in the order `set()` was given them, which is
  ///     the convention `Nuclear::Operator::Parameters` now states: entry `i`
  ///     is atom `i`. `dVC` must therefore be `ldV*npure(A)*npure(B)*3*ncenters`
  ///     doubles.
  ///
  /// **`dV/dB` is NOT the negative of `dV/dA` for this operator.** That
  /// identity holds for `S` and `T` because they depend on the centres only
  /// through `r_a - r_b`; `V` depends on `R_C` as well, and what vanishes is
  /// the three-way sum `dV/dA + dV/dB + sum_C dV/dR_C`. A caller assembling a
  /// gradient obtains `dV/dB` the way it obtains `dS/dB`: from the transposed
  /// bin, `V(a,b) = V(b,a)`.
  ///
  /// Both contributions land in the same atom's slot for an atom that carries
  /// both a nucleus and basis functions -- every atom in a normal molecule --
  /// so the caller's scatter must accumulate, not assign.
  ///
  /// Same batch contract, same guards and the same solid-harmonic requirement
  /// as `potential_en` above. Asynchronous on `stream`; the caller
  /// synchronizes.
  void potential_en1(
    const GaussianPairs &ab,
    const NuclearCenter *centers,
    int ncenters,
    double *dV,
    double *dVC,
    size_t ldV,
    gpuStream_t stream
  );

}

#endif /* LIBINTX_GPU_POTENTIAL_EN_POTENTIAL_EN_H */
