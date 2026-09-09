#ifndef LIBINTX_GPU_COULOMB2_COULOMB2_H
#define LIBINTX_GPU_COULOMB2_COULOMB2_H

#include "libintx/gpu/forward.h"
#include "libintx/gpu/onebody/basis.h"

// The device two-centre Coulomb metric,
//
//     V[P,Q] = (P|Q) = int int P(r1) 1/|r1-r2| Q(r2) dr1 dr2
//
// as Operator::Coulomb on gpu::md::IntegralEngine<2>.
//
// Same shape of interface as gpu/overlap/overlap.h and
// gpu/potential_en/potential_en.h -- ONE non-template launcher taking a bin the
// engine has already uploaded, with the (A|B) dispatch on the kernel side,
// because a function template crossing the nvcc/host boundary would have to be
// explicitly instantiated over a table whose size is a configure-time decision.
//
// Two things set this operator apart from the other three, and both are visible
// from here:
//
//  - It is the one two-ELECTRON operator on a two-CENTRE engine. The shell pair
//    a batch carries is not a product density: `P` and `Q` sit on opposite sides
//    of 1/r12, so there is no Gaussian product, no `K_ab` pre-factor, and the
//    Hermite expansion is two ONE-centre expansions rather than one two-centre
//    one. That is why the kernel does not build on gpu/onebody/kernel.h's
//    `compute2`, which is written against a product density; see coulomb2.cu.
//
//  - Its shells come from an AUXILIARY basis, whose angular momentum reaches
//    XMAX (`LIBINTX_MAX_X`, default LMAX+1), not LMAX. The (A|B) table is sized
//    for max(LMAX,XMAX) accordingly.
//
// The consumer is the density-fitting metric every DF caller has to build:
// gpu::make_df_jengine and gpu::make_df_kengine both take V^-1 from the caller,
// and until this landed there was no device route to V at all.

namespace libintx::gpu::md::onebody {

  /// Launch the two-centre Coulomb kernel for one bin of shell pairs.
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
  /// Throws if the batch is empty, is above max(LMAX,XMAX), or is not
  /// solid-harmonic.
  void coulomb2(
    const GaussianPairs &ab,
    double *V,
    size_t ldV,
    gpuStream_t stream
  );

}

#endif /* LIBINTX_GPU_COULOMB2_COULOMB2_H */
