#ifndef LIBINTX_AO_MD_KENGINE_H
#define LIBINTX_AO_MD_KENGINE_H

#include "libintx/kengine.h"
#include "libintx/shell.h"

#include <memory>

namespace libintx::md {

  /// Host McMurchie-Davidson exchange engine: the CPU half of the K engine,
  /// driving libintx::md::IntegralEngine<4> over canonical shell quartets.
  ///
  /// The device engine (libintx::gpu::make_kengine) is the same algorithm on
  /// libintx::gpu::md::IntegralEngine<4>; both share
  /// libintx/gpu/kengine/md/driver.h, so this one is also the reference the GPU
  /// path is checked against.
  ///
  /// **Contraction coefficients must be primitive-normalized.** A basis-set
  /// library's coefficients are defined against normalized primitives, and for
  /// a contracted shell that is not an overall per-basis-function scale -- the
  /// primitives carry different exponents, so feeding the raw coefficients
  /// through is a physically different (wrong) basis, not a rescaled one, and
  /// the error survives all the way to the SCF energy. libintx::make_basis
  /// applies gto::normalized by default and that is the convention here; if
  /// you build a Basis by hand with normalize=false, K, the one-electron
  /// integrals and the J engine must all agree on that choice.
  std::unique_ptr<libintx::KEngine> make_kengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::KEngine::Screening> screening = nullptr,
    int num_threads = 1
  );

  /// Host density-fitted McMurchie-Davidson exchange engine: the same
  /// KEngine interface as make_kengine above, with the four-centre integrals
  /// factored through @p df_basis instead of computed. See
  /// libintx/gpu/kengine/md/df.h for the algorithm; the short version is
  ///
  ///   K[mu,nu] = sum_P (B_P D A_P)[mu,nu],
  ///   A[P,mu,nu] = (P|mu nu),   B = V^-1 A,   V[P,Q] = (P|Q)
  ///
  /// so it drives libintx::md::IntegralEngine<3> and a pair of GEMMs where
  /// make_kengine drives IntegralEngine<4> and a permutation digest.
  ///
  /// @p V_linv applies V^-1 over the auxiliary index; see
  /// KEngine::MetricTransform for the layout, and note that it is the caller's
  /// to supply exactly as it is for libintx::gpu::make_jengine -- the metric
  /// is a property of the auxiliary basis, not of the K build. It is required:
  /// K() throws without one.
  ///
  /// **This is an approximation** where make_kengine is not. A DF K matrix
  /// reproduces a direct one only to the quality of the auxiliary basis, so
  /// the two engines are alternatives to pick between, not implementations to
  /// check against each other to round-off.
  ///
  /// Built into libintx.md3, not libintx.md4: it needs the three-centre
  /// engine and nothing four-centre. A caller that wants both K engines links
  /// both libraries.
  std::unique_ptr<libintx::KEngine> make_df_kengine(
    const Basis<Gaussian> &basis,
    const Basis<Gaussian> &df_basis,
    libintx::KEngine::MetricTransform V_linv,
    std::shared_ptr<const libintx::KEngine::Screening> screening = nullptr,
    int num_threads = 1
  );

  /// Schwarz bounds for every canonical shell pair, sqrt(max |(ij|ij)|), as a
  /// screening object both K engines accept. @p threshold is the tau below
  /// which a quartet's max2(i,j)*max2(k,l)*max|D| is dropped; 0 disables
  /// screening entirely.
  std::shared_ptr<const libintx::KEngine::Screening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold
  );

}

#endif /* LIBINTX_AO_MD_KENGINE_H */
