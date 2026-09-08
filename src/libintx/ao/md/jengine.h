#ifndef LIBINTX_AO_MD_JENGINE_H
#define LIBINTX_AO_MD_JENGINE_H

#include "libintx/ao/md/screening.h"
#include "libintx/jengine.h"
#include "libintx/shell.h"

#include <memory>

namespace libintx::md {

  /// Host McMurchie-Davidson Coulomb engine, **conventional** (integral
  /// direct): four centres, no auxiliary basis, no metric solve.
  ///
  ///   J[mu,nu] = sum_{lambda,sigma} (mu nu | lambda sigma) D[lambda,sigma]
  ///
  /// This is the J that pairs exactly with libintx::md::make_kengine -- the
  /// same driver (libintx/fock/md/driver.h) over the same
  /// libintx::md::IntegralEngine<4>, differing only in which two of the four
  /// permuted slots accumulate and which two contract. A caller assembling
  /// F = H + J - K/2 without density fitting gets both halves from the same
  /// integrals, with no fitting error on one side to reconcile against an
  /// exact other side.
  ///
  /// The density-fitted alternative is libintx::gpu::make_df_jengine, which is
  /// a different algorithm (two three-centre passes around a fitted vector)
  /// behind the same libintx::JEngine interface. It is GPU only, and it needs
  /// an auxiliary basis and V^-1; this one needs neither.
  ///
  /// The primitive-normalization note on libintx::md::make_kengine applies
  /// here verbatim: J, K and the one-electron integrals must all agree on the
  /// convention, and libintx::make_basis normalizes by default.
  std::unique_ptr<libintx::JEngine> make_jengine(
    const Basis<Gaussian> &basis,
    std::shared_ptr<const libintx::PairScreening> screening = nullptr,
    int num_threads = 1
  );

}

#endif /* LIBINTX_AO_MD_JENGINE_H */
