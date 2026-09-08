#ifndef LIBINTX_AO_MD_KENGINE_H
#define LIBINTX_AO_MD_KENGINE_H

#include "libintx/ao/md/screening.h"
#include "libintx/kengine.h"
#include "libintx/shell.h"

#include <memory>

namespace libintx::md {

  /// Host McMurchie-Davidson exchange engine: the CPU half of the K engine,
  /// driving libintx::md::IntegralEngine<4> over canonical shell quartets.
  ///
  /// The device engine (libintx::gpu::make_kengine) is the same algorithm on
  /// libintx::gpu::md::IntegralEngine<4>; both share
  /// libintx/fock/md/driver.h, so this one is also the reference the GPU
  /// path is checked against. libintx::md::make_jengine is its Coulomb
  /// counterpart -- the same driver with the other scatter.
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
    std::shared_ptr<const libintx::PairScreening> screening = nullptr,
    int num_threads = 1
  );

}

#endif /* LIBINTX_AO_MD_KENGINE_H */
