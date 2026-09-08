#ifndef LIBINTX_AO_MD_SCREENING_H
#define LIBINTX_AO_MD_SCREENING_H

#include "libintx/screening.h"
#include "libintx/shell.h"

#include <memory>

namespace libintx::md {

  /// Schwarz bounds for every canonical shell pair, sqrt(max |(ij|ij)|),
  /// evaluated with the host four-centre MD engine.
  ///
  /// @p threshold is the tau below which a quartet's
  /// max2(i,j)*max2(k,l)*max|D| is dropped; 0 disables screening entirely.
  ///
  /// The result is plain host data and carries nothing engine-specific, so one
  /// object screens the conventional J engine, the K engine, or a fused sweep
  /// over both -- and libintx::gpu::make_schwarz_screening's result is
  /// interchangeable with this one. It runs once per geometry, not once per
  /// SCF iteration.
  std::shared_ptr<const libintx::PairScreening> make_schwarz_screening(
    const Basis<Gaussian> &basis,
    float threshold
  );

}

#endif /* LIBINTX_AO_MD_SCREENING_H */
